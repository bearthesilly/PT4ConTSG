"""
Evaluator for ConTSG models.

Provides standardized evaluation pipeline for generated time series,
including metric computation, CLIP embedding extraction, and visualization.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from contsg.config.model_validation import validate_model_config
from contsg.config.schema import ExperimentConfig, EvalConfig
from contsg.registry import Registry
from contsg.utils.progress import smart_tqdm

if TYPE_CHECKING:
    from contsg.eval.visualizer import CaseStudyVisualizer

logger = logging.getLogger(__name__)


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class BatchEvaluationData:
    """Container for single batch evaluation data."""

    pred: Tensor  # (B, L, F) Median prediction
    multi_preds: Tensor  # (n_samples, B, L, F) Multi-sample predictions
    ts: Tensor  # (B, L, F) Ground truth time series
    ts_len: Tensor  # (B,) Sequence lengths
    ts_gen_emb: Optional[Tensor] = None  # (B, D) CLIP embeddings of generated
    ts_gt_emb: Optional[Tensor] = None  # (B, D) CLIP embeddings of ground truth
    cap_emb: Optional[Tensor] = None  # (B, D) Text embeddings
    attrs_idx: Optional[Tensor] = None  # (B, 17) Attribute indices (for segment metrics)


@dataclass
class EvaluationContext:
    """Context for evaluation session."""

    cached_data: Optional[Dict[str, Any]] = None
    all_batches_data: Optional[List[Dict]] = None
    n_samples: int = 10
    sampler: str = "ddim"


# ==============================================================================
# Metric Runner
# ==============================================================================


class MetricRunner:
    """
    Orchestrates metric lifecycle during evaluation.

    Manages metric registration, reset, update, and computation.
    Filters metrics based on CLIP availability.

    Example:
        runner = MetricRunner(["dtw", "fid", "cttp"])
        runner.reset_all()
        for batch in loader:
            runner.update_all(batch_data, clip_available=True)
        results = runner.compute_all()
    """

    def __init__(
        self,
        metric_names: List[str],
        metric_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize metric runner.

        Args:
            metric_names: List of metric names to use
            metric_configs: Optional per-metric configuration overrides
        """
        self.metric_names = metric_names
        self.metric_configs = metric_configs or {}
        self.metrics: Dict[str, Any] = {}
        self._clip_dependent: List[str] = []
        self._clip_independent: List[str] = []

        self._instantiate_metrics()

    def _instantiate_metrics(self) -> None:
        """Instantiate all registered metrics."""
        for name in self.metric_names:
            try:
                metric_cls = Registry.get_metric(name)
                config = self.metric_configs.get(name, {})
                metric = metric_cls(name=name, **config)
                self.metrics[name] = metric

                if metric.requires_clip:
                    self._clip_dependent.append(name)
                else:
                    self._clip_independent.append(name)

                logger.debug(f"Registered metric: {name} (clip_required={metric.requires_clip})")
            except Exception as e:
                logger.warning(f"Failed to instantiate metric '{name}': {e}")

    def reset_all(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()

    def update_all(
        self,
        batch_data: Dict[str, Any],
        clip_available: bool = False,
    ) -> None:
        """
        Update all applicable metrics with batch data.

        Args:
            batch_data: Dictionary containing batch tensors
            clip_available: Whether CLIP embeddings are available
        """
        for name, metric in self.metrics.items():
            if metric.requires_clip and not clip_available:
                continue
            try:
                metric.update(batch_data)
            except Exception as e:
                logger.warning(f"Metric '{name}' update failed: {e}")

    def compute_all(self, clip_available: bool = False) -> Dict[str, float]:
        """
        Compute all metric values.

        Args:
            clip_available: Whether CLIP embeddings were available

        Returns:
            Dictionary mapping metric names to computed values.
            If a metric returns a dict, keys are flattened as "metric_name.subkey".
        """
        results: Dict[str, float] = {}
        for name, metric in self.metrics.items():
            if metric.requires_clip and not clip_available:
                results[name] = float("nan")
                continue
            try:
                value = metric.compute()
                if isinstance(value, dict):
                    # Flatten dict results: metric_name.subkey
                    for subkey, subvalue in value.items():
                        flattened_name = f"{name}.{subkey}"
                        results[flattened_name] = float(subvalue) if subvalue is not None else float("nan")
                else:
                    results[name] = float(value) if value is not None else float("nan")
            except Exception as e:
                logger.warning(f"Metric '{name}' compute failed: {e}")
                results[name] = float("nan")
        return results

    def set_reference_stats(
        self,
        fid_stats: Optional[tuple] = None,
        jftsd_stats: Optional[tuple] = None,
        sd_stats: Optional[np.ndarray] = None,
        kd_stats: Optional[np.ndarray] = None,
        mdd_stats: Optional[tuple] = None,
    ) -> None:
        """
        Inject reference statistics for metrics.

        Args:
            fid_stats: (mean, cov) for FID
            jftsd_stats: (mean, cov) for JFTSD
            sd_stats: (F,) skewness for SD
            kd_stats: (F,) kurtosis for KD
            mdd_stats: (bin_edges, ref_hist, ref_tot) for MDD
        """
        # Frechet metrics
        frechet_mapping = {"fid": fid_stats, "jftsd": jftsd_stats}
        for name, stats in frechet_mapping.items():
            if stats is not None and name in self.metrics:
                metric = self.metrics[name]
                if hasattr(metric, "set_reference"):
                    metric.set_reference(stats[0], stats[1])
                    logger.debug(f"Set reference stats for {name}")

        # SD metric
        if sd_stats is not None and "sd" in self.metrics:
            self.metrics["sd"].set_reference_stats(sd_stats)
            logger.debug("Set reference skewness for sd")

        # KD metric
        if kd_stats is not None and "kd" in self.metrics:
            self.metrics["kd"].set_reference_stats(kd_stats)
            logger.debug("Set reference kurtosis for kd")

        # MDD metric
        if mdd_stats is not None and "mdd" in self.metrics:
            self.metrics["mdd"].set_reference(mdd_stats[0], mdd_stats[1], mdd_stats[2])
            logger.debug("Set reference histogram for mdd")

    def get_clip_dependent(self) -> List[str]:
        """Get list of CLIP-dependent metric names."""
        return self._clip_dependent.copy()

    def get_clip_independent(self) -> List[str]:
        """Get list of CLIP-independent metric names."""
        return self._clip_independent.copy()


# ==============================================================================
# Evaluator
# ==============================================================================


class Evaluator:
    """
    Full-featured evaluator for conditional time series generation.

    Handles:
    - Sample generation from trained models
    - CLIP embedding extraction (optional)
    - Metric computation with reference statistics
    - Result aggregation and reporting

    Example:
        evaluator = Evaluator.from_experiment("experiments/exp1/")
        results = evaluator.evaluate(n_samples=10, sampler="ddpm")

    Legacy source: base_evaluator.py
    """

    def __init__(
        self,
        model: Optional[Any],
        config: ExperimentConfig,
        datamodule: Any,
        device: Optional[torch.device] = None,
        cache_only_mode: bool = False,
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained model (LightningModule), None if cache_only_mode
            config: Experiment configuration
            datamodule: Data module for test data
            device: Torch device for evaluation
            cache_only_mode: If True, skip model operations (requires valid cache)
        """
        self.model = model
        self.config = config
        self.datamodule = datamodule
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.cache_only_mode = cache_only_mode

        # Move model to device and set eval mode (skip if cache_only_mode)
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()

        # Optional CLIP components
        self.embedder: Optional[Any] = None
        self.stats_cache: Optional[Any] = None
        self.clip_available = False

        # Metric runner (initialized later with metrics list)
        self.metric_runner: Optional[MetricRunner] = None

        # Optional visualizer (initialized in evaluate if enabled)
        self.visualizer: Optional["CaseStudyVisualizer"] = None

    @classmethod
    def from_experiment(
        cls,
        experiment_dir: Union[Path, str],
        checkpoint: str = "best",
        device: Optional[torch.device] = None,
        cache_only: bool = False,
    ) -> "Evaluator":
        """
        Load evaluator from experiment directory.

        Args:
            experiment_dir: Path to experiment directory
            checkpoint: Checkpoint to load ("best", "last", or filename)
            device: Torch device for evaluation
            cache_only: If True, skip model loading (requires valid cache)

        Returns:
            Evaluator instance
        """
        experiment_dir = Path(experiment_dir)

        # Load config
        config = ExperimentConfig.from_yaml(experiment_dir / "config.yaml")

        # Keep model schema validation behavior consistent with train path.
        config = validate_model_config(config, strict_schema=False).config

        # Dataset registration is still needed in cache-only mode.
        import contsg.data.datasets  # noqa: F401

        # Load model (skip if cache_only)
        model = None
        if not cache_only:
            # Load model registrations only when a generator instance is required.
            import contsg.models  # noqa: F401

            # Determine checkpoint path
            ckpt_path = None
            if checkpoint == "best":
                summary_path = experiment_dir / "summary.json"
                if summary_path.exists():
                    with open(summary_path) as f:
                        summary = json.load(f)
                        ckpt_path_str = summary.get("best_checkpoint")
                        if ckpt_path_str:
                            ckpt_path = Path(ckpt_path_str)
            elif checkpoint == "last":
                ckpt_path = experiment_dir / "checkpoints" / "last.ckpt"
            else:
                ckpt_path = experiment_dir / "checkpoints" / checkpoint

            if not ckpt_path or not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            # Load model
            model_cls = Registry.get_model(config.model.name)
            model = model_cls.load_from_checkpoint(ckpt_path, config=config)
            logger.info(f"Loaded model from checkpoint: {ckpt_path}")
        else:
            logger.info("Cache-only mode: skipping model loading")

        # Create datamodule via registry to support custom datasets
        dataset_cls = Registry.get_dataset(config.data.name)
        datamodule = dataset_cls(
            config.data,
            train_config={
                "batch_size": config.eval.batch_size,
                "num_workers": config.train.num_workers,
                "pin_memory": config.train.pin_memory,
                "model_name": config.model.name,
                "condition": config.condition,
                "seed": config.seed,
            },
        )
        datamodule.setup("test")

        return cls(model, config, datamodule, device, cache_only_mode=cache_only)

    def init_clip(
        self,
        clip_config_path: Optional[Union[Path, str]] = None,
        clip_model_path: Optional[Union[Path, str]] = None,
        cache_folder: Optional[Union[Path, str]] = None,
        use_longalign: bool = False,
    ) -> bool:
        """
        Initialize CLIP embedder and statistics cache.

        Args:
            clip_config_path: Path to CLIP config YAML
            clip_model_path: Path to CLIP model checkpoint
            cache_folder: Path to statistics cache folder
            use_longalign: Whether to use LongAlign mode

        Returns:
            True if CLIP initialized successfully, False otherwise
        """
        # Use config values if not provided
        eval_cfg = self.config.eval
        clip_config_path = clip_config_path or getattr(eval_cfg, "clip_config_path", None)
        clip_model_path = clip_model_path or getattr(eval_cfg, "clip_model_path", None)
        cache_folder = cache_folder or getattr(eval_cfg, "cache_folder", None)
        use_longalign = use_longalign or getattr(eval_cfg, "use_longalign", False)
        normalize_embeddings = getattr(eval_cfg, "clip_normalize_embeddings", None)
        if normalize_embeddings is None:
            logger.info("CLIP embedder: using model embedding normalization setting")
        else:
            logger.info(
                "CLIP embedder: override embedding normalization to %s",
                normalize_embeddings,
            )

        if clip_config_path is None or clip_model_path is None:
            logger.info("CLIP config/model paths not provided, skipping CLIP initialization")
            return False

        try:
            from contsg.eval.embedder import CLIPEmbedder
            from contsg.eval.stats_cache import StatsCache

            self.embedder = CLIPEmbedder(
                clip_config_path=clip_config_path,
                clip_model_path=clip_model_path,
                device=self.device,
                use_longalign=use_longalign,
                normalize_embeddings=normalize_embeddings,
            )

            if cache_folder:
                self.stats_cache = StatsCache(
                    cache_folder=Path(cache_folder),
                    embedder=self.embedder,
                    use_longalign=use_longalign,
                    device=self.device,
                )
                # Load or compute stats from training set
                train_loader = self.datamodule.train_dataloader() if getattr(self.datamodule, "train_dataset", None) is not None else (self.datamodule.setup("fit") or self.datamodule.train_dataloader())
                self.stats_cache.load_or_compute(train_loader)

            self.clip_available = True
            logger.info("CLIP embedder initialized successfully")
            return True

        except FileNotFoundError as e:
            logger.warning(f"CLIP files not found: {e}")
            logger.warning("CLIP-dependent metrics will be skipped")
            return False
        except Exception as e:
            logger.warning(f"CLIP initialization failed: {e}")
            return False

    # ==========================================================================
    # Prediction Caching
    # ==========================================================================

    def _load_predictions_cache(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load predictions from cache file.

        Args:
            cache_path: Path to cache file (.pkl)

        Returns:
            Cached data dict or None if load fails
        """
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            logger.info(f"Loaded predictions cache from {cache_path}")
            return cached
        except Exception as e:
            logger.warning(f"Failed to load predictions cache: {e}")
            return None

    def _save_predictions_cache(
        self,
        cache_path: Path,
        all_batches_data: List[Dict[str, Any]],
        n_samples: int,
        sampler: str,
    ) -> None:
        """
        Save predictions to cache file.

        Args:
            cache_path: Path to cache file
            all_batches_data: List of batch data dicts
            n_samples: Number of samples used
            sampler: Sampler name
        """
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if CLIP embeddings are present in batches
        has_clip_embeddings = False
        if all_batches_data:
            first_batch = all_batches_data[0]
            has_clip_embeddings = any(
                key in first_batch 
                for key in ["ts_gt_emb", "ts_gen_emb", "cap_emb"]
            )
        
        cache_data = {
            "all_batches_data": all_batches_data,
            "n_samples": n_samples,
            "sampler": sampler,
            "has_clip_embeddings": has_clip_embeddings,
            "model_name": self.config.model.name,
            "dataset_name": self.config.data.name,
            "dataset_size": len(self.datamodule.test_dataloader().dataset) if hasattr(self.datamodule, "test_dataloader") else 0,
        }
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            logger.info(f"Saved predictions cache to {cache_path} (CLIP embeddings: {has_clip_embeddings})")
        except Exception as e:
            logger.warning(f"Failed to save predictions cache: {e}")

    def _validate_cache(
        self,
        cached: Dict[str, Any],
        n_samples: int,
        sampler: str,
    ) -> bool:
        """
        Validate cached predictions match current settings.

        Args:
            cached: Cached data dict
            n_samples: Expected n_samples
            sampler: Expected sampler

        Returns:
            True if cache is valid
        """
        if cached.get("n_samples") != n_samples:
            logger.warning(
                f"Cache n_samples mismatch: {cached.get('n_samples')} vs {n_samples}"
            )
            return False
        if cached.get("sampler") != sampler:
            logger.warning(
                f"Cache sampler mismatch: {cached.get('sampler')} vs {sampler}"
            )
            return False
        if not cached.get("all_batches_data"):
            logger.warning("Cache has no batch data")
            return False
        
        # Validate model/dataset match (optional, with fallback for old caches)
        if "model_name" in cached and cached["model_name"] != self.config.model.name:
            logger.warning(
                f"Cache model mismatch: {cached['model_name']} vs {self.config.model.name}"
            )
            return False
        if "dataset_name" in cached and cached["dataset_name"] != self.config.data.name:
            logger.warning(
                f"Cache dataset mismatch: {cached['dataset_name']} vs {self.config.data.name}"
            )
            return False
        
        return True

    def evaluate(
        self,
        metrics: Optional[List[str]] = None,
        n_samples: Optional[int] = None,
        sampler: Optional[str] = None,
        metric_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        use_cache: bool = False,
        cache_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation.

        Args:
            metrics: List of metrics to compute (default: from config)
            n_samples: Number of samples per condition (default: from config)
            sampler: Sampling method ("ddpm", "ddim", etc.); defaults to config
            metric_configs: Optional per-metric configuration
            use_cache: Whether to use/save prediction cache
            cache_path: Path to cache file (default: results/predictions_cache.pkl)

        Returns:
            Dictionary of metric results
        """
        metrics = metrics or self.config.eval.metrics
        n_samples = n_samples or self.config.eval.n_samples
        sampler = sampler or self.config.eval.sampler
        
        # Validate cache_only_mode requirements
        if self.cache_only_mode:
            if not use_cache:
                raise ValueError("cache_only_mode requires use_cache=True")
            if not cache_path or not cache_path.exists():
                raise FileNotFoundError(
                    f"cache_only_mode requires valid cache file. Not found: {cache_path}"
                )
            # Pre-validate cache
            cached_data = self._load_predictions_cache(cache_path)
            if not cached_data:
                raise ValueError("Failed to load cache file")
            if not self._validate_cache(cached_data, n_samples, sampler):
                raise ValueError(
                    f"Cache validation failed. Expected n_samples={n_samples}, sampler={sampler}"
                )
            logger.info("Cache-only mode: cache validated successfully")

        # Initialize metric runner
        self.metric_runner = MetricRunner(metrics, metric_configs)
        self.metric_runner.reset_all()

        # Set reference statistics if available
        if self.stats_cache is not None:
            self.metric_runner.set_reference_stats(
                fid_stats=self.stats_cache.fid_stats,
                jftsd_stats=self.stats_cache.jftsd_stats,
                sd_stats=self.stats_cache.sd_stats,
                kd_stats=self.stats_cache.kd_stats,
                mdd_stats=self.stats_cache.mdd_stats,
            )

        # Initialize visualizer if enabled
        test_loader = self.datamodule.test_dataloader()
        if self.config.eval.viz.enable:
            from contsg.eval.visualizer import CaseStudyVisualizer
            self.visualizer = CaseStudyVisualizer(self.config.eval.viz)
            self.visualizer.select_samples(len(test_loader.dataset), n_samples)
        else:
            self.visualizer = None

        # Handle caching
        cached_data: Optional[Dict[str, Any]] = None
        if use_cache and cache_path and cache_path.exists():
            cached_data = self._load_predictions_cache(cache_path)
            if cached_data and not self._validate_cache(cached_data, n_samples, sampler):
                cached_data = None  # Invalid cache, regenerate

        # Run evaluation loop
        all_batches_data = self._run_evaluation_loop(
            test_loader, n_samples, sampler, cached_data
        )

        # Save cache if generated new predictions
        if use_cache and cache_path and cached_data is None and all_batches_data:
            self._save_predictions_cache(cache_path, all_batches_data, n_samples, sampler)

        # Compute final results
        results = self.metric_runner.compute_all(clip_available=self.clip_available)

        # Render case study visualizations if enabled
        if self.visualizer is not None:
            output_folder = cache_path.parent if cache_path else Path("results")
            self.visualizer.render(output_folder / "case_study")

        logger.info("Evaluation complete:")
        for name, value in results.items():
            logger.info(f"  {name}: {value:.6f}")

        return results

    def _run_evaluation_loop(
        self,
        test_loader: DataLoader,
        n_samples: int,
        sampler: str,
        cached_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run evaluation loop over all batches.

        Args:
            test_loader: Test data loader
            n_samples: Number of samples per condition
            sampler: Sampling method
            cached_data: Optional cached predictions

        Returns:
            List of batch data dicts (for caching) or empty list if using cache
        """
        all_batches_data: List[Dict[str, Any]] = []
        using_cache = cached_data is not None

        if using_cache:
            data_source = cached_data["all_batches_data"]
            desc = "Evaluating (using cache)"
            total_batches = len(data_source)
        else:
            data_source = test_loader
            desc = "Evaluating (generating)"
            total_batches = len(test_loader)

        logger.info(f"Starting evaluation: {total_batches} batches, {n_samples} samples/condition")

        seen_so_far = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(smart_tqdm(data_source, desc=desc)):
                start_time = time.time()

                if using_cache:
                    # Restore cached predictions
                    batch_data = self._process_cached_batch(batch)
                else:
                    # Generate new predictions
                    batch_data, cache_item = self._process_batch_with_cache(
                        batch, n_samples, sampler
                    )
                    all_batches_data.append(cache_item)

                # Update metrics
                self.metric_runner.update_all(batch_data, clip_available=self.clip_available)

                # Collect visualization cases
                if self.visualizer is not None:
                    self.visualizer.collect(batch_data, batch_data["multi_preds"], seen_so_far)

                # Update counter for global IDs
                batch_size = batch_data["ts"].shape[0]
                seen_so_far += batch_size

                elapsed = time.time() - start_time
                if (batch_idx + 1) % 10 == 0:
                    logger.debug(f"Batch {batch_idx + 1}/{total_batches} ({elapsed:.2f}s)")

        return all_batches_data

    def _process_cached_batch(
        self,
        cached_batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process a cached batch for evaluation.

        Args:
            cached_batch: Cached batch data dict

        Returns:
            Dictionary with processed batch data for metrics
        """
        # Restore tensors to device
        multi_preds = cached_batch["multi_preds"].to(self.device)
        ts = cached_batch["ts"].to(self.device)
        ts_len = cached_batch["ts_len"].to(self.device)

        # Compute median prediction
        pred = multi_preds.median(dim=0).values

        # Build batch data dictionary
        batch_data: Dict[str, Any] = {
            "pred": pred,
            "multi_preds": multi_preds,
            "ts": ts,
            "ts_len": ts_len,
        }

        # Add CLIP embeddings if available
        # First try to load from cache, fallback to dynamic computation (for old caches)
        if "ts_gt_emb" in cached_batch and cached_batch["ts_gt_emb"] is not None:
            # Load CLIP embeddings from cache
            batch_data["ts_gt_emb"] = cached_batch["ts_gt_emb"].to(self.device)
            batch_data["ts_gen_emb"] = cached_batch["ts_gen_emb"].to(self.device)
            if "cap_emb" in cached_batch and cached_batch["cap_emb"] is not None:
                batch_data["cap_emb"] = cached_batch["cap_emb"].to(self.device)
            logger.debug("Loaded CLIP embeddings from cache")
        elif self.clip_available and self.embedder is not None:
            # Fallback: compute CLIP embeddings dynamically (for old caches without CLIP)
            ts_gt_emb = self.embedder.get_ts_embedding(ts, ts_len)
            ts_gen_emb = self.embedder.get_ts_embedding(pred, ts_len)
            # Reconstruct batch for text embedding
            text_batch = {}
            if "cap" in cached_batch:
                text_batch["cap"] = cached_batch["cap"]
            if "caps" in cached_batch:
                text_batch["caps"] = cached_batch["caps"]
            if text_batch:
                cap_emb = self.embedder.get_text_embedding(text_batch)
                batch_data["cap_emb"] = cap_emb
            batch_data["ts_gt_emb"] = ts_gt_emb
            batch_data["ts_gen_emb"] = ts_gen_emb
            logger.debug("Computed CLIP embeddings dynamically (old cache format)")

        # Add optional fields
        if "attrs_idx" in cached_batch:
            batch_data["attrs_idx"] = cached_batch["attrs_idx"].to(self.device)
        if "attrs" in cached_batch:
            batch_data["attrs"] = cached_batch["attrs"].to(self.device)
        if "label" in cached_batch:
            batch_data["label"] = cached_batch["label"].to(self.device)
        if "labels" in cached_batch:
            batch_data["labels"] = cached_batch["labels"].to(self.device)

        return batch_data

    def _process_batch_with_cache(
        self,
        batch: Dict[str, Any],
        n_samples: int,
        sampler: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process a batch and prepare cache data.

        Args:
            batch: Raw batch from dataloader
            n_samples: Number of samples to generate
            sampler: Sampling method

        Returns:
            Tuple of (batch_data for metrics, cache_item for saving)
        """
        # Move batch to device
        batch = self._move_batch_to_device(batch)

        # Generate predictions
        multi_preds = self._generate_predictions(batch, n_samples, sampler)

        # Compute median prediction
        pred = multi_preds.median(dim=0).values

        # Extract core tensors
        ts = batch["ts"]
        # ts_len may not be provided; infer from ts shape if missing
        if "ts_len" in batch:
            ts_len = batch["ts_len"]
        else:
            # Assume all sequences have the same length (no padding)
            ts_len = torch.full((ts.shape[0],), ts.shape[1], device=ts.device, dtype=torch.long)

        # Build batch data dictionary
        batch_data: Dict[str, Any] = {
            "pred": pred,
            "multi_preds": multi_preds,
            "ts": ts,
            "ts_len": ts_len,
        }

        # Build cache item (CPU tensors)
        cache_item: Dict[str, Any] = {
            "multi_preds": multi_preds.cpu(),
            "ts": ts.cpu(),
            "ts_len": ts_len.cpu(),
        }

        # Add text captions for cache
        if "cap" in batch:
            cache_item["cap"] = batch["cap"]
        if "caps" in batch:
            cache_item["caps"] = batch["caps"]

        # Add optional fields
        if "attrs_idx" in batch:
            batch_data["attrs_idx"] = batch["attrs_idx"]
            cache_item["attrs_idx"] = batch["attrs_idx"].cpu()
        if "attrs" in batch:
            batch_data["attrs"] = batch["attrs"]
            cache_item["attrs"] = batch["attrs"].cpu()
        if "label" in batch:
            batch_data["label"] = batch["label"]
            cache_item["label"] = batch["label"].cpu()
        if "labels" in batch:
            batch_data["labels"] = batch["labels"]
            cache_item["labels"] = batch["labels"].cpu()

        # Add CLIP embeddings if available
        if self.clip_available and self.embedder is not None:
            ts_gt_emb = self.embedder.get_ts_embedding(ts, ts_len)
            ts_gen_emb = self.embedder.get_ts_embedding(pred, ts_len)
            cap_emb = self.embedder.get_text_embedding(batch)

            batch_data["ts_gt_emb"] = ts_gt_emb
            batch_data["ts_gen_emb"] = ts_gen_emb
            batch_data["cap_emb"] = cap_emb
            
            # Cache CLIP embeddings (CPU tensors)
            cache_item["ts_gt_emb"] = ts_gt_emb.cpu()
            cache_item["ts_gen_emb"] = ts_gen_emb.cpu()
            cache_item["cap_emb"] = cap_emb.cpu()

        return batch_data, cache_item

    def _process_batch(
        self,
        batch: Dict[str, Any],
        n_samples: int,
        sampler: str,
    ) -> Dict[str, Any]:
        """
        Process a single batch for evaluation.

        Args:
            batch: Raw batch from dataloader
            n_samples: Number of samples to generate
            sampler: Sampling method

        Returns:
            Dictionary with processed batch data for metrics
        """
        # Move batch to device
        batch = self._move_batch_to_device(batch)

        # Generate predictions
        multi_preds = self._generate_predictions(batch, n_samples, sampler)

        # Compute median prediction
        pred = multi_preds.median(dim=0).values

        # Extract core tensors
        ts = batch["ts"]
        # ts_len may not be provided; infer from ts shape if missing
        if "ts_len" in batch:
            ts_len = batch["ts_len"]
        else:
            # Assume all sequences have the same length (no padding)
            ts_len = torch.full((ts.shape[0],), ts.shape[1], device=ts.device, dtype=torch.long)

        # Build batch data dictionary
        batch_data: Dict[str, Any] = {
            "pred": pred,
            "multi_preds": multi_preds,
            "ts": ts,
            "ts_len": ts_len,
        }

        # Add CLIP embeddings if available
        if self.clip_available and self.embedder is not None:
            ts_gt_emb = self.embedder.get_ts_embedding(ts, ts_len)
            ts_gen_emb = self.embedder.get_ts_embedding(pred, ts_len)
            cap_emb = self.embedder.get_text_embedding(batch)

            batch_data["ts_gt_emb"] = ts_gt_emb
            batch_data["ts_gen_emb"] = ts_gen_emb
            batch_data["cap_emb"] = cap_emb

        # Add optional fields
        if "attrs_idx" in batch:
            batch_data["attrs_idx"] = batch["attrs_idx"]

        return batch_data

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        moved = {}
        for key, value in batch.items():
            if isinstance(value, Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _generate_predictions(
        self,
        batch: Dict[str, Any],
        n_samples: int,
        sampler: str,
    ) -> Tensor:
        """
        Generate predictions from model.

        Args:
            batch: Batch dictionary
            n_samples: Number of samples
            sampler: Sampling method

        Returns:
            Multi-sample predictions (n_samples, B, L, F)
        """
        # Sanity check: should not be called in cache_only_mode
        if self.cache_only_mode or self.model is None:
            raise RuntimeError(
                "Cannot generate predictions in cache_only_mode or when model is None. "
                "This is likely a bug - check that cache is properly loaded."
            )
        
        # Handle Bridge model's need for example time series
        if hasattr(self.model, "needs_bridge_example") and self.model.needs_bridge_example:
            try:
                if getattr(self.datamodule, "train_dataset", None) is None:
                    self.datamodule.setup("fit")
                train_loader = self.datamodule.train_dataloader()
            except TypeError as exc:
                if "NoneType" not in str(exc):
                    raise
                self.datamodule.setup("fit")
                train_loader = self.datamodule.train_dataloader()
            train_batch = next(iter(train_loader))
            current_batch_size = batch["ts"].shape[0]
            train_batch_size = train_batch["ts"].shape[0]

            if train_batch_size >= current_batch_size:
                batch["bridge_example_ts"] = train_batch["ts"][:current_batch_size].to(self.device)
            else:
                repeat_times = (current_batch_size + train_batch_size - 1) // train_batch_size
                repeated = train_batch["ts"].repeat(repeat_times, 1, 1)
                batch["bridge_example_ts"] = repeated[:current_batch_size].to(self.device)

        # Extract condition from batch
        # The model.generate() expects condition tensor, not batch dict
        condition = self._select_condition(batch)
        if condition is None and getattr(self.model, "use_condition", True):
            raise ValueError(
                "No valid condition found for generation. "
                "Check config.condition and batch keys."
            )

        # Pass additional batch info through kwargs
        gen_kwargs = {"tp": batch.get("tp"), "sampler": sampler}
        if "attrs" in batch:
            gen_kwargs["attrs"] = batch["attrs"]
        if "bridge_example_ts" in batch:
            gen_kwargs["bridge_example_ts"] = batch["bridge_example_ts"]

        # Generate samples
        multi_preds = self.model.generate(condition, n_samples, **gen_kwargs)

        # Normalize to (n_samples, B, L, F) for downstream metrics
        return self._normalize_multi_preds(multi_preds, batch, n_samples)

    def _normalize_multi_preds(
        self,
        multi_preds: Tensor,
        batch: Dict[str, Any],
        n_samples: int,
    ) -> Tensor:
        """Normalize generated outputs to (n_samples, B, L, F)."""
        ts = batch["ts"]
        if ts.dim() != 3:
            raise ValueError(f"Expected batch['ts'] to be 3D (B, L, F), got {ts.shape}")
        B, L, F = ts.shape

        preds = multi_preds
        if preds.dim() == 3:
            if preds.shape[0] == B and n_samples == 1:
                preds = preds.unsqueeze(0)
            elif preds.shape[0] == B * n_samples:
                preds = preds.reshape(B, n_samples, preds.shape[1], preds.shape[2])
                preds = preds.permute(1, 0, 2, 3)
            else:
                raise ValueError(
                    f"Unexpected predictions shape {preds.shape} for n_samples={n_samples}, batch={B}"
                )
        elif preds.dim() == 4:
            if preds.shape[0] == B and preds.shape[1] == n_samples:
                preds = preds.permute(1, 0, 2, 3)
            elif preds.shape[0] == n_samples and preds.shape[1] == B:
                pass
            else:
                raise ValueError(
                    f"Unexpected predictions shape {preds.shape} for n_samples={n_samples}, batch={B}"
                )
        else:
            raise ValueError(f"Unsupported predictions rank: {preds.dim()} for shape {preds.shape}")

        if preds.shape[2] == L and preds.shape[3] == F:
            return preds
        if preds.shape[2] == F and preds.shape[3] == L:
            return preds.permute(0, 1, 3, 2)

        raise ValueError(
            f"Predictions shape {preds.shape} is incompatible with ts shape {ts.shape}"
        )

    def _select_condition(self, batch: Dict[str, Any]) -> Optional[Tensor]:
        """Select the appropriate condition tensor based on config and batch.

        When both text and attribute are enabled, text (cap_emb) is returned as
        the primary condition tensor; attrs are passed via gen_kwargs separately
        so that the model's generate() can fuse them internally.
        """
        if not getattr(self.model, "use_condition", True):
            return None

        cond_cfg = self.config.condition

        # When both text+attr enabled, return text as primary (attrs via kwargs)
        if cond_cfg.text.enabled:
            if "cap_emb" in batch:
                return batch["cap_emb"]
            raise ValueError(
                "Text condition enabled but no 'cap_emb' found in batch. "
                "Provide precomputed embeddings in the dataset."
            )

        if cond_cfg.attribute.enabled:
            if "attrs" in batch:
                return batch["attrs"]
            if "attrs_discrete" in batch:
                return batch["attrs_discrete"]
            if "attrs_continuous" in batch:
                return batch["attrs_continuous"]
            raise ValueError(
                "Attribute condition enabled but no attribute keys found in batch."
            )

        if cond_cfg.label.enabled:
            if "label" in batch:
                return batch["label"]
            if "labels" in batch:
                return batch["labels"]
            raise ValueError(
                "Label condition enabled but no 'label(s)' found in batch."
            )

        return None

    def _generate_samples(self, n_samples: int) -> Dict[str, Tensor]:
        """
        Generate samples from the model (legacy interface).

        Args:
            n_samples: Number of samples per condition

        Returns:
            Dictionary with generated, conditions, and real tensors
        """
        all_generated = []
        all_conditions = []
        all_real = []

        test_loader = self.datamodule.test_dataloader()

        with torch.no_grad():
            for batch in test_loader:
                batch = self._move_batch_to_device(batch)
                cond = self._select_condition(batch)
                generated = self.model.generate(cond, n_samples=n_samples)
                all_generated.append(generated.cpu())
                all_conditions.append(cond.cpu() if isinstance(cond, Tensor) else cond)
                all_real.append(batch["ts"].cpu())

        return {
            "generated": torch.cat(all_generated, dim=0),
            "conditions": torch.cat([c for c in all_conditions if isinstance(c, Tensor)], dim=0),
            "real": torch.cat(all_real, dim=0),
        }


# ==============================================================================
# Exports
# ==============================================================================


__all__ = [
    "BatchEvaluationData",
    "EvaluationContext",
    "MetricRunner",
    "Evaluator",
]
