"""
CLIP Embedder for evaluation metrics.

This module provides a wrapper around CTTPModel for extracting time series
and text embeddings used by CLIP-dependent metrics (FID, CTTP, JFTSD, etc.).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
import yaml

logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """
    CLIP embedder wrapper for evaluation.

    Wraps CTTPModel to provide consistent embedding extraction for metrics.
    Only supports online text encoding for CLIP-based evaluation.

    Features:
    - Single-segment and LongAlign (multi-segment averaged) text modes
    - Legacy checkpoint conversion support
    - Device management

    Example:
        embedder = CLIPEmbedder(eval_config, device)
        ts_emb = embedder.get_ts_embedding(ts, ts_len)
        text_emb = embedder.get_text_embedding(batch)
    """

    def __init__(
        self,
        clip_config_path: Union[str, Path],
        clip_model_path: Union[str, Path],
        device: torch.device,
        use_longalign: bool = False,
        normalize_embeddings: Optional[bool] = None,
    ):
        """
        Initialize CLIP embedder.

        Args:
            clip_config_path: Path to CLIP/CTTP config YAML
            clip_model_path: Path to CTTP model checkpoint
            device: Torch device for inference
            use_longalign: Whether to use LongAlign (multi-segment averaged) mode

        Raises:
            FileNotFoundError: If config or model files don't exist
            ImportError: If CTTPModel is not available
        """
        self.device = device
        self.use_longalign = use_longalign
        self.normalize_embeddings = normalize_embeddings

        clip_config_path = Path(clip_config_path)
        clip_model_path = Path(clip_model_path)

        if not clip_config_path.exists():
            raise FileNotFoundError(
                f"CLIP config file not found: {clip_config_path}\n"
                f"Please train CTTP model first or provide valid config path."
            )

        if not clip_model_path.exists():
            raise FileNotFoundError(
                f"CLIP model checkpoint not found: {clip_model_path}\n"
                f"Please train CTTP model first or provide valid checkpoint path."
            )

        self._load_model(clip_config_path, clip_model_path)
        if getattr(self.model, "text_encoding", None) != "online":
            raise ValueError(
                "CLIPEmbedder requires a CTTP model with text_encoding='online' "
                "so that embeddings are computed by the CLIP text encoder."
            )

    def _load_model(self, config_path: Path, model_path: Path) -> None:
        """
        Load CTTP model from checkpoint.

        Supports both legacy VerbalTS format and new contsg ExperimentConfig format.
        """
        # Load config
        with open(config_path) as f:
            clip_configs = yaml.safe_load(f)

        # Try to load using new contsg CTTPModel
        try:
            from contsg.models.cttp import CTTPModel

            # Detect config format and load accordingly
            if isinstance(clip_configs, dict):
                # Check if it's new ExperimentConfig format (has "model" with name="cttp")
                model_cfg = clip_configs.get("model", {})
                if isinstance(model_cfg, dict) and model_cfg.get("name") == "cttp":
                    # New contsg ExperimentConfig format
                    logger.info("Detected new ExperimentConfig format for CTTP")
                    self.model = self._load_new_format_checkpoint(
                        clip_configs, model_path
                    )
                elif "ts" in clip_configs or "text" in clip_configs:
                    # Legacy dict format with ts/text keys
                    logger.info("Detected legacy dict format for CTTP")
                    self.model = self._load_legacy_checkpoint(
                        clip_configs, model_path
                    )
                else:
                    raise ValueError(
                        f"Unrecognized CTTP config format. Expected either:\n"
                        f"  1) New format: config with 'model.name: cttp'\n"
                        f"  2) Legacy format: config with 'ts' and 'text' keys\n"
                        f"Got keys: {list(clip_configs.keys())}"
                    )
            else:
                raise ValueError(
                    f"CTTP config must be a dict, got {type(clip_configs)}"
                )

            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("CTTPModel loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load CTTPModel: {e}")
            raise

    def _load_new_format_checkpoint(
        self, clip_configs: Dict[str, Any], model_path: Path
    ) -> Any:
        """
        Load CTTP model from new contsg ExperimentConfig format.

        This handles the full ExperimentConfig structure where model parameters
        are under the 'model' key instead of 'ts'/'text' keys.
        """
        from contsg.models.cttp import CTTPModel

        model_cfg = clip_configs.get("model", {})
        data_cfg = clip_configs.get("data", {})
        condition_cfg = clip_configs.get("condition", {})

        # Build a minimal config object for CTTPModel initialization
        class MinimalConfig:
            """Minimal config wrapper for new format compatibility."""

            def __init__(self, cfg_dict: Dict):
                # Model config - copy all model parameters
                model_attrs = cfg_dict.get("model", {}).copy()
                self.model = type("ModelConfig", (), model_attrs)()

                # Data config
                data_attrs = cfg_dict.get("data", {})
                self.data = type(
                    "DataConfig",
                    (),
                    {
                        "seq_length": data_attrs.get("seq_length", 512),
                        "n_var": data_attrs.get("n_var", 1),
                    },
                )()

                # Condition config
                cond_attrs = cfg_dict.get("condition", {})
                text_attrs = cond_attrs.get("text", {})
                self.condition = type(
                    "CondConfig",
                    (),
                    {
                        "mode": "text" if text_attrs.get("enabled", False) else "none",
                        "text": type("TextConfig", (), {
                            "input_dim": text_attrs.get("input_dim", 768),
                        })(),
                        "text_model_path": model_attrs.get("pretrain_model_path", ""),
                    },
                )()

        minimal_config = MinimalConfig(clip_configs)

        # Determine mode from config
        mode = model_cfg.get("mode", "instance")

        # Determine text_encoding from pretrain_model_path
        text_encoding = "online" if model_cfg.get("pretrain_model_path") else "precomputed"

        # Create model
        model = CTTPModel(
            config=minimal_config,
            mode=mode,
            text_encoding=text_encoding,
        )

        # Load state dict
        state_dict = torch.load(model_path, map_location="cpu")

        # Handle Lightning checkpoint format
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Try direct load first
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info("Loaded CTTP weights with strict=True")
        except RuntimeError as e:
            logger.warning(f"Strict loading failed ({e}), trying strict=False")
            model.load_state_dict(state_dict, strict=False)

        return model

    def _load_legacy_checkpoint(
        self, clip_configs: Dict[str, Any], model_path: Path
    ) -> Any:
        """
        Load legacy VerbalTS CTTP checkpoint.

        Handles weight mapping from legacy nn.Module to new LightningModule.
        """
        from contsg.models.cttp import CTTPModel

        # Extract necessary parameters from legacy config
        ts_cfg = clip_configs.get("ts", {})
        text_cfg = clip_configs.get("text", {})

        # Build a minimal config object for initialization
        class MinimalConfig:
            """Minimal config wrapper for legacy compatibility."""

            def __init__(self, cfg_dict: Dict):
                self.model = type("ModelConfig", (), cfg_dict.get("model", cfg_dict))()
                self.data = type(
                    "DataConfig",
                    (),
                    {
                        "seq_length": ts_cfg.get("seq_len", 512),  # CTTPModel uses seq_length
                        "n_var": ts_cfg.get("n_var", 1),
                    },
                )()
                self.condition = type(
                    "CondConfig",
                    (),
                    {
                        "mode": "text",
                        "text": type("TextConfig", (), {
                            "input_dim": text_cfg.get("pretrain_model_dim", 768),
                        })(),
                        "text_model_path": text_cfg.get("pretrain_model_path", ""),
                    },
                )()

                # Copy all ts config to model
                for k, v in ts_cfg.items():
                    if not hasattr(self.model, k):
                        setattr(self.model, k, v)
                for k, v in text_cfg.items():
                    if not hasattr(self.model, k):
                        setattr(self.model, k, v)

        minimal_config = MinimalConfig(clip_configs)

        # Determine mode from legacy config
        # Priority: 1) explicit "mode" field, 2) infer from "clip_type" field
        clip_type = clip_configs.get("clip_type", "")
        if "segment" in clip_type.lower():
            mode = "segment"
            logger.info(f"Detected segment mode from clip_type='{clip_type}'")
        else:
            mode = clip_configs.get("mode", "instance")

        text_encoding = "online" if "pretrain_model_path" in text_cfg else "precomputed"

        # Create model
        model = CTTPModel(
            config=minimal_config,
            mode=mode,
            text_encoding=text_encoding,
        )

        # Load state dict with potential key mapping
        state_dict = torch.load(model_path, map_location="cpu")

        # Handle Lightning checkpoint format
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Try direct load first
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # Try with key mapping for legacy format
            logger.info("Attempting legacy weight conversion...")
            converted_state = self._convert_legacy_weights(state_dict)
            model.load_state_dict(converted_state, strict=False)

        return model

    def _convert_legacy_weights(
        self, state_dict: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        Convert legacy CTTP weights to new format.

        Maps keys from VerbalTS nn.Module to contsg LightningModule.
        """
        converted = {}
        for key, value in state_dict.items():
            new_key = key
            # Common mappings
            if key.startswith("text_enc."):
                # text_enc.text_enc.* -> text_enc.*
                new_key = key.replace("text_enc.text_enc.", "text_enc.text_enc.")
            elif key.startswith("ts_enc."):
                # Keep ts_enc mappings
                new_key = key

            converted[new_key] = value

        return converted

    @torch.no_grad()
    def get_ts_embedding(self, ts: Tensor, ts_len: Optional[Tensor] = None) -> Tensor:
        """
        Get time series embedding.

        Args:
            ts: (B, L, C) time series tensor
            ts_len: (B,) sequence lengths (optional, for masking)

        Returns:
            (B, D) global time series embeddings
        """
        ts = ts.to(self.device).float()
        if self.normalize_embeddings is None:
            return self.model.get_global_ts_embedding(ts)
        original = getattr(self.model, "normalize_embeddings", False)
        self.model.normalize_embeddings = self.normalize_embeddings
        try:
            return self.model.get_global_ts_embedding(ts)
        finally:
            self.model.normalize_embeddings = original

    @torch.no_grad()
    def get_text_embedding(self, batch: Dict[str, Any]) -> Tensor:
        """
        Get text embedding from batch, handling LongAlign mode.

        Args:
            batch: Batch dictionary containing either:
                - "cap": Single caption per sample
                - "caps": List of captions per sample (LongAlign)

        Returns:
            (B, D) text embeddings
        """
        if self.use_longalign:
            return self._get_longalign_embedding(batch)
        else:
            return self._get_single_segment_embedding(batch)

    def _get_single_segment_embedding(self, batch: Dict[str, Any]) -> Tensor:
        """Get embedding for single-segment text."""
        if "cap" in batch:
            cap = batch["cap"]
            if self.normalize_embeddings is None:
                return self.model.get_text_embedding(cap)
            original = getattr(self.model, "normalize_embeddings", False)
            self.model.normalize_embeddings = self.normalize_embeddings
            try:
                return self.model.get_text_embedding(cap)
            finally:
                self.model.normalize_embeddings = original
        raise KeyError("Batch must contain 'cap' (raw text) for CLIP text embedding")

    def _get_longalign_embedding(self, batch: Dict[str, Any]) -> Tensor:
        """
        Get averaged embedding for LongAlign (multi-segment) mode.

        For LongAlign, each sample has multiple captions. We compute
        embeddings for all segments and average them.
        """
        if "caps" not in batch:
            # Fall back to single-segment if LongAlign data not available
            logger.warning(
                "LongAlign mode enabled but 'caps' not in batch. "
                "Falling back to single-segment mode."
            )
            return self._get_single_segment_embedding(batch)

        caps = batch["caps"]  # List of lists: [[cap1, cap2, ...], ...]

        # Flatten all captions
        all_caps = []
        segment_counts = []
        for caps_list in caps:
            all_caps.extend(caps_list)
            segment_counts.append(len(caps_list))

        # Get embeddings for all segments
        if self.normalize_embeddings is None:
            all_cap_emb = self.model.get_text_embedding(all_caps)  # (total_segments, D)
        else:
            original = getattr(self.model, "normalize_embeddings", False)
            self.model.normalize_embeddings = self.normalize_embeddings
            try:
                all_cap_emb = self.model.get_text_embedding(all_caps)  # (total_segments, D)
            finally:
                self.model.normalize_embeddings = original

        # Compute average embedding for each sample
        avg_cap_emb_list = []
        start_idx = 0
        for count in segment_counts:
            end_idx = start_idx + count
            seg_emb = all_cap_emb[start_idx:end_idx]  # (count, D)
            avg_emb = seg_emb.mean(dim=0, keepdim=True)  # (1, D)
            avg_cap_emb_list.append(avg_emb)
            start_idx = end_idx

        return torch.cat(avg_cap_emb_list, dim=0)  # (B, D)

    @torch.no_grad()
    def get_joint_embedding(
        self, ts_emb: Tensor, text_emb: Tensor
    ) -> Tensor:
        """
        Get joint time series + text embedding (concatenation).

        Used for JFTSD metric.

        Args:
            ts_emb: (B, D) time series embeddings
            text_emb: (B, D) text embeddings

        Returns:
            (B, 2*D) joint embeddings
        """
        return torch.cat([ts_emb, text_emb], dim=-1)


__all__ = ["CLIPEmbedder"]
