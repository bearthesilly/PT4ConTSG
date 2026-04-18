"""
Statistics caching for Frechet-based and statistical metrics.

This module provides caching and computation of reference statistics needed for:
- FID, JFTSD: mean and covariance matrices
- SD, KD, MDD: skewness, kurtosis, and histogram statistics
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy import stats
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from contsg.utils.progress import smart_tqdm

if TYPE_CHECKING:
    from contsg.eval.embedder import CLIPEmbedder

logger = logging.getLogger(__name__)


class StatsCache:
    """
    Cache for reference statistics used by evaluation metrics.

    Caches:
    - FID: Time series CLIP embeddings (mean, covariance)
    - JFTSD: Joint (TS + text) CLIP embeddings (mean, covariance)
    - SD: Reference skewness per feature
    - KD: Reference kurtosis per feature
    - MDD: Histogram bin edges and reference histogram

    Statistics are computed from training set and cached to disk for reuse.
    """

    def __init__(
        self,
        cache_folder: Path,
        embedder: Optional["CLIPEmbedder"] = None,
        use_longalign: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize statistics cache.

        Args:
            cache_folder: Directory for cached statistics files
            embedder: CLIPEmbedder for FID/JFTSD (optional; SD/KD/MDD need only raw series)
            use_longalign: Whether using LongAlign mode (affects JFTSD cache path)
            device: Torch device for computation
        """
        self.cache_folder = Path(cache_folder)
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.use_longalign = use_longalign
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Frechet metrics stats
        self.fid_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.jftsd_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None

        # Statistical metrics stats
        self.sd_stats: Optional[np.ndarray] = None  # (F,) skewness
        self.kd_stats: Optional[np.ndarray] = None  # (F,) kurtosis
        self.mdd_stats: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None  # (bin_edges, hist, tot)

        # Build cache paths
        self._cache_paths = self._build_cache_paths()

    def _build_cache_paths(self) -> Dict[str, Path]:
        """Build cache file paths."""
        paths = {
            "fid_mean": self.cache_folder / "fid_mean.npy",
            "fid_cov": self.cache_folder / "fid_cov.npy",
            # Statistical metrics
            "sd_skew": self.cache_folder / "sd_ref_skew.npy",
            "kd_kurt": self.cache_folder / "kd_ref_kurt.npy",
            "mdd_bin_edges": self.cache_folder / "mdd_bin_edges.npy",
            "mdd_ref_hist": self.cache_folder / "mdd_ref_hist.npy",
            "mdd_ref_tot": self.cache_folder / "mdd_ref_tot.npy",
        }

        # Use separate JFTSD cache for LongAlign mode
        if self.use_longalign:
            paths["jftsd_mean"] = self.cache_folder / "jftsd_mean_longalign_avg.npy"
            paths["jftsd_cov"] = self.cache_folder / "jftsd_cov_longalign_avg.npy"
        else:
            paths["jftsd_mean"] = self.cache_folder / "jftsd_mean.npy"
            paths["jftsd_cov"] = self.cache_folder / "jftsd_cov.npy"

        return paths

    def load_or_compute(self, train_loader: DataLoader) -> None:
        """
        Load cached statistics or compute from training set.

        Args:
            train_loader: DataLoader for training data
        """
        if self.embedder is None:
            self.load_or_compute_statistical_only(train_loader)
            return

        if self._try_load():
            return

        logger.info("Cache miss - computing statistics from training set")
        self._compute_and_save(train_loader)

    def load_or_compute_statistical_only(
        self, train_loader: DataLoader, num_bins: int = 32
    ) -> None:
        """
        Load or compute SD/KD/MDD reference stats only (no CTTP / Frechet).

        Uses the same cache files as the full pipeline so a prior CTTP run's
        sd_ref_skew.npy etc. are reused when CTTP is unavailable.
        """
        if self._try_load_statistical_from_disk():
            logger.info("Loaded SD/KD/MDD reference statistics from cache (no CTTP)")
            return

        logger.info("Computing SD/KD/MDD reference statistics from training set (no CTTP)")
        self._compute_statistical_only(train_loader, num_bins=num_bins)
        self._save_statistical_only()

    def _try_load(self) -> bool:
        """Try to load cached statistics from disk."""
        # Required files for Frechet metrics
        frechet_files = [
            self._cache_paths["fid_mean"],
            self._cache_paths["fid_cov"],
            self._cache_paths["jftsd_mean"],
            self._cache_paths["jftsd_cov"],
        ]

        if not all(p.exists() for p in frechet_files):
            return False

        try:
            # Load Frechet stats
            self.fid_stats = (
                np.load(self._cache_paths["fid_mean"]),
                np.load(self._cache_paths["fid_cov"]),
            )
            self.jftsd_stats = (
                np.load(self._cache_paths["jftsd_mean"]),
                np.load(self._cache_paths["jftsd_cov"]),
            )
            logger.info("Loaded Frechet stats (FID, JFTSD)")

            # Load statistical metrics stats (optional)
            self._try_load_statistical_stats()

            return True
        except Exception as e:
            logger.warning(f"Failed to load cached statistics: {e}")
            return False

    def _try_load_statistical_from_disk(self) -> bool:
        """Return True if SD, KD, and MDD reference files are all present and loaded."""
        if not self._cache_paths["sd_skew"].exists():
            return False
        if not self._cache_paths["kd_kurt"].exists():
            return False
        mdd_files = [
            self._cache_paths["mdd_bin_edges"],
            self._cache_paths["mdd_ref_hist"],
            self._cache_paths["mdd_ref_tot"],
        ]
        if not all(p.exists() for p in mdd_files):
            return False
        try:
            self._try_load_statistical_stats()
        except Exception as e:
            logger.warning(f"Failed to load SD/KD/MDD cache files: {e}")
            return False
        return (
            self.sd_stats is not None
            and self.kd_stats is not None
            and self.mdd_stats is not None
        )

    def _try_load_statistical_stats(self) -> None:
        """Try to load SD/KD/MDD statistics (optional)."""
        try:
            if self._cache_paths["sd_skew"].exists():
                self.sd_stats = np.load(self._cache_paths["sd_skew"])
                logger.info("Loaded SD reference skewness")
        except Exception as e:
            logger.debug(f"SD stats not available: {e}")

        try:
            if self._cache_paths["kd_kurt"].exists():
                self.kd_stats = np.load(self._cache_paths["kd_kurt"])
                logger.info("Loaded KD reference kurtosis")
        except Exception as e:
            logger.debug(f"KD stats not available: {e}")

        try:
            mdd_files = [
                self._cache_paths["mdd_bin_edges"],
                self._cache_paths["mdd_ref_hist"],
                self._cache_paths["mdd_ref_tot"],
            ]
            if all(p.exists() for p in mdd_files):
                self.mdd_stats = (
                    np.load(self._cache_paths["mdd_bin_edges"]),
                    np.load(self._cache_paths["mdd_ref_hist"]),
                    np.load(self._cache_paths["mdd_ref_tot"]),
                )
                logger.info("Loaded MDD reference histogram")
        except Exception as e:
            logger.debug(f"MDD stats not available: {e}")

    def _compute_and_save(self, train_loader: DataLoader) -> None:
        """Compute statistics from training set and save to cache."""
        all_ts_emb = []
        all_joint_emb = []
        all_ts_raw = []
        all_ts_len = []

        logger.info("Computing all reference statistics from training set")

        with torch.no_grad():
            for batch in smart_tqdm(train_loader, desc="Extracting data"):
                ts = batch["ts"].to(self.device).float()
                ts_len = batch.get("ts_len")
                if ts_len is not None:
                    ts_len = ts_len.to(self.device).int()
                else:
                    ts_len = torch.full((ts.shape[0],), ts.shape[1], device=self.device, dtype=torch.int)

                # CLIP embeddings
                ts_emb = self.embedder.get_ts_embedding(ts, ts_len)
                all_ts_emb.append(ts_emb.cpu())

                text_emb = self.embedder.get_text_embedding(batch)
                joint_emb = self.embedder.get_joint_embedding(ts_emb, text_emb)
                all_joint_emb.append(joint_emb.cpu())

                # Raw data for statistical metrics
                all_ts_raw.append(ts.cpu())
                all_ts_len.append(ts_len.cpu())

        # Compute Frechet stats
        self.fid_stats = self._compute_frechet_stats(all_ts_emb, "FID")
        self.jftsd_stats = self._compute_frechet_stats(all_joint_emb, "JFTSD")

        # Compute statistical metrics stats using scipy
        self._compute_statistical_stats(all_ts_raw, all_ts_len)

        self._save_stats()

    def _compute_frechet_stats(
        self, emb_list: List[Tensor], name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and covariance for Frechet distance."""
        all_emb = torch.cat(emb_list, dim=0).numpy().astype(np.float64)
        mean = np.mean(all_emb, axis=0)
        cov = np.cov(all_emb, rowvar=False)
        # Regularization
        cov += np.eye(cov.shape[0]) * 1e-4
        logger.info(f"{name} cache: {mean.shape[0]} features")
        return mean, cov

    def _compute_statistical_stats(
        self, ts_raw_list: List[Tensor], ts_len_list: List[Tensor], num_bins: int = 32
    ) -> None:
        """
        Compute SD/KD/MDD reference statistics using scipy.

        Uses scipy.stats.skew and scipy.stats.kurtosis for robust computation.
        """
        logger.info("Computing SD/KD/MDD reference statistics")

        # Concatenate all data
        all_ts = torch.cat(ts_raw_list, dim=0).numpy()  # (N, L, F)
        all_len = torch.cat(ts_len_list, dim=0).numpy()  # (N,)
        N, L, F = all_ts.shape

        # Build valid data per feature (flatten across time with masking)
        data_per_feature = [[] for _ in range(F)]
        for i in range(N):
            valid_len = int(all_len[i])
            for f in range(F):
                data_per_feature[f].extend(all_ts[i, :valid_len, f].tolist())

        # Compute skewness and kurtosis per feature using scipy
        ref_skew = np.zeros(F, dtype=np.float64)
        ref_kurt = np.zeros(F, dtype=np.float64)
        for f in range(F):
            data = np.array(data_per_feature[f])
            if len(data) > 2:
                ref_skew[f] = stats.skew(data, nan_policy='omit')
                ref_kurt[f] = stats.kurtosis(data, fisher=False, nan_policy='omit')  # Fisher=False for excess kurtosis

        self.sd_stats = ref_skew
        self.kd_stats = ref_kurt
        logger.info(f"SD/KD: {F} features, skew range [{ref_skew.min():.3f}, {ref_skew.max():.3f}]")

        # Compute MDD histogram per (timestep, feature)
        bin_edges = np.empty((L, F, num_bins + 1), dtype=np.float32)
        ref_hist = np.zeros((L, F, num_bins), dtype=np.float64)
        ref_tot = np.zeros((L, F), dtype=np.float64)

        for t in range(L):
            for f in range(F):
                # Collect valid values at timestep t
                valid_mask = all_len > t
                values = all_ts[valid_mask, t, f]

                if len(values) == 0:
                    bin_edges[t, f] = np.linspace(-0.5, 0.5, num_bins + 1)
                    continue

                # Compute histogram using numpy
                vmin, vmax = values.min(), values.max()
                if vmax - vmin < 1e-6:
                    vmin, vmax = vmin - 0.5, vmax + 0.5

                hist, edges = np.histogram(values, bins=num_bins, range=(vmin, vmax))
                bin_edges[t, f] = edges
                ref_hist[t, f] = hist.astype(np.float64)
                ref_tot[t, f] = len(values)

        self.mdd_stats = (bin_edges, ref_hist, ref_tot)
        logger.info(f"MDD: L={L}, F={F}, bins={num_bins}")

    def _compute_statistical_only(
        self, train_loader: DataLoader, num_bins: int = 32
    ) -> None:
        """Collect raw training series and compute SD/KD/MDD references."""
        all_ts_raw: List[Tensor] = []
        all_ts_len: List[Tensor] = []

        with torch.no_grad():
            for batch in smart_tqdm(train_loader, desc="SD/KD/MDD reference data"):
                ts = batch["ts"].to(self.device).float()
                ts_len = batch.get("ts_len")
                if ts_len is not None:
                    ts_len = ts_len.to(self.device).int()
                else:
                    ts_len = torch.full(
                        (ts.shape[0],), ts.shape[1], device=self.device, dtype=torch.int
                    )
                all_ts_raw.append(ts.cpu())
                all_ts_len.append(ts_len.cpu())

        self._compute_statistical_stats(all_ts_raw, all_ts_len, num_bins=num_bins)

    def _save_stats(self) -> None:
        """Save all computed statistics to cache files."""
        # Frechet stats
        if self.fid_stats is not None and self.jftsd_stats is not None:
            np.save(self._cache_paths["fid_mean"], self.fid_stats[0])
            np.save(self._cache_paths["fid_cov"], self.fid_stats[1])
            np.save(self._cache_paths["jftsd_mean"], self.jftsd_stats[0])
            np.save(self._cache_paths["jftsd_cov"], self.jftsd_stats[1])

        # Statistical metrics stats
        self._save_statistical_only()

        logger.info(f"Saved all statistics to: {self.cache_folder}")

    def _save_statistical_only(self) -> None:
        """Persist SD/KD/MDD arrays only."""
        if self.sd_stats is not None:
            np.save(self._cache_paths["sd_skew"], self.sd_stats)
        if self.kd_stats is not None:
            np.save(self._cache_paths["kd_kurt"], self.kd_stats)
        if self.mdd_stats is not None:
            np.save(self._cache_paths["mdd_bin_edges"], self.mdd_stats[0])
            np.save(self._cache_paths["mdd_ref_hist"], self.mdd_stats[1])
            np.save(self._cache_paths["mdd_ref_tot"], self.mdd_stats[2])


__all__ = ["StatsCache"]
