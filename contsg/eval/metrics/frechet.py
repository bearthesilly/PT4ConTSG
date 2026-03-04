"""
Frechet distance-based evaluation metrics.

This module implements metrics based on Frechet Distance (Wasserstein-2 for Gaussians),
including FID and JFTSD. All require reference statistics (mean, covariance)
computed from training data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor

from contsg.eval.metrics.base import CollectiveMetric
from contsg.eval.metrics.utils import calculate_frechet_distance
from contsg.registry import Registry

logger = logging.getLogger(__name__)


@Registry.register_metric("fid")
class FIDMetric(CollectiveMetric):
    """
    Frechet Inception Distance (FID) metric.

    Computes Frechet Distance between generated and reference time series
    in the CLIP embedding space.

    FD = ||mu_gen - mu_ref||^2 + Tr(cov_gen + cov_ref - 2*sqrt(cov_gen * cov_ref))

    Requires:
    - CLIP embeddings (ts_gen_emb in batch_data)
    - Reference statistics via set_reference()

    Legacy source: metrics.py:120-147
    """

    def __init__(
        self,
        name: str = "fid",
        reference_mean: Optional[np.ndarray] = None,
        reference_cov: Optional[np.ndarray] = None,
    ):
        """
        Initialize FID metric.

        Args:
            name: Metric identifier
            reference_mean: (D,) mean of reference embeddings
            reference_cov: (D, D) covariance of reference embeddings
        """
        super().__init__(name)
        self.reference_mean = reference_mean
        self.reference_cov = reference_cov

    @property
    def requires_clip(self) -> bool:
        """FID requires CLIP embeddings."""
        return True

    def set_reference(self, mean: np.ndarray, cov: np.ndarray) -> None:
        """
        Set reference statistics from cache.

        Args:
            mean: (D,) reference mean
            cov: (D, D) reference covariance
        """
        self.reference_mean = mean
        self.reference_cov = cov

    def update(self, batch_data: Dict[str, Any]) -> None:
        """
        Collect generated CLIP embeddings.

        Args:
            batch_data: Must contain 'ts_gen_emb' (B, D)
        """
        ts_gen_emb: Tensor = batch_data["ts_gen_emb"]
        self.collected_data.append(ts_gen_emb.cpu().numpy())

    def compute(self) -> float:
        """
        Compute FID between generated and reference distributions.

        Returns:
            FID value, or NaN if no data or no reference
        """
        if not self.collected_data:
            return float("nan")
        if self.reference_mean is None or self.reference_cov is None:
            logger.warning("FID: Reference statistics not set")
            return float("nan")

        # Concatenate all embeddings
        all_emb = np.concatenate(self.collected_data, axis=0).astype(np.float64)

        # Compute generated statistics
        gen_mean = np.mean(all_emb, axis=0)
        gen_cov = np.cov(all_emb, rowvar=False)

        # Add diagonal regularization for numerical stability
        reg = 1e-4
        gen_cov += np.eye(gen_cov.shape[0]) * reg

        # Compute Frechet Distance
        return calculate_frechet_distance(
            self.reference_mean, self.reference_cov, gen_mean, gen_cov
        )


@Registry.register_metric("jftsd")
class JFTSDMetric(CollectiveMetric):
    """
    Joint Frechet Text-Series Distance (JFTSD) metric.

    Computes Frechet Distance in the joint (TS + text) embedding space.
    Evaluates both time series quality and text-series alignment.

    Legacy source: metrics.py:148-177
    """

    def __init__(
        self,
        name: str = "jftsd",
        reference_mean: Optional[np.ndarray] = None,
        reference_cov: Optional[np.ndarray] = None,
    ):
        """
        Initialize JFTSD metric.

        Args:
            name: Metric identifier
            reference_mean: (2*D,) mean of reference joint embeddings
            reference_cov: (2*D, 2*D) covariance of reference joint embeddings
        """
        super().__init__(name)
        self.reference_mean = reference_mean
        self.reference_cov = reference_cov

    @property
    def requires_clip(self) -> bool:
        """JFTSD requires CLIP embeddings."""
        return True

    def set_reference(self, mean: np.ndarray, cov: np.ndarray) -> None:
        """
        Set reference statistics from cache.

        Args:
            mean: (2*D,) reference mean (joint embedding)
            cov: (2*D, 2*D) reference covariance
        """
        self.reference_mean = mean
        self.reference_cov = cov

    def update(self, batch_data: Dict[str, Any]) -> None:
        """
        Collect joint (TS + text) embeddings.

        Args:
            batch_data: Must contain 'ts_gen_emb' (B, D) and 'cap_emb' (B, D)
        """
        ts_gen_emb: Tensor = batch_data["ts_gen_emb"]
        cap_emb: Tensor = batch_data["cap_emb"]

        # Concatenate to form joint embedding
        joint_emb = torch.cat([ts_gen_emb, cap_emb], dim=-1)
        self.collected_data.append(joint_emb.cpu().numpy())

    def compute(self) -> float:
        """
        Compute JFTSD between generated and reference distributions.

        Returns:
            JFTSD value, or NaN if no data or no reference
        """
        if not self.collected_data:
            return float("nan")
        if self.reference_mean is None or self.reference_cov is None:
            logger.warning("JFTSD: Reference statistics not set")
            return float("nan")

        # Concatenate all embeddings
        all_emb = np.concatenate(self.collected_data, axis=0).astype(np.float64)

        # Compute generated statistics
        gen_mean = np.mean(all_emb, axis=0)
        gen_cov = np.cov(all_emb, rowvar=False)

        # Add diagonal regularization
        reg = 1e-4
        gen_cov += np.eye(gen_cov.shape[0]) * reg

        return calculate_frechet_distance(
            self.reference_mean, self.reference_cov, gen_mean, gen_cov
        )


__all__ = [
    "FIDMetric",
    "JFTSDMetric",
]
