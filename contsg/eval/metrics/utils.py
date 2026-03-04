"""
Utility functions for evaluation metrics.

This module provides common mathematical operations used across multiple metrics,
including Frechet Distance calculation.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Calculate Frechet Distance between two multivariate Gaussians.

    The Frechet Distance (also known as Wasserstein-2 distance for Gaussians) is:
        FD = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 * sigma2))

    Used for FID and JFTSD metrics.

    Args:
        mu1: Mean of first Gaussian, shape (D,)
        sigma1: Covariance of first Gaussian, shape (D, D)
        mu2: Mean of second Gaussian, shape (D,)
        sigma2: Covariance of second Gaussian, shape (D, D)
        eps: Small epsilon for numerical stability

    Returns:
        Frechet Distance as float

    Raises:
        AssertionError: If shapes don't match
    """
    # Ensure proper shapes and precision
    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    sigma1 = np.atleast_2d(sigma1).astype(np.float64)
    sigma2 = np.atleast_2d(sigma2).astype(np.float64)

    assert mu1.shape == mu2.shape, (
        f"Mean vectors have different shapes: {mu1.shape} vs {mu2.shape}"
    )
    assert sigma1.shape == sigma2.shape, (
        f"Covariance matrices have different shapes: {sigma1.shape} vs {sigma2.shape}"
    )

    # Mean difference term: ||mu1 - mu2||^2
    diff = mu1 - mu2

    # Matrix square root term: sqrt(sigma1 * sigma2)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Handle numerical issues with singular matrices
    if not np.isfinite(covmean).all():
        logger.info(
            f"FID calculation produces singular product; "
            f"adding {eps} to diagonal of cov estimates"
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Handle complex output (imaginary part should be negligible)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    # FD = ||mu1 - mu2||^2 + Tr(sigma1) + Tr(sigma2) - 2*Tr(sqrt(sigma1*sigma2))
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


__all__ = [
    "calculate_frechet_distance",
]
