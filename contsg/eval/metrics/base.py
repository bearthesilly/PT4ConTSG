"""
Base classes for evaluation metrics.

This module defines the abstract base classes for all metrics in the evaluation system.
All concrete metrics should inherit from one of these base classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import logging

logger = logging.getLogger(__name__)


class BaseMetric(ABC):
    """
    Abstract base class for all metrics.

    All metrics must implement:
    - reset(): Clear metric state
    - update(batch_data): Process batch data incrementally
    - compute(): Return final metric value

    Attributes:
        name: Identifier for this metric instance
    """

    def __init__(self, name: str):
        """
        Initialize metric.

        Args:
            name: Identifier for this metric
        """
        self.name = name
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Reset metric state for new evaluation run."""
        pass

    @abstractmethod
    def update(self, batch_data: Dict[str, Any]) -> None:
        """
        Update metric with batch data.

        Args:
            batch_data: Dictionary containing batch tensors:
                - 'pred': (B, L, F) median prediction
                - 'ts': (B, L, F) ground truth
                - 'ts_len': (B,) sequence lengths
                - 'multi_preds': (n_samples, B, L, F) multiple samples
                - 'ts_gen_emb': (B, D) CLIP embeddings of generated (if available)
                - 'ts_gt_emb': (B, D) CLIP embeddings of ground truth (if available)
                - 'cap_emb': (B, D) text CLIP embeddings (if available)
        """
        pass

    @abstractmethod
    def compute(self) -> Union[float, Dict[str, float]]:
        """
        Compute final metric value.

        Returns:
            Final metric value as float, or dict of multiple metric values
        """
        pass

    @property
    def requires_clip(self) -> bool:
        """
        Whether this metric requires CLIP embeddings.

        Metrics that return True will be skipped if CLIP is unavailable.
        Override in subclass if CLIP embeddings are needed.

        Returns:
            False by default
        """
        return False


class AccumulativeMetric(BaseMetric):
    """
    Base class for metrics that accumulate running averages.

    Suitable for metrics where the final value is computed as
    total_value / total_samples across all batches.

    Examples: CTTP, WAPE, EuclideanDistance
    """

    def __init__(self, name: str):
        """
        Initialize accumulative metric.

        Args:
            name: Identifier for this metric
        """
        super().__init__(name)
        self.total_value: float = 0.0
        self.total_samples: int = 0

    def reset(self) -> None:
        """Reset accumulators to zero."""
        self.total_value = 0.0
        self.total_samples = 0

    def compute(self) -> float:
        """
        Compute mean value from accumulated totals.

        Returns:
            total_value / total_samples, or 0.0 if no samples
        """
        if self.total_samples > 0:
            return self.total_value / self.total_samples
        return 0.0


class CollectiveMetric(BaseMetric):
    """
    Base class for metrics that need all data before computing.

    Suitable for metrics that require the full dataset distribution,
    such as FID (covariance), t-SNE, or histogram-based metrics.

    Examples: FID, JFTSD, DTW, CRPS, PRDC-F1
    """

    def __init__(self, name: str):
        """
        Initialize collective metric.

        Args:
            name: Identifier for this metric
        """
        super().__init__(name)
        self.collected_data: List[Any] = []

    def reset(self) -> None:
        """Clear collected data list."""
        self.collected_data = []


__all__ = [
    "BaseMetric",
    "AccumulativeMetric",
    "CollectiveMetric",
]
