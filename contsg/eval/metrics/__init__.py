"""
Evaluation metrics package for ConTSG.

This package provides a comprehensive set of metrics for evaluating
conditional time series generation quality.

Metric Categories:
- Basic: DTW, WAPE, ED, CRPS (no CLIP dependency)
- Statistical: ACD, SD, KD, MDD (distribution comparison)
- Frechet: FID, JFTSD (Frechet distance-based)
- PRDC: PRDCF1, JointPRDCF1 (precision/recall)
- Specialized: CTTP, DiscAUC, TSNEViz
- Segment: SegmentParameterAccuracy (dataset-specific)

All metrics implement the BaseMetric interface and are auto-registered
via the @Registry.register_metric decorator.

Example:
    from contsg.registry import Registry

    # Get metric by name
    metric_cls = Registry.get_metric("dtw")
    metric = metric_cls()

    # Use metric
    metric.reset()
    for batch in loader:
        metric.update(batch_data)
    result = metric.compute()
"""

from __future__ import annotations

# Import base classes
from contsg.eval.metrics.base import (
    BaseMetric,
    AccumulativeMetric,
    CollectiveMetric,
)

# Import utility functions
from contsg.eval.metrics.utils import (
    calculate_frechet_distance,
)

# Import all metric modules to trigger registration
# Note: Import order matters for any dependencies

# Basic metrics (no CLIP required)
from contsg.eval.metrics import basic
from contsg.eval.metrics.basic import (
    DTWMetric,
    WAPEMetric,
    EuclideanDistanceMetric,
    CRPSMetric,
)

# Statistical metrics
from contsg.eval.metrics import statistical
from contsg.eval.metrics.statistical import (
    ACDMetric,
    SkewnessDiffMetric,
    KurtosisDiffMetric,
    MDDMetric,
)

# Frechet-based metrics (require CLIP)
from contsg.eval.metrics import frechet
from contsg.eval.metrics.frechet import (
    FIDMetric,
    JFTSDMetric,
)

# PRDC metrics (require CLIP)
from contsg.eval.metrics import prdc
from contsg.eval.metrics.prdc import (
    PRDCF1Metric,
    JointPRDCF1Metric,
)

# Specialized metrics
from contsg.eval.metrics import specialized
from contsg.eval.metrics.specialized import (
    CTTPMetric,
    RealVsFakeDiscriminatorAUCMetric,
    TSNETimeSeriesVisualizationMetric,
)

# Segment metrics (dataset-specific)
from contsg.eval.metrics import segment
from contsg.eval.metrics.segment import (
    PresenceClassifier1D,
    ParameterClassifier1D,
    SegmentParameterAccuracyMetric,
)


# ==============================================================================
# Package Exports
# ==============================================================================

__all__ = [
    # Base classes
    "BaseMetric",
    "AccumulativeMetric",
    "CollectiveMetric",
    # Utilities
    "calculate_frechet_distance",
    # Basic metrics
    "DTWMetric",
    "WAPEMetric",
    "EuclideanDistanceMetric",
    "CRPSMetric",
    # Statistical metrics
    "ACDMetric",
    "SkewnessDiffMetric",
    "KurtosisDiffMetric",
    "MDDMetric",
    # Frechet metrics
    "FIDMetric",
    "JFTSDMetric",
    # PRDC metrics
    "PRDCF1Metric",
    "JointPRDCF1Metric",
    # Specialized metrics
    "CTTPMetric",
    "RealVsFakeDiscriminatorAUCMetric",
    "TSNETimeSeriesVisualizationMetric",
    # Segment metrics
    "PresenceClassifier1D",
    "ParameterClassifier1D",
    "SegmentParameterAccuracyMetric",
]
