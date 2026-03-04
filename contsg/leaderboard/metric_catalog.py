"""
Canonical metric catalog for the ConTSG-Bench leaderboard.

This module is the **single source of truth** for all metric definitions used
in leaderboard generation, validation, and frontend rendering.  It mirrors the
``metric_catalog.json`` snapshot described in the framework document (§4.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class MetricDef:
    """Definition of a single evaluation metric."""

    metric_name: str
    display_name: str
    metric_group: str  # fidelity | adherence | utility
    direction: str  # higher_better | lower_better
    default_weight: float = 1.0
    required_for_rank: bool = False


# ---------------------------------------------------------------------------
# All 15 benchmark metrics (3 groups)
# ---------------------------------------------------------------------------

METRIC_CATALOG: List[MetricDef] = [
    # ── Fidelity (7 metrics) ─────────────────────────────────────────────
    MetricDef("acd", "ACD", "fidelity", "lower_better"),
    MetricDef("sd", "SD", "fidelity", "lower_better"),
    MetricDef("kd", "KD", "fidelity", "lower_better"),
    MetricDef("mdd", "MDD", "fidelity", "lower_better"),
    MetricDef("fid", "FID", "fidelity", "lower_better", required_for_rank=True),
    MetricDef("prdc_f1.precision", "kNN-PRF Precision", "fidelity", "higher_better"),
    MetricDef("prdc_f1.recall", "kNN-PRF Recall", "fidelity", "higher_better"),
    # ── Adherence (4 metrics) ────────────────────────────────────────────
    MetricDef("jftsd", "JFTSD", "adherence", "lower_better", required_for_rank=True),
    MetricDef(
        "joint_prdc_f1.precision", "Joint kNN-PRF Precision", "adherence", "higher_better"
    ),
    MetricDef(
        "joint_prdc_f1.recall", "Joint kNN-PRF Recall", "adherence", "higher_better"
    ),
    MetricDef("cttp", "CTTP", "adherence", "higher_better", required_for_rank=True),
    # ── Utility (4 metrics) ──────────────────────────────────────────────
    MetricDef("dtw", "DTW", "utility", "lower_better"),
    MetricDef("crps", "CRPS", "utility", "lower_better"),
    MetricDef("ed", "ED", "utility", "lower_better"),
    MetricDef("wape", "WAPE", "utility", "lower_better"),
]

# ---------------------------------------------------------------------------
# Derived lookup structures
# ---------------------------------------------------------------------------

METRIC_BY_NAME: Dict[str, MetricDef] = {m.metric_name: m for m in METRIC_CATALOG}
METRIC_NAMES: frozenset = frozenset(METRIC_BY_NAME.keys())

METRIC_GROUPS = ("fidelity", "adherence", "utility")

METRICS_BY_GROUP: Dict[str, List[MetricDef]] = {
    group: [m for m in METRIC_CATALOG if m.metric_group == group]
    for group in METRIC_GROUPS
}

REQUIRED_METRICS: frozenset = frozenset(
    m.metric_name for m in METRIC_CATALOG if m.required_for_rank
)


def to_json_list() -> list[dict]:
    """Serialize the catalog to a list of dicts (for ``metric_catalog.json``)."""
    return [
        {
            "metric_name": m.metric_name,
            "display_name": m.display_name,
            "metric_group": m.metric_group,
            "direction": m.direction,
            "default_weight": m.default_weight,
            "required_for_rank": m.required_for_rank,
        }
        for m in METRIC_CATALOG
    ]
