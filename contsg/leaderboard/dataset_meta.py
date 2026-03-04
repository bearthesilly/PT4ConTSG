"""
Dataset metadata registry for the ConTSG-Bench leaderboard.

Maps each benchmark dataset to its domain and semantic abstraction level.
The condition_modality is NOT a dataset property — it is inferred per-experiment
from the config's ``condition.{text/attribute/label}.enabled`` flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class DatasetMeta:
    """Metadata for a single benchmark dataset."""

    name: str
    domain: str
    semantic_level: str  # "morphological" | "conceptual"
    description: str
    n_var: int
    seq_length: int


# ---------------------------------------------------------------------------
# Canonical dataset metadata (10 benchmark datasets)
# ---------------------------------------------------------------------------

DATASET_META: Dict[str, DatasetMeta] = {
    # Synthetic datasets
    "synth-m": DatasetMeta(
        name="synth-m",
        domain="synthetic",
        semantic_level="morphological",
        description="Synthetic multivariable dataset with various patterns (sine, triangle, square).",
        n_var=2,
        seq_length=128,
    ),
    "synth-u": DatasetMeta(
        name="synth-u",
        domain="synthetic",
        semantic_level="morphological",
        description="Synthetic univariable dataset with diverse patterns and natural language descriptions.",
        n_var=1,
        seq_length=128,
    ),
    # Real-world datasets
    "ettm1": DatasetMeta(
        name="ettm1",
        domain="energy",
        semantic_level="morphological",
        description="Electricity Transformer Temperature (ETTm1): univariate after decomposition, 15-min granularity.",
        n_var=1,
        seq_length=120,
    ),
    "weather_concept": DatasetMeta(
        name="weather_concept",
        domain="weather",
        semantic_level="conceptual",
        description="Weather with concept-level (semantic) text/attribute conditions.",
        n_var=10,
        seq_length=36,
    ),
    "weather_morphology": DatasetMeta(
        name="weather_morphology",
        domain="weather",
        semantic_level="morphological",
        description="Weather with morphology-level (structural) text/attribute conditions.",
        n_var=10,
        seq_length=36,
    ),
    "telecomts_segment": DatasetMeta(
        name="telecomts_segment",
        domain="telecom",
        semantic_level="morphological",
        description="Telecom network metrics (RSRP, UL SNR) with segment-level text annotations.",
        n_var=2,
        seq_length=128,
    ),
    "istanbul_traffic": DatasetMeta(
        name="istanbul_traffic",
        domain="traffic",
        semantic_level="morphological",
        description="Istanbul traffic flow measurements from road network.",
        n_var=1,
        seq_length=144,
    ),
    "airquality_beijing": DatasetMeta(
        name="airquality_beijing",
        domain="environment",
        semantic_level="morphological",
        description="Beijing air quality dataset with 6 pollutant variates.",
        n_var=6,
        seq_length=24,
    ),
    "ptbxl_concept": DatasetMeta(
        name="ptbxl_concept",
        domain="health",
        semantic_level="conceptual",
        description="PTB-XL ECG dataset with concept-level conditions.",
        n_var=12,
        seq_length=1000,
    ),
    "ptbxl_morphology": DatasetMeta(
        name="ptbxl_morphology",
        domain="health",
        semantic_level="morphological",
        description="PTB-XL ECG dataset with morphology-level conditions.",
        n_var=12,
        seq_length=1000,
    ),
}

# Convenience sets for quick membership checks
BENCHMARK_DATASET_NAMES = frozenset(DATASET_META.keys())
CONCEPTUAL_DATASETS = frozenset(
    name for name, meta in DATASET_META.items() if meta.semantic_level == "conceptual"
)
MORPHOLOGICAL_DATASETS = frozenset(
    name for name, meta in DATASET_META.items() if meta.semantic_level == "morphological"
)


def get_dataset_meta(name: str) -> DatasetMeta:
    """Look up dataset metadata by name.

    Raises:
        KeyError: If the dataset name is not registered.
    """
    if name not in DATASET_META:
        raise KeyError(
            f"Unknown dataset '{name}'. "
            f"Registered datasets: {sorted(DATASET_META.keys())}"
        )
    return DATASET_META[name]


def infer_condition_modality(condition_config: dict) -> str:
    """Infer condition modality from an experiment's condition config dict.

    Precedence: text > attribute > label (first enabled wins).

    Args:
        condition_config: The ``condition`` section from a config YAML, e.g.
            ``{"text": {"enabled": True}, "attribute": {"enabled": False}, ...}``

    Returns:
        One of ``"text"``, ``"attribute"``, or ``"label"``.

    Raises:
        ValueError: If no condition type is enabled.
    """
    for modality in ("text", "attribute", "label"):
        section = condition_config.get(modality, {})
        if isinstance(section, dict) and section.get("enabled", False):
            return modality
    raise ValueError(
        f"No condition modality enabled in config: {condition_config}"
    )
