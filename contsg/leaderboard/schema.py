"""
Pydantic validation models for the 5 leaderboard snapshot files.

These schemas are used by both the aggregation pipeline (to validate output)
and the HF Space frontend (to validate input at load time).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enum types
# ---------------------------------------------------------------------------


class ConditionModality(str, Enum):
    TEXT = "text"
    ATTRIBUTE = "attribute"
    LABEL = "label"


class SemanticLevel(str, Enum):
    MORPHOLOGICAL = "morphological"
    CONCEPTUAL = "conceptual"
    MIXED = "mixed"


class MetricGroup(str, Enum):
    FIDELITY = "fidelity"
    ADHERENCE = "adherence"
    UTILITY = "utility"


class Direction(str, Enum):
    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"


# ---------------------------------------------------------------------------
# §4.3  leaderboard_long.parquet
# ---------------------------------------------------------------------------


class LeaderboardLongRow(BaseModel):
    """One row in ``leaderboard_long.parquet`` — a single metric value for
    one (model, dataset) pair."""

    version: str
    submission_id: str
    model: str
    model_type: str
    dataset: str
    domain: str
    condition_modality: ConditionModality
    semantic_level: SemanticLevel
    metric_name: str
    metric_value: float
    metric_group: MetricGroup
    direction: Direction
    split: str = "test"
    n_runs: int = Field(ge=1, default=1)
    seed_mean: float
    seed_std: float = Field(ge=0, default=0.0)
    timestamp: str  # ISO 8601

    @field_validator("metric_value")
    @classmethod
    def _finite(cls, v: float) -> float:
        import math

        if not math.isfinite(v):
            raise ValueError(f"metric_value must be finite, got {v}")
        return v


# ---------------------------------------------------------------------------
# §4.5  model_cards.parquet
# ---------------------------------------------------------------------------


class ModelCardRow(BaseModel):
    """One row in ``model_cards.parquet`` — metadata for a single model."""

    model: str
    org: str
    model_link: str = ""
    code_link: str = ""
    paper_link: str = ""
    params: str = ""
    model_type: str
    training_data_compliance: bool = True
    testdata_leakage: bool = False
    replication_code_available: bool = True
    notes: str = ""


# ---------------------------------------------------------------------------
# §4.6  metric_catalog.json
# ---------------------------------------------------------------------------


class MetricCatalogEntry(BaseModel):
    """One entry in ``metric_catalog.json``."""

    metric_name: str
    display_name: str
    metric_group: MetricGroup
    direction: Direction
    default_weight: float = Field(gt=0, default=1.0)
    required_for_rank: bool = False


# ---------------------------------------------------------------------------
# §4.7  version_manifest.json
# ---------------------------------------------------------------------------


class VersionFiles(BaseModel):
    """File paths within a version snapshot."""

    leaderboard_long: str = "leaderboard_long.parquet"
    leaderboard_wide: str = "leaderboard_wide.parquet"
    model_cards: str = "model_cards.parquet"
    metric_catalog: str = "metric_catalog.json"


class VersionEntry(BaseModel):
    """One version in ``version_manifest.json``."""

    version: str
    release_date: str  # ISO 8601 date
    hash: str  # SHA-256 hex digest
    n_models: int = Field(ge=0)
    n_datasets: int = Field(ge=0)
    n_metrics: int = Field(ge=0)
    changelog: str = ""
    files: VersionFiles = Field(default_factory=VersionFiles)


class VersionManifest(BaseModel):
    """Top-level structure of ``version_manifest.json``."""

    current_version: str
    versions: List[VersionEntry]


# ---------------------------------------------------------------------------
# Submission file schema (for GitHub PR submissions)
# ---------------------------------------------------------------------------


class SubmissionModelMeta(BaseModel):
    """Model metadata provided by the submitter."""

    name: str
    model_type: str  # diffusion | flow | vae | gan | retrieval | other
    native_condition: str = ""  # text | attribute | label (model's native condition type)
    org: str = ""
    paper_link: str = ""
    code_link: str = ""
    model_link: str = ""
    ckpt_scope: List[str] = Field(default_factory=list)
    reproducibility: Dict[str, str] = Field(default_factory=dict)
    params: str = ""  # e.g. "45M"
    notes: str = ""


class SubmissionMetric(BaseModel):
    """A single metric value with optional standard deviation."""

    mean: float
    std: float = Field(ge=0, default=0.0)

    @field_validator("mean")
    @classmethod
    def _finite_mean(cls, v: float) -> float:
        import math

        if not math.isfinite(v):
            raise ValueError(f"mean must be finite, got {v}")
        return v


class SubmissionResult(BaseModel):
    """Results for one dataset within a submission."""

    dataset: str
    condition_modality: ConditionModality
    n_runs: int = Field(ge=1, default=1)
    metrics: Dict[str, SubmissionMetric]


class SubmissionFile(BaseModel):
    """Top-level schema for a ``submissions/<model>.yaml`` file."""

    model: SubmissionModelMeta
    results: List[SubmissionResult] = Field(min_length=1)
