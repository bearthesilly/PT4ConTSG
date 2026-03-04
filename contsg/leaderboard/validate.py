"""
Snapshot file validator for the ConTSG-Bench leaderboard.

Checks that generated parquet/json files conform to the expected schemas,
enum values, and value constraints.

Usage:
    python -m contsg.leaderboard.validate ./snapshots
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from contsg.leaderboard.metric_catalog import METRIC_NAMES
from contsg.leaderboard.schema import (
    ConditionModality,
    Direction,
    LeaderboardLongRow,
    MetricCatalogEntry,
    MetricGroup,
    SemanticLevel,
    VersionManifest,
)

logger = logging.getLogger(__name__)


class ValidationError:
    """A single validation issue."""

    def __init__(self, file: str, severity: str, message: str):
        self.file = file
        self.severity = severity  # "error" | "warning"
        self.message = message

    def __repr__(self) -> str:
        return f"[{self.severity.upper()}] {self.file}: {self.message}"


def validate_snapshot(snapshot_dir: str | Path) -> Tuple[bool, List[ValidationError]]:
    """Validate all 5 snapshot files in the given directory.

    Args:
        snapshot_dir: Path to the directory containing snapshot files.

    Returns:
        Tuple of (is_valid, errors) where is_valid is True if no errors
        (warnings are allowed).
    """
    snapshot_dir = Path(snapshot_dir)
    errors: List[ValidationError] = []

    # Check required files exist
    required_files = [
        "leaderboard_long.parquet",
        "leaderboard_wide.parquet",
        "model_cards.parquet",
        "metric_catalog.json",
        "version_manifest.json",
    ]
    for fname in required_files:
        if not (snapshot_dir / fname).exists():
            errors.append(ValidationError(fname, "error", f"File not found: {fname}"))

    # If critical files are missing, return early
    if any(e.severity == "error" for e in errors):
        return False, errors

    # Validate each file
    errors.extend(_validate_leaderboard_long(snapshot_dir / "leaderboard_long.parquet"))
    errors.extend(_validate_leaderboard_wide(snapshot_dir / "leaderboard_wide.parquet"))
    errors.extend(_validate_model_cards(snapshot_dir / "model_cards.parquet"))
    errors.extend(_validate_metric_catalog(snapshot_dir / "metric_catalog.json"))
    errors.extend(_validate_version_manifest(snapshot_dir / "version_manifest.json"))

    is_valid = not any(e.severity == "error" for e in errors)
    return is_valid, errors


def _validate_leaderboard_long(path: Path) -> List[ValidationError]:
    """Validate leaderboard_long.parquet."""
    errors: List[ValidationError] = []
    fname = path.name

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        errors.append(ValidationError(fname, "error", f"Failed to read parquet: {e}"))
        return errors

    if df.empty:
        errors.append(ValidationError(fname, "warning", "Parquet file is empty"))
        return errors

    # Check required columns
    required_cols = {
        "version",
        "submission_id",
        "model",
        "model_type",
        "dataset",
        "domain",
        "condition_modality",
        "semantic_level",
        "metric_name",
        "metric_value",
        "metric_group",
        "direction",
        "split",
        "n_runs",
        "seed_mean",
        "seed_std",
        "timestamp",
    }
    missing = required_cols - set(df.columns)
    if missing:
        errors.append(
            ValidationError(fname, "error", f"Missing columns: {sorted(missing)}")
        )
        return errors

    # Check enum values
    valid_modalities = {e.value for e in ConditionModality}
    invalid = set(df["condition_modality"].unique()) - valid_modalities
    if invalid:
        errors.append(
            ValidationError(fname, "error", f"Invalid condition_modality values: {invalid}")
        )

    valid_semantics = {e.value for e in SemanticLevel}
    invalid = set(df["semantic_level"].unique()) - valid_semantics
    if invalid:
        errors.append(
            ValidationError(fname, "error", f"Invalid semantic_level values: {invalid}")
        )

    valid_groups = {e.value for e in MetricGroup}
    invalid = set(df["metric_group"].unique()) - valid_groups
    if invalid:
        errors.append(
            ValidationError(fname, "error", f"Invalid metric_group values: {invalid}")
        )

    valid_directions = {e.value for e in Direction}
    invalid = set(df["direction"].unique()) - valid_directions
    if invalid:
        errors.append(
            ValidationError(fname, "error", f"Invalid direction values: {invalid}")
        )

    # Check metric names are registered
    unknown_metrics = set(df["metric_name"].unique()) - METRIC_NAMES
    if unknown_metrics:
        errors.append(
            ValidationError(
                fname, "warning", f"Unregistered metric names: {sorted(unknown_metrics)}"
            )
        )

    # Check numeric sanity
    non_finite = df["metric_value"].apply(lambda x: not math.isfinite(x)).sum()
    if non_finite > 0:
        errors.append(
            ValidationError(fname, "error", f"{non_finite} non-finite metric_value entries")
        )

    negative_std = (df["seed_std"] < 0).sum()
    if negative_std > 0:
        errors.append(
            ValidationError(fname, "error", f"{negative_std} negative seed_std values")
        )

    bad_nruns = (df["n_runs"] < 1).sum()
    if bad_nruns > 0:
        errors.append(
            ValidationError(fname, "error", f"{bad_nruns} n_runs values < 1")
        )

    return errors


def _validate_leaderboard_wide(path: Path) -> List[ValidationError]:
    """Validate leaderboard_wide.parquet."""
    errors: List[ValidationError] = []
    fname = path.name

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        errors.append(ValidationError(fname, "error", f"Failed to read parquet: {e}"))
        return errors

    if df.empty:
        errors.append(ValidationError(fname, "warning", "Parquet file is empty"))
        return errors

    # Check key dimension columns
    key_cols = {"model", "dataset", "version"}
    missing = key_cols - set(df.columns)
    if missing:
        errors.append(
            ValidationError(fname, "error", f"Missing key columns: {sorted(missing)}")
        )

    return errors


def _validate_model_cards(path: Path) -> List[ValidationError]:
    """Validate model_cards.parquet."""
    errors: List[ValidationError] = []
    fname = path.name

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        errors.append(ValidationError(fname, "error", f"Failed to read parquet: {e}"))
        return errors

    required_cols = {"model", "org", "model_type"}
    missing = required_cols - set(df.columns)
    if missing:
        errors.append(
            ValidationError(fname, "error", f"Missing columns: {sorted(missing)}")
        )

    # Check for duplicate models
    if "model" in df.columns:
        dupes = df["model"].duplicated().sum()
        if dupes > 0:
            errors.append(
                ValidationError(fname, "error", f"{dupes} duplicate model entries")
            )

    return errors


def _validate_metric_catalog(path: Path) -> List[ValidationError]:
    """Validate metric_catalog.json."""
    errors: List[ValidationError] = []
    fname = path.name

    try:
        with open(path, "r") as f:
            catalog = json.load(f)
    except Exception as e:
        errors.append(ValidationError(fname, "error", f"Failed to parse JSON: {e}"))
        return errors

    if not isinstance(catalog, list):
        errors.append(ValidationError(fname, "error", "Expected a JSON array"))
        return errors

    for i, entry in enumerate(catalog):
        try:
            MetricCatalogEntry(**entry)
        except Exception as e:
            errors.append(
                ValidationError(fname, "error", f"Entry {i} validation failed: {e}")
            )

    return errors


def _validate_version_manifest(path: Path) -> List[ValidationError]:
    """Validate version_manifest.json."""
    errors: List[ValidationError] = []
    fname = path.name

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        errors.append(ValidationError(fname, "error", f"Failed to parse JSON: {e}"))
        return errors

    try:
        VersionManifest(**data)
    except Exception as e:
        errors.append(ValidationError(fname, "error", f"Schema validation failed: {e}"))

    return errors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Validate leaderboard snapshot files.")
    parser.add_argument("snapshot_dir", type=str, help="Path to snapshot directory")
    args = parser.parse_args()

    is_valid, errors = validate_snapshot(args.snapshot_dir)

    for err in errors:
        print(err)

    if is_valid:
        print(f"\nValidation PASSED ({len(errors)} warnings)")
    else:
        n_errors = sum(1 for e in errors if e.severity == "error")
        print(f"\nValidation FAILED ({n_errors} errors, {len(errors) - n_errors} warnings)")
        sys.exit(1)
