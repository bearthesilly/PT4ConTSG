"""
Validate a submission YAML file before creating a GitHub PR.

Usage:
    python -m contsg.leaderboard.validate_submission submissions/my_model.yaml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import yaml

from contsg.leaderboard.dataset_meta import BENCHMARK_DATASET_NAMES
from contsg.leaderboard.metric_catalog import METRIC_NAMES
from contsg.leaderboard.schema import SubmissionFile

logger = logging.getLogger(__name__)


class SubmissionIssue:
    """A single validation issue."""

    def __init__(self, severity: str, message: str):
        self.severity = severity  # "error" | "warning"
        self.message = message

    def __repr__(self) -> str:
        return f"[{self.severity.upper()}] {self.message}"


def validate_submission_file(path: str | Path) -> Tuple[bool, List[SubmissionIssue]]:
    """Validate a single submission YAML file.

    Checks:
      1. YAML parses correctly
      2. Matches SubmissionFile schema (Pydantic)
      3. Dataset names are in the benchmark set
      4. Metric names are in the catalog
      5. No duplicate dataset entries

    Args:
        path: Path to the YAML file.

    Returns:
        Tuple of (is_valid, issues).
    """
    path = Path(path)
    issues: List[SubmissionIssue] = []

    # 1. File exists and is readable
    if not path.exists():
        issues.append(SubmissionIssue("error", f"File not found: {path}"))
        return False, issues

    # 2. YAML parses
    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        issues.append(SubmissionIssue("error", f"YAML parse error: {e}"))
        return False, issues

    if not isinstance(raw, dict):
        issues.append(SubmissionIssue("error", "Root element must be a YAML mapping"))
        return False, issues

    # 3. Schema validation
    try:
        sub = SubmissionFile(**raw)
    except Exception as e:
        issues.append(SubmissionIssue("error", f"Schema validation failed: {e}"))
        return False, issues

    # 4. Check model name
    model_name = sub.model.name
    if not model_name:
        issues.append(SubmissionIssue("error", "model.name is empty"))
    if " " in model_name:
        issues.append(
            SubmissionIssue("warning", f"model.name '{model_name}' contains spaces")
        )

    # 5. Check datasets and metrics
    seen_datasets = set()
    total_metrics = 0

    for result in sub.results:
        ds = result.dataset

        # Duplicate dataset check
        if ds in seen_datasets:
            issues.append(
                SubmissionIssue("warning", f"Duplicate dataset entry: '{ds}'")
            )
        seen_datasets.add(ds)

        # Dataset in benchmark
        if ds not in BENCHMARK_DATASET_NAMES:
            issues.append(
                SubmissionIssue(
                    "error",
                    f"Unknown dataset '{ds}'. "
                    f"Valid datasets: {sorted(BENCHMARK_DATASET_NAMES)}",
                )
            )

        # Metric names
        for metric_name in result.metrics:
            total_metrics += 1
            if metric_name not in METRIC_NAMES:
                issues.append(
                    SubmissionIssue(
                        "warning",
                        f"Unknown metric '{metric_name}' in {ds} "
                        f"(will be ignored by the leaderboard)",
                    )
                )

    # Summary checks
    if total_metrics == 0:
        issues.append(SubmissionIssue("error", "No metrics found in any result entry"))

    if not sub.model.model_type:
        issues.append(SubmissionIssue("warning", "model.model_type is empty"))

    is_valid = not any(i.severity == "error" for i in issues)

    # Add informational summary
    if is_valid:
        n_datasets = len(seen_datasets & BENCHMARK_DATASET_NAMES)
        known_metrics = sum(
            1
            for r in sub.results
            for m in r.metrics
            if m in METRIC_NAMES
        )
        logger.info(
            "Submission '%s': %d datasets, %d recognized metrics",
            model_name,
            n_datasets,
            known_metrics,
        )

    return is_valid, issues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Validate a submission YAML file for the ConTSG-Bench leaderboard."
    )
    parser.add_argument("yaml_file", type=str, help="Path to the submission YAML file")
    args = parser.parse_args()

    is_valid, issues = validate_submission_file(args.yaml_file)

    for issue in issues:
        print(issue)

    if is_valid:
        n_warnings = sum(1 for i in issues if i.severity == "warning")
        print(f"\nValidation PASSED ({n_warnings} warnings)")
    else:
        n_errors = sum(1 for i in issues if i.severity == "error")
        n_warnings = sum(1 for i in issues if i.severity == "warning")
        print(f"\nValidation FAILED ({n_errors} errors, {n_warnings} warnings)")
        sys.exit(1)
