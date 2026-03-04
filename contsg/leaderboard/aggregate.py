"""
Aggregation pipeline for the ConTSG-Bench leaderboard.

Two entry points:
  1. aggregate_submissions(): reads submissions/*.yaml → 5 snapshot files
     (primary path for open-source leaderboard)
  2. aggregate_experiments(): reads experiment dirs with eval_results.json + config.yaml
     (legacy path for internal use)

Usage (submissions):
    python -m contsg.leaderboard.aggregate submissions ./snapshots --version v2026.02.06

Usage (experiments, legacy):
    python -m contsg.leaderboard.aggregate experiments ./snapshots --version v2026.02.06 --mode experiments
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from contsg.leaderboard.dataset_meta import (
    BENCHMARK_DATASET_NAMES,
    get_dataset_meta,
    infer_condition_modality,
)
from contsg.leaderboard.metric_catalog import (
    METRIC_BY_NAME,
    METRIC_NAMES,
    to_json_list,
)
from contsg.leaderboard.schema import SubmissionFile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _compute_snapshot_hash(*file_paths: Path) -> str:
    """Compute a SHA-256 hash over the contents of multiple files."""
    hasher = hashlib.sha256()
    for fp in sorted(file_paths):
        if fp.exists():
            hasher.update(fp.read_bytes())
    return f"sha256:{hasher.hexdigest()}"


def _write_snapshots(
    all_rows: List[Dict[str, Any]],
    model_card_rows: List[Dict[str, Any]],
    output_dir: Path,
    version: str,
) -> Path:
    """Write the 5 snapshot files from prepared rows.

    Args:
        all_rows: List of leaderboard long-format row dicts.
        model_card_rows: List of model card row dicts.
        output_dir: Where to write the snapshot files.
        version: Version string.

    Returns:
        Path to the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. leaderboard_long.parquet ---
    df_long = pd.DataFrame(all_rows)
    long_path = output_dir / "leaderboard_long.parquet"
    df_long.to_parquet(long_path, index=False)
    logger.info("Wrote %s (%d rows)", long_path, len(df_long))

    # --- 2. leaderboard_wide.parquet ---
    if not df_long.empty:
        pivot_index = [
            "version",
            "submission_id",
            "model",
            "model_type",
            "dataset",
            "domain",
            "condition_modality",
            "semantic_level",
            "split",
            "n_runs",
            "timestamp",
        ]
        df_wide = df_long.pivot_table(
            index=pivot_index,
            columns="metric_name",
            values="metric_value",
            aggfunc="first",
        ).reset_index()
        df_wide.columns.name = None
    else:
        df_wide = pd.DataFrame()

    wide_path = output_dir / "leaderboard_wide.parquet"
    df_wide.to_parquet(wide_path, index=False)
    logger.info("Wrote %s (%d rows)", wide_path, len(df_wide))

    # --- 3. model_cards.parquet ---
    df_cards = pd.DataFrame(model_card_rows)
    cards_path = output_dir / "model_cards.parquet"
    df_cards.to_parquet(cards_path, index=False)
    logger.info("Wrote %s (%d rows)", cards_path, len(df_cards))

    # --- 4. metric_catalog.json ---
    catalog_path = output_dir / "metric_catalog.json"
    with open(catalog_path, "w") as f:
        json.dump(to_json_list(), f, indent=2)
    logger.info("Wrote %s", catalog_path)

    # --- 5. version_manifest.json ---
    snapshot_hash = _compute_snapshot_hash(long_path, wide_path, cards_path, catalog_path)

    n_models = df_long["model"].nunique() if not df_long.empty else 0
    n_datasets = df_long["dataset"].nunique() if not df_long.empty else 0
    n_metrics = df_long["metric_name"].nunique() if not df_long.empty else 0

    manifest = {
        "current_version": version,
        "versions": [
            {
                "version": version,
                "release_date": datetime.now().strftime("%Y-%m-%d"),
                "hash": snapshot_hash,
                "n_models": n_models,
                "n_datasets": n_datasets,
                "n_metrics": n_metrics,
                "changelog": (
                    f"Generated from {len(all_rows)} metric rows across "
                    f"{n_models} models and {n_datasets} datasets."
                ),
                "files": {
                    "leaderboard_long": "leaderboard_long.parquet",
                    "leaderboard_wide": "leaderboard_wide.parquet",
                    "model_cards": "model_cards.parquet",
                    "metric_catalog": "metric_catalog.json",
                },
            }
        ],
    }

    manifest_path = output_dir / "version_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote %s", manifest_path)

    logger.info(
        "Aggregation complete: %d rows, %d models, %d datasets, %d metrics",
        len(all_rows),
        n_models,
        n_datasets,
        n_metrics,
    )

    return output_dir


# ---------------------------------------------------------------------------
# Submission-based aggregation (primary path)
# ---------------------------------------------------------------------------


def _load_submission(yaml_path: Path) -> Optional[SubmissionFile]:
    """Load and validate a single submission YAML file."""
    try:
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)
        return SubmissionFile(**raw)
    except Exception as e:
        logger.warning("Failed to load submission %s: %s", yaml_path, e)
        return None


def _submission_to_rows(
    sub: SubmissionFile,
    version: str,
    timestamp: str,
) -> List[Dict[str, Any]]:
    """Convert one submission file into leaderboard long-format rows."""
    model_name = sub.model.name
    model_type = sub.model.model_type

    rows: List[Dict[str, Any]] = []
    for result in sub.results:
        dataset_name = result.dataset

        # Validate dataset is a benchmark dataset
        if dataset_name not in BENCHMARK_DATASET_NAMES:
            logger.warning(
                "Skipping unknown dataset '%s' in submission for model '%s'",
                dataset_name,
                model_name,
            )
            continue

        dataset_meta = get_dataset_meta(dataset_name)
        submission_id = f"{model_name}_{dataset_name}"

        for metric_name, metric_val in result.metrics.items():
            # Only include metrics that are in our catalog
            if metric_name not in METRIC_NAMES:
                logger.debug(
                    "Skipping unknown metric '%s' in %s/%s",
                    metric_name,
                    model_name,
                    dataset_name,
                )
                continue

            # Skip non-finite values
            if not math.isfinite(metric_val.mean):
                logger.warning(
                    "Skipping non-finite metric %s=%s in %s/%s",
                    metric_name,
                    metric_val.mean,
                    model_name,
                    dataset_name,
                )
                continue

            metric_def = METRIC_BY_NAME[metric_name]
            rows.append(
                {
                    "version": version,
                    "submission_id": submission_id,
                    "model": model_name,
                    "model_type": model_type,
                    "dataset": dataset_name,
                    "domain": dataset_meta.domain,
                    "condition_modality": result.condition_modality.value,
                    "semantic_level": dataset_meta.semantic_level,
                    "metric_name": metric_name,
                    "metric_value": metric_val.mean,
                    "metric_group": metric_def.metric_group,
                    "direction": metric_def.direction,
                    "split": "test",
                    "n_runs": result.n_runs,
                    "seed_mean": metric_val.mean,
                    "seed_std": metric_val.std,
                    "timestamp": timestamp,
                }
            )

    return rows


def _submission_to_model_card(sub: SubmissionFile) -> Dict[str, Any]:
    """Extract model card from a submission file."""
    m = sub.model
    return {
        "model": m.name,
        "org": m.org,
        "model_link": m.model_link,
        "code_link": m.code_link,
        "paper_link": m.paper_link,
        "params": m.params,
        "model_type": m.model_type,
        "native_condition": m.native_condition,
        "training_data_compliance": True,
        "testdata_leakage": False,
        "replication_code_available": bool(m.code_link),
        "notes": m.notes,
    }


def aggregate_submissions(
    submissions_dir: str | Path,
    output_dir: str | Path,
    version: str = "",
) -> Path:
    """Aggregate submission YAML files into 5 leaderboard snapshot files.

    Each YAML file in ``submissions_dir`` represents one model's results.
    Model metadata is extracted from the YAML (no pre-registration required).

    Args:
        submissions_dir: Path to directory containing ``*.yaml`` submission files.
        output_dir: Where to write the snapshot files.
        version: Version string (e.g. ``"v2026.02.06"``).  Defaults to today.

    Returns:
        Path to the output directory.
    """
    submissions_dir = Path(submissions_dir)
    output_dir = Path(output_dir)

    if not version:
        version = f"v{datetime.now().strftime('%Y.%m.%d')}"

    timestamp = datetime.now().isoformat()

    # Discover submission files
    yaml_files = sorted(submissions_dir.glob("*.yaml")) + sorted(
        submissions_dir.glob("*.yml")
    )
    logger.info("Found %d submission files in %s", len(yaml_files), submissions_dir)

    # Load and convert all submissions
    all_rows: List[Dict[str, Any]] = []
    model_cards: Dict[str, Dict[str, Any]] = {}  # keyed by model name for dedup

    for yaml_path in yaml_files:
        sub = _load_submission(yaml_path)
        if sub is None:
            continue

        rows = _submission_to_rows(sub, version, timestamp)
        all_rows.extend(rows)

        # Collect model card (last submission wins if duplicate model names)
        card = _submission_to_model_card(sub)
        model_cards[sub.model.name] = card

        if rows:
            logger.info(
                "  %s: %d metric rows (model=%s)",
                yaml_path.name,
                len(rows),
                sub.model.name,
            )

    if not all_rows:
        logger.warning("No valid submission results found. Writing empty snapshots.")

    return _write_snapshots(
        all_rows,
        list(model_cards.values()),
        output_dir,
        version,
    )


# ---------------------------------------------------------------------------
# Experiment-based aggregation (legacy path)
# ---------------------------------------------------------------------------


def _flatten_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Flatten nested metric dicts (e.g. prdc_f1.precision) into dot-notation."""
    flat: Dict[str, float] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, full_key))
        elif isinstance(value, (int, float)):
            flat[full_key if prefix else key] = float(value)
    return flat


def _load_experiment(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """Load a single experiment's eval results and config.

    Returns None if required files are missing or parsing fails.
    """
    eval_path = exp_dir / "results" / "eval_results.json"
    config_path = exp_dir / "config.yaml"

    if not eval_path.exists() or not config_path.exists():
        logger.debug("Skipping %s: missing eval_results.json or config.yaml", exp_dir)
        return None

    try:
        with open(eval_path, "r") as f:
            eval_data = json.load(f)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.warning("Failed to load experiment %s: %s", exp_dir, e)
        return None

    return {
        "eval_data": eval_data,
        "config": config,
        "exp_dir": exp_dir,
    }


def _experiment_to_rows(
    exp: Dict[str, Any],
    version: str,
    timestamp: str,
) -> List[Dict[str, Any]]:
    """Convert one experiment into leaderboard long-format rows."""
    from contsg.leaderboard.model_meta import BENCHMARK_MODEL_NAMES, get_model_meta

    config = exp["config"]
    eval_data = exp["eval_data"]

    model_name = config.get("model", {}).get("name", "")
    dataset_name = config.get("data", {}).get("name", "")

    if model_name not in BENCHMARK_MODEL_NAMES:
        logger.debug("Skipping non-benchmark model: %s", model_name)
        return []
    if dataset_name not in BENCHMARK_DATASET_NAMES:
        logger.debug("Skipping non-benchmark dataset: %s", dataset_name)
        return []

    condition_config = config.get("condition", {})
    try:
        condition_modality = infer_condition_modality(condition_config)
    except ValueError:
        logger.warning(
            "Cannot infer condition modality for %s/%s, skipping",
            model_name,
            dataset_name,
        )
        return []

    model_meta = get_model_meta(model_name)
    dataset_meta = get_dataset_meta(dataset_name)
    submission_id = f"{model_name}_{dataset_name}"

    raw_metrics = eval_data.get("metrics", {})
    flat_metrics = _flatten_metrics(raw_metrics)
    eval_timestamp = eval_data.get("timestamp", timestamp)

    rows: List[Dict[str, Any]] = []
    for metric_name, metric_value in flat_metrics.items():
        if metric_name not in METRIC_NAMES:
            logger.debug("Skipping unknown metric: %s", metric_name)
            continue

        metric_def = METRIC_BY_NAME[metric_name]
        rows.append(
            {
                "version": version,
                "submission_id": submission_id,
                "model": model_name,
                "model_type": model_meta.model_type,
                "dataset": dataset_name,
                "domain": dataset_meta.domain,
                "condition_modality": condition_modality,
                "semantic_level": dataset_meta.semantic_level,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "metric_group": metric_def.metric_group,
                "direction": metric_def.direction,
                "split": "test",
                "n_runs": 1,
                "seed_mean": metric_value,
                "seed_std": 0.0,
                "timestamp": eval_timestamp,
            }
        )

    return rows


def aggregate_experiments(
    experiments_dir: str | Path,
    output_dir: str | Path,
    version: str = "",
    *,
    include_patterns: Optional[List[str]] = None,
) -> Path:
    """Aggregate experiment results into 5 leaderboard snapshot files (legacy).

    Args:
        experiments_dir: Path to the experiments directory.
        output_dir: Where to write the snapshot files.
        version: Version string.  Defaults to today.
        include_patterns: Optional glob patterns to filter directories.

    Returns:
        Path to the output directory.
    """
    from contsg.leaderboard.model_meta import MODEL_META

    experiments_dir = Path(experiments_dir)
    output_dir = Path(output_dir)

    if not version:
        version = f"v{datetime.now().strftime('%Y.%m.%d')}"

    timestamp = datetime.now().isoformat()

    exp_dirs = sorted(
        d for d in experiments_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if include_patterns:
        import fnmatch

        exp_dirs = [
            d
            for d in exp_dirs
            if any(fnmatch.fnmatch(d.name, pat) for pat in include_patterns)
        ]

    logger.info("Found %d experiment directories in %s", len(exp_dirs), experiments_dir)

    all_rows: List[Dict[str, Any]] = []
    for exp_dir in exp_dirs:
        exp = _load_experiment(exp_dir)
        if exp is None:
            continue
        rows = _experiment_to_rows(exp, version, timestamp)
        all_rows.extend(rows)
        if rows:
            logger.info(
                "  %s: %d metric rows (%s / %s)",
                exp_dir.name,
                len(rows),
                rows[0]["model"],
                rows[0]["dataset"],
            )

    if not all_rows:
        logger.warning("No valid experiment results found. Writing empty snapshots.")

    # Model cards from hardcoded registry (legacy)
    model_card_rows = [
        {
            "model": meta.name,
            "org": meta.org,
            "model_link": meta.model_link,
            "code_link": meta.code_link,
            "paper_link": meta.paper_link,
            "params": meta.params,
            "model_type": meta.model_type,
            "native_condition": meta.native_condition,
            "training_data_compliance": meta.training_data_compliance,
            "testdata_leakage": meta.testdata_leakage,
            "replication_code_available": meta.replication_code_available,
            "notes": meta.notes,
        }
        for meta in MODEL_META.values()
    ]

    return _write_snapshots(all_rows, model_card_rows, output_dir, version)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Aggregate results into leaderboard snapshots."
    )
    parser.add_argument("input_dir", type=str, help="Path to submissions or experiments dir")
    parser.add_argument("output_dir", type=str, help="Path to write snapshot files")
    parser.add_argument("--version", type=str, default="", help="Version string")
    parser.add_argument(
        "--mode",
        choices=["submissions", "experiments"],
        default="submissions",
        help="Aggregation mode: 'submissions' (default) reads YAML files, "
        "'experiments' reads eval_results.json+config.yaml dirs.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        help="Glob patterns to filter directories (experiments mode only)",
    )

    args = parser.parse_args()

    if args.mode == "submissions":
        aggregate_submissions(args.input_dir, args.output_dir, version=args.version)
    else:
        aggregate_experiments(
            args.input_dir,
            args.output_dir,
            version=args.version,
            include_patterns=args.include,
        )
