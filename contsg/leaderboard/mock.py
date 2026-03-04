"""
Mock data generator for ConTSG-Bench leaderboard development.

Generates realistic mock evaluation results for all model × dataset
combinations, then runs the aggregation pipeline to produce snapshot files.

Usage:
    python -m contsg.leaderboard.mock ./snapshots
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from contsg.leaderboard.dataset_meta import DATASET_META
from contsg.leaderboard.metric_catalog import METRIC_BY_NAME, METRIC_CATALOG, MetricDef
from contsg.leaderboard.model_meta import MODEL_META

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Realistic metric value ranges (based on typical benchmark results)
# ---------------------------------------------------------------------------

# (min, max) for each metric — values are sampled uniformly
METRIC_RANGES: Dict[str, tuple] = {
    # Fidelity (lower is better for most)
    "acd": (0.01, 0.50),
    "sd": (0.01, 0.80),
    "kd": (0.01, 1.00),
    "mdd": (0.01, 0.60),
    "fid": (5.0, 150.0),
    "prdc_f1.precision": (0.10, 0.95),  # higher is better
    "prdc_f1.recall": (0.10, 0.95),  # higher is better
    # Adherence (lower is better for frechet-based)
    "jftsd": (3.0, 120.0),
    "joint_prdc_f1.precision": (0.05, 0.85),  # higher is better
    "joint_prdc_f1.recall": (0.05, 0.85),  # higher is better
    "cttp": (0.10, 0.90),  # higher is better
    # Fine-grained (higher is better)
    "segment_param_acc": (0.05, 0.80),
    # Utility (lower is better)
    "dtw": (0.5, 15.0),
    "crps": (0.05, 1.50),
    "ed": (0.1, 5.0),
    "wape": (0.1, 3.0),
}

# Model "quality tiers" — better models get values shifted toward the good end
MODEL_QUALITY: Dict[str, float] = {
    "verbalts": 0.80,
    "t2s": 0.70,
    "bridge": 0.65,
    "diffusets": 0.60,
    "timeweaver": 0.55,
    "wavestitch": 0.50,
    "tedit": 0.55,
    "timevqvae": 0.45,
    "ttscgan": 0.35,
    "text2motion": 0.50,
    "retrieval": 0.30,
}

# Which condition modality each model uses
MODEL_CONDITION: Dict[str, str] = {
    "verbalts": "text",
    "t2s": "text",
    "bridge": "text",
    "diffusets": "text",
    "retrieval": "text",
    "text2motion": "text",
    "timeweaver": "attribute",
    "wavestitch": "attribute",
    "tedit": "attribute",
    "ttscgan": "label",
    "timevqvae": "label",
}


def _sample_metric_value(
    metric: MetricDef,
    quality: float,
    rng: random.Random,
) -> float:
    """Sample a realistic metric value given model quality (0–1 scale)."""
    lo, hi = METRIC_RANGES.get(metric.metric_name, (0.0, 1.0))

    if metric.direction == "lower_better":
        # High quality → lower values
        center = lo + (hi - lo) * (1.0 - quality)
    else:
        # High quality → higher values
        center = lo + (hi - lo) * quality

    # Add noise (±15% of range)
    noise = rng.gauss(0, (hi - lo) * 0.08)
    value = center + noise
    value = max(lo * 0.5, min(hi * 1.2, value))  # soft clamp

    return round(value, 6)


def generate_mock_experiments(
    output_dir: str | Path,
    *,
    seed: int = 42,
    n_runs: int = 3,
    skip_combinations: Optional[List[tuple]] = None,
) -> Path:
    """Generate mock experiment directories with eval_results.json + config.yaml.

    Creates a directory structure compatible with the aggregation pipeline:
        output_dir/
          mock_<model>_<dataset>/
            config.yaml
            results/
              eval_results.json

    Args:
        output_dir: Where to write mock experiment directories.
        seed: Random seed for reproducibility.
        n_runs: Number of runs to simulate (for seed_mean/std).
        skip_combinations: List of (model, dataset) tuples to skip.

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    skip = set(skip_combinations or [])
    n_created = 0

    for model_name, model_meta in MODEL_META.items():
        condition = MODEL_CONDITION.get(model_name, "text")
        quality = MODEL_QUALITY.get(model_name, 0.5)

        for dataset_name, dataset_meta in DATASET_META.items():
            if (model_name, dataset_name) in skip:
                continue

            exp_name = f"mock_{model_name}_{dataset_name}"
            exp_dir = output_dir / exp_name
            results_dir = exp_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            # Generate metrics
            metrics: Dict[str, Any] = {}
            for metric_def in METRIC_CATALOG:
                mname = metric_def.metric_name

                # Skip segment_param_acc for non-segment datasets
                if mname == "segment_param_acc" and dataset_name != "telecomts_segment":
                    continue

                # For dict-style metrics (prdc_f1.*, joint_prdc_f1.*), nest them
                if "." in mname:
                    parent, child = mname.split(".", 1)
                    if parent not in metrics:
                        metrics[parent] = {}
                    metrics[parent][child] = _sample_metric_value(
                        metric_def, quality, rng
                    )
                else:
                    metrics[mname] = _sample_metric_value(metric_def, quality, rng)

            # Write eval_results.json
            eval_data = {
                "timestamp": datetime.now().isoformat(),
                "experiment": str(exp_dir),
                "metrics": metrics,
            }
            with open(results_dir / "eval_results.json", "w") as f:
                json.dump(eval_data, f, indent=2)

            # Write config.yaml
            config = {
                "model": {"name": model_name},
                "data": {
                    "name": dataset_name,
                    "data_folder": f"./datasets/{dataset_name}",
                    "n_var": dataset_meta.n_var,
                    "seq_length": dataset_meta.seq_length,
                },
                "condition": {
                    "text": {"enabled": condition == "text"},
                    "attribute": {"enabled": condition == "attribute"},
                    "label": {"enabled": condition == "label"},
                },
            }
            with open(exp_dir / "config.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            n_created += 1

    logger.info("Generated %d mock experiment directories in %s", n_created, output_dir)
    return output_dir


def generate_mock_snapshots(
    output_dir: str | Path,
    *,
    seed: int = 42,
    version: str = "",
) -> Path:
    """Generate mock snapshot files (end-to-end: mock experiments → aggregate).

    This is the main entry point for generating development data for the
    HF Space frontend.

    Args:
        output_dir: Where to write the 5 snapshot files.
        seed: Random seed.
        version: Version string.

    Returns:
        Path to the snapshot output directory.
    """
    from contsg.leaderboard.aggregate import aggregate_experiments

    output_dir = Path(output_dir)

    # Step 1: Generate mock experiments in a temp subdirectory
    mock_experiments_dir = output_dir / "_mock_experiments"
    generate_mock_experiments(mock_experiments_dir, seed=seed)

    # Step 2: Run aggregation pipeline
    snapshots_dir = output_dir / "snapshots"
    aggregate_experiments(mock_experiments_dir, snapshots_dir, version=version)

    logger.info("Mock snapshots generated in %s", snapshots_dir)
    return snapshots_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate mock leaderboard data.")
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for mock data",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--version", type=str, default="", help="Version string")

    args = parser.parse_args()
    generate_mock_snapshots(args.output_dir, seed=args.seed, version=args.version)
