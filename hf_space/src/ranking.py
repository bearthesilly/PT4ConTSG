"""
Ranking engine for the ConTSG-Bench leaderboard.

Implements percentile normalization and weighted scoring across metric groups.

Only **fidelity** and **adherence** groups participate in overall ranking.
Utility metrics are tracked in the data but excluded from ranking.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Ranking policies
# ---------------------------------------------------------------------------

# Policy weights: which metric groups contribute to the overall score.
# "Balanced" averages fidelity and adherence equally.
# "Fidelity-only" ranks solely by the fidelity group.
# "Adherence-only" ranks solely by the adherence group.
RANKING_POLICIES: Dict[str, Dict[str, float]] = {
    "Balanced": {
        "fidelity": 1.0,
        "adherence": 1.0,
    },
    "Fidelity-only": {
        "fidelity": 1.0,
    },
    "Adherence-only": {
        "adherence": 1.0,
    },
}

# Metric groups excluded from overall ranking (still normalized for display).
_EXCLUDED_GROUPS = {"utility"}


def normalize_metrics(df_long: pd.DataFrame) -> pd.DataFrame:
    """Apply percentile normalization to metric values.

    For each metric within each dataset, compute percentile rank across **all**
    models regardless of condition modality.  This enables direct cross-modality
    comparison (text vs. attribute vs. label), which is a core design principle
    of ConTSG-Bench: because each dataset provides aligned conditions across all
    three modalities, models using different condition types are directly
    comparable on the same data.

      - higher_better: score = percentile(metric_value)
      - lower_better:  score = 1 - percentile(metric_value)

    Args:
        df_long: Leaderboard long-format DataFrame.

    Returns:
        Copy of df_long with an added ``norm_score`` column (0-1, higher = better).
    """
    if df_long.empty:
        return df_long.assign(norm_score=pd.Series(dtype=float))

    df = df_long.copy()
    df["norm_score"] = 0.5  # default for single-entry groups

    for _, group_idx in df.groupby(
        ["metric_name", "dataset"]
    ).groups.items():
        if len(group_idx) <= 1:
            continue

        values = df.loc[group_idx, "metric_value"]
        direction = df.loc[group_idx[0], "direction"]
        ranked = values.rank(pct=True, method="average")

        if direction == "lower_better":
            df.loc[group_idx, "norm_score"] = 1.0 - ranked
        else:
            df.loc[group_idx, "norm_score"] = ranked

    return df


def compute_group_scores(df_normalized: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model, per-group average normalized scores.

    Args:
        df_normalized: Output of ``normalize_metrics`` (must have ``norm_score``).

    Returns:
        DataFrame with columns: model, metric_group, group_score, n_metrics.
    """
    if df_normalized.empty or "norm_score" not in df_normalized.columns:
        return pd.DataFrame(columns=["model", "metric_group", "group_score", "n_metrics"])

    group_scores = (
        df_normalized.groupby(["model", "metric_group"])
        .agg(
            group_score=("norm_score", "mean"),
            n_metrics=("norm_score", "count"),
        )
        .reset_index()
    )

    return group_scores


def compute_overall_ranking(
    df_long: pd.DataFrame,
    policy: str = "Balanced",
    metric_catalog: Optional[List[dict]] = None,
    model_cards: Optional[pd.DataFrame] = None,
    source_filter: str = "all",
) -> pd.DataFrame:
    """Compute overall model rankings under a given policy.

    Pipeline:
      1. Percentile-normalize each metric within its slice
      2. Average within each metric group -> group_score
      3. Arithmetic mean of fidelity and adherence group_scores -> overall_score
         (utility is excluded from ranking)
      4. Rank by overall_score (descending)

    Args:
        df_long: Leaderboard long-format DataFrame.
        policy: Name of the ranking policy (key in RANKING_POLICIES).
        metric_catalog: Optional metric catalog for coverage calculation.
        model_cards: Optional model cards DataFrame with native_condition / code metadata.
        source_filter: "all" (default), "open_source", or "closed_source".

    Returns:
        DataFrame with columns: rank, model, model_type, native_condition,
        overall_score, fidelity, adherence, coverage.
    """
    if df_long.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "model",
                "model_type",
                "native_condition",
                "replication_code_available",
                "overall_score",
                "fidelity",
                "adherence",
                "coverage",
                "code_link",
                "model_link",
            ]
        )

    weights = RANKING_POLICIES.get(policy, RANKING_POLICIES["Balanced"])

    # Step 1: Normalize all metrics (including utility, for potential display use)
    df_norm = normalize_metrics(df_long)

    # Step 2: Group scores (all groups)
    group_scores = compute_group_scores(df_norm)

    # Step 3: Pivot to wide format (model x group)
    pivot = group_scores.pivot_table(
        index="model", columns="metric_group", values="group_score", aggfunc="mean"
    ).reset_index()

    # Ensure ranking group columns exist
    for group in weights:
        if group not in pivot.columns:
            pivot[group] = 0.0

    # Step 4: Weighted overall score (only fidelity + adherence)
    total_weight = sum(weights.values())
    pivot["overall_score"] = sum(
        pivot[group] * w for group, w in weights.items()
    ) / total_weight

    # Add model_type
    model_types = df_long[["model", "model_type"]].drop_duplicates()
    pivot = pivot.merge(model_types, on="model", how="left")

    # Add metadata from model_cards when available
    if model_cards is not None and not model_cards.empty:
        merge_cols = ["model"]
        if "native_condition" in model_cards.columns:
            merge_cols.append("native_condition")
        if "code_link" in model_cards.columns:
            merge_cols.append("code_link")
        if "model_link" in model_cards.columns:
            merge_cols.append("model_link")
        if "replication_code_available" in model_cards.columns:
            merge_cols.append("replication_code_available")
        meta = model_cards[merge_cols].drop_duplicates(subset=["model"])
        pivot = pivot.merge(meta, on="model", how="left")

    if "native_condition" not in pivot.columns:
        pivot["native_condition"] = ""
    if "code_link" not in pivot.columns:
        pivot["code_link"] = ""
    else:
        pivot["code_link"] = pivot["code_link"].fillna("")
    if "model_link" not in pivot.columns:
        pivot["model_link"] = ""
    else:
        pivot["model_link"] = pivot["model_link"].fillna("")
    if "replication_code_available" not in pivot.columns:
        pivot["replication_code_available"] = (
            pivot["code_link"].astype(str).str.strip() != ""
        )
    else:
        pivot["replication_code_available"] = (
            pivot["replication_code_available"].fillna(False).astype(bool)
        )

    # Optional filtering by open-source availability
    if source_filter == "open_source":
        pivot = pivot[pivot["code_link"].astype(str).str.strip() != ""].copy()
    elif source_filter == "closed_source":
        pivot = pivot[pivot["code_link"].astype(str).str.strip() == ""].copy()

    # Coverage: fraction of metrics reported vs total possible
    if metric_catalog:
        total_metrics = len(metric_catalog)
        metrics_per_model = (
            df_long.groupby("model")["metric_name"].nunique().reset_index()
        )
        metrics_per_model.columns = ["model", "n_reported"]
        pivot = pivot.merge(metrics_per_model, on="model", how="left")
        pivot["coverage"] = (pivot["n_reported"] / total_metrics).round(3)
        pivot = pivot.drop(columns=["n_reported"])
    else:
        pivot["coverage"] = 1.0

    # Sort and rank
    pivot = pivot.sort_values("overall_score", ascending=False).reset_index(drop=True)
    pivot["rank"] = range(1, len(pivot) + 1)

    # Round scores for display
    score_cols = ["overall_score", "fidelity", "adherence"]
    for col in score_cols:
        if col in pivot.columns:
            pivot[col] = pivot[col].round(4)

    # Reorder columns — exclude utility from ranking display
    display_cols = [
        "rank",
        "model",
        "model_type",
        "native_condition",
        "replication_code_available",
        "overall_score",
        "fidelity",
        "adherence",
        "coverage",
        "code_link",
        "model_link",
    ]
    display_cols = [c for c in display_cols if c in pivot.columns]

    return pivot[display_cols]
