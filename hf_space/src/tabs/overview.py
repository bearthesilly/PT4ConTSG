"""
Overview Tab — comprehensive leaderboard with ranking policy switching.

Displays overall model rankings with per-group sub-scores and coverage.
Supports switching between Balanced / Fidelity-only / Adherence-only policies.
"""

from __future__ import annotations

from typing import Optional

import gradio as gr
import pandas as pd

from src.data_loader import SnapshotData
from src.ranking import RANKING_POLICIES, compute_overall_ranking


def _build_ranking_table(
    data: SnapshotData,
    policy: str,
    source_filter: str,
) -> pd.DataFrame:
    """Build the ranking table for the given policy."""
    return compute_overall_ranking(
        data.leaderboard_long,
        policy=policy,
        metric_catalog=data.metric_catalog,
        model_cards=data.model_cards,
        source_filter=source_filter,
    )


def _format_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format the ranking table for Gradio display."""
    display = df.copy()

    # Render model + optional links in one cell, e.g.:
    # verbalts ([code](...), [weights](...))
    def _fmt_model(row: pd.Series) -> str:
        name = str(row.get("model", ""))
        code_link = str(row.get("code_link", "")).strip()
        model_link = str(row.get("model_link", "")).strip()

        links = []
        if code_link:
            links.append(f"[code]({code_link})")
        if model_link:
            links.append(f"[weights]({model_link})")

        if links:
            return f"{name} ({', '.join(links)})"
        return name

    display["model"] = display.apply(_fmt_model, axis=1)

    # Rename columns for display
    rename_map = {
        "rank": "Rank",
        "model": "Model",
        "model_type": "Type",
        "native_condition": "Condition",
        "replication_code_available": "Replication Code",
        "overall_score": "Overall",
        "fidelity": "Fidelity",
        "adherence": "Adherence",
        "coverage": "Coverage",
    }
    display = display.rename(columns=rename_map)

    # Keep model_type in backend ranking output, but hide Type in UI table.
    if "Type" in display.columns:
        display = display.drop(columns=["Type"])

    if "Replication Code" in display.columns:
        display["Replication Code"] = display["Replication Code"].map(
            lambda x: "Yes" if bool(x) else "No"
        )

    # Format coverage as percentage
    if "Coverage" in display.columns:
        display["Coverage"] = (display["Coverage"] * 100).round(1).astype(str) + "%"

    if "code_link" in display.columns:
        display = display.drop(columns=["code_link"])
    if "model_link" in display.columns:
        display = display.drop(columns=["model_link"])

    preferred_order = [
        "Rank",
        "Model",
        "Condition",
        "Replication Code",
        "Overall",
        "Fidelity",
        "Adherence",
        "Coverage",
    ]
    existing = [c for c in preferred_order if c in display.columns]
    remaining = [c for c in display.columns if c not in existing]
    display = display[existing + remaining]

    return display


def create_overview_tab(data: SnapshotData) -> gr.Blocks:
    """Create the Overview tab component."""

    with gr.Blocks() as tab:
        gr.Markdown(
            f"""
## Overall Leaderboard

Comprehensive ranking of **{len(data.models)} models** across
**{len(data.datasets)} datasets** and **{len(data.metric_catalog)} metrics**.

Scores are percentile-normalized within each (metric, dataset, condition) slice,
then averaged per metric group and weighted according to the selected policy.

> **Version**: `{data.version}`
"""
        )

        with gr.Row():
            policy_dropdown = gr.Dropdown(
                choices=list(RANKING_POLICIES.keys()),
                value="Balanced",
                label="Ranking Policy",
                info="Controls how metric group scores are weighted.",
                interactive=True,
            )
            source_filter = gr.Dropdown(
                choices=[
                    ("All models", "all"),
                    ("Open-source only", "open_source"),
                ],
                value="all",
                label="Source Filter",
                info="Filter by code availability in the model card.",
                interactive=True,
            )

        # Initial table
        initial_df = _format_table_for_display(
            _build_ranking_table(data, "Balanced", "all")
        )

        ranking_table = gr.Dataframe(
            value=initial_df,
            label="Model Rankings",
            interactive=False,
            wrap=True,
            datatype=[
                "number",   # Rank
                "markdown", # Model
                "str",      # Condition
                "str",      # Replication Code
                "number",   # Overall
                "number",   # Fidelity
                "number",   # Adherence
                "str",      # Coverage
            ],
        )

        # Update on control change
        def update_ranking(policy: str, filter_mode: str) -> pd.DataFrame:
            df = _build_ranking_table(data, policy, filter_mode)
            return _format_table_for_display(df)

        policy_dropdown.change(
            fn=update_ranking,
            inputs=[policy_dropdown, source_filter],
            outputs=[ranking_table],
        )
        source_filter.change(
            fn=update_ranking,
            inputs=[policy_dropdown, source_filter],
            outputs=[ranking_table],
        )

        # Methodology and column explanations
        with gr.Accordion("How are scores computed?", open=False):
            gr.Markdown(
                f"""
**Scoring Methodology:**

1. **Percentile Normalization** — For each metric, models are compared within the same
   *(metric, dataset, condition)* slice. Raw values are converted to percentile ranks
   (0–1, higher = better). Lower-is-better metrics are inverted so that higher scores
   always indicate better performance.

2. **Group Score** — All normalized scores belonging to the same metric group
   (Fidelity or Adherence) are averaged to produce a single group score per model.

3. **Overall Score** — Determined by the selected ranking policy:
   - *Balanced*: average of Fidelity and Adherence group scores.
   - *Fidelity-only*: equals the Fidelity group score.
   - *Adherence-only*: equals the Adherence group score.

4. **Coverage** — Fraction of benchmark metrics that the model reports results for
   (out of {len(data.metric_catalog)} total metrics). Models that skip certain metrics
   will show < 100%.

**Column Definitions:**

| Column | Description |
|--------|-------------|
| **Rank** | Position based on Overall score (descending) |
| **Model** | Model identifier with optional links: `code`, `weights` |
| **Condition** | Condition modalities used (text, attribute, label) |
| **Replication Code** | Whether the model card provides an open-source code link |
| **Overall** | Aggregated score under the selected ranking policy (0–1) |
| **Fidelity** | Average percentile score across {_count_metrics(data, 'fidelity')} fidelity metrics (ACD, SD, KD, MDD, FID, kNN-PRF Precision/Recall) |
| **Adherence** | Average percentile score across {_count_metrics(data, 'adherence')} adherence metrics (JFTSD, Joint kNN-PRF Precision/Recall, CTTP) |
| **Coverage** | Percentage of benchmark metrics reported |
"""
            )

    return tab


def _count_metrics(data: SnapshotData, group: str) -> int:
    """Count the number of metrics in a given group."""
    return sum(1 for m in data.metric_catalog if m["metric_group"] == group)
