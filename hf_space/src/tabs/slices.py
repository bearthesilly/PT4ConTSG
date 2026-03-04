"""
Slices Tab — multi-dimensional filtering and comparison.

Supports filtering by dataset, condition modality, semantic level, and
metric group. Displays a filtered leaderboard table.
"""

from __future__ import annotations

from typing import List

import gradio as gr
import pandas as pd

from src.data_loader import SnapshotData


def _filter_data(
    data: SnapshotData,
    datasets: List[str],
    native_conditions: List[str],
    semantic_levels: List[str],
    metric_groups: List[str],
) -> pd.DataFrame:
    """Filter the long-format leaderboard by selected dimensions."""
    df = data.leaderboard_long.copy()

    if datasets:
        df = df[df["dataset"].isin(datasets)]
    if native_conditions and not data.model_cards.empty and "native_condition" in data.model_cards.columns:
        selected_models = data.model_cards[
            data.model_cards["native_condition"].isin(native_conditions)
        ]["model"].unique()
        df = df[df["model"].isin(selected_models)]
    if semantic_levels:
        df = df[df["semantic_level"].isin(semantic_levels)]
    if metric_groups:
        df = df[df["metric_group"].isin(metric_groups)]

    return df


def _build_comparison_table(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """Build a model comparison table from filtered data."""
    if df_filtered.empty:
        return pd.DataFrame(columns=["Model", "Metric", "Value", "Group", "Direction"])

    display = df_filtered[
        ["model", "metric_name", "metric_value", "metric_group", "direction"]
    ].copy()
    display.columns = ["Model", "Metric", "Value", "Group", "Direction"]
    display["Value"] = display["Value"].round(4)

    # Pivot for a cleaner view
    try:
        pivot = display.pivot_table(
            index="Model",
            columns="Metric",
            values="Value",
            aggfunc="mean",
        ).reset_index()
        # Clear columns axis name to avoid an extra "Metric" header cell in HTML output.
        pivot.columns.name = None
        # Round all numeric columns and format as compact strings to avoid clipped display.
        for col in pivot.columns:
            if col != "Model":
                pivot[col] = pivot[col].map(
                    lambda x: "" if pd.isna(x) else f"{float(x):.4f}".rstrip("0").rstrip(".")
                )
        return pivot
    except Exception:
        return display


def _build_comparison_html(df_filtered: pd.DataFrame) -> str:
    """Render the filtered comparison table as HTML with horizontal scrolling."""
    table = _build_comparison_table(df_filtered)
    if table.empty:
        return "<div id='slices-results-table'>No results.</div>"

    html_table = table.to_html(index=False, classes="slices-grid", border=0, escape=True)
    return f"<div id='slices-results-table' class='slices-results-wrap'>{html_table}</div>"


def create_slices_tab(data: SnapshotData) -> gr.Blocks:
    """Create the Slices tab component."""

    if not data.model_cards.empty and "native_condition" in data.model_cards.columns:
        condition_choices = sorted(data.model_cards["native_condition"].dropna().unique().tolist())
    else:
        # Fallback for old snapshots without native_condition in model cards.
        condition_choices = data.condition_modalities

    with gr.Blocks() as tab:
        gr.Markdown(
            """
## Slice Explorer

Filter the leaderboard by dataset, condition modality, semantic level,
and metric group.
"""
        )

        # Filters
        with gr.Row():
            dataset_filter = gr.Dropdown(
                choices=data.datasets,
                value=[],
                label="Datasets",
                multiselect=True,
                info="Filter by dataset (empty = all)",
            )
            modality_filter = gr.Dropdown(
                choices=condition_choices,
                value=[],
                label="Condition Modality (Native)",
                multiselect=True,
                info="Filter by model native condition type",
            )

        with gr.Row():
            semantic_filter = gr.Dropdown(
                choices=data.semantic_levels,
                value=[],
                label="Semantic Level",
                multiselect=True,
                info="Filter by semantic abstraction",
            )
            group_filter = gr.Dropdown(
                choices=data.metric_groups,
                value=[],
                label="Metric Group",
                multiselect=True,
                info="Filter by metric group",
            )

        # Outputs
        gr.Markdown("Filtered Results")
        comparison_table = gr.HTML(
            value=_build_comparison_html(data.leaderboard_long),
        )

        # Update handler
        def update_slices(
            datasets: List[str],
            modalities: List[str],
            semantics: List[str],
            groups: List[str],
        ):
            df_filtered = _filter_data(data, datasets, modalities, semantics, groups)
            table = _build_comparison_html(df_filtered)
            return table

        # Wire up all filter changes
        filter_inputs = [
            dataset_filter,
            modality_filter,
            semantic_filter,
            group_filter,
        ]
        filter_outputs = [comparison_table]

        for inp in filter_inputs:
            inp.change(
                fn=update_slices,
                inputs=filter_inputs,
                outputs=filter_outputs,
            )

    return tab
