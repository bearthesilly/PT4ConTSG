"""
ConTSG-Bench Leaderboard — Hugging Face Space entry point.

This Gradio app displays the ConTSG-Bench leaderboard with three tabs:
  1. Overview — Overall model rankings with policy switching
  2. Slices — Multi-dimensional filtering and radar chart comparison
  3. Submission & Version — Submission guide and version history

Usage (local development):
    # First generate mock data:
    python -m contsg.leaderboard.mock ./hf_space/data

    # Then run the Space:
    cd hf_space && python app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import gradio as gr

# Add parent directory to path so we can import src.*
sys.path.insert(0, str(Path(__file__).parent))

import src.state as state
from src.tabs.overview import create_overview_tab
from src.tabs.slices import create_slices_tab
from src.tabs.submission import create_submission_tab

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


CUSTOM_CSS = """
.top-link-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 10px 0 18px;
}

.top-link-badge {
  align-items: center;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 999px;
  color: inherit;
  display: inline-flex;
  gap: 8px;
  padding: 8px 14px;
  text-decoration: none;
  transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
}

.top-link-badge:hover {
  background: rgba(255, 255, 255, 0.1);
  border-color: rgba(255, 255, 255, 0.24);
  transform: translateY(-1px);
}

.top-link-icon {
  align-items: center;
  display: inline-flex;
  justify-content: center;
  min-width: 18px;
}

.top-link-icon img {
  display: block;
  height: 18px;
  width: auto;
}

.top-link-icon--github img {
  width: 18px;
}

.top-link-icon--arxiv img {
  height: 16px;
}

.top-link-label {
  font-size: 14px;
  font-weight: 600;
}

/* Keep metric values fully visible in the slices dataframe */
#slices-results-table {
  width: 100%;
  overflow-x: auto;
}

#slices-results-table .slices-grid {
  border-collapse: collapse;
  width: max-content;
  min-width: 100%;
  table-layout: auto;
}

#slices-results-table .slices-grid th,
#slices-results-table .slices-grid td,
#slices-results-table .slices-grid th *,
#slices-results-table .slices-grid td * {
  white-space: nowrap !important;
  word-break: keep-all !important;
  overflow-wrap: normal !important;
  text-overflow: clip !important;
}

#slices-results-table .slices-grid th,
#slices-results-table .slices-grid td {
  padding: 6px 8px;
}
"""

HEADER_MARKDOWN = """
# ConTSG-Bench Leaderboard

**A Comprehensive Benchmark for Conditional Time Series Generation**

Evaluating generative models across **Fidelity** and **Condition Adherence**
"""

HEADER_LINKS_HTML = """
<div class="top-link-row">
  <a class="top-link-badge" href="https://github.com/seqml/ConTSG-Bench" target="_blank" rel="noopener noreferrer">
    <span class="top-link-icon top-link-icon--github">
      <img src="https://cdn.simpleicons.org/github/111111" alt="GitHub icon">
    </span>
    <span class="top-link-label">GitHub</span>
  </a>
  <a class="top-link-badge" href="https://arxiv.org/abs/2603.04767" target="_blank" rel="noopener noreferrer">
    <span class="top-link-icon top-link-icon--arxiv">
      <img src="https://upload.wikimedia.org/wikipedia/commons/b/bc/ArXiv_logo_2022.svg" alt="arXiv icon">
    </span>
    <span class="top-link-label">arXiv</span>
  </a>
</div>
"""


def find_snapshot_dir() -> Path:
    """Find the snapshot data directory.

    Search order:
      1. ./data/snapshots/    (local dev, from mock generator)
      2. ./snapshots/         (direct snapshot dir)
      3. ./data/              (flat layout)
    """
    candidates = [
        Path("data/snapshots"),
        Path("snapshots"),
        Path("data"),
    ]
    for candidate in candidates:
        if (candidate / "leaderboard_long.parquet").exists():
            return candidate

    raise FileNotFoundError(
        "Cannot find snapshot data. Run the mock data generator first:\n"
        "  python -m contsg.leaderboard.mock ./hf_space/data\n"
        "Expected file: <dir>/leaderboard_long.parquet"
    )


def create_app() -> gr.Blocks:
    """Build the Gradio application."""
    snapshot_dir = find_snapshot_dir()
    data = state.init(snapshot_dir)

    with gr.Blocks(
        title="ConTSG-Bench Leaderboard",
    ) as app:
        gr.Markdown(HEADER_MARKDOWN)
        gr.HTML(HEADER_LINKS_HTML)

        with gr.Tabs():
            with gr.Tab("Overview"):
                create_overview_tab(data)

            with gr.Tab("Slices"):
                create_slices_tab(data)

            with gr.Tab("Submission & Version"):
                create_submission_tab(data)

    return app


if __name__ == "__main__":
    app = create_app()
    try:
        # Gradio 6.x moved some constructor options (such as css) to launch().
        app.launch(css=CUSTOM_CSS)
    except TypeError:
        # Backward compatibility for older Gradio versions.
        app.launch()
