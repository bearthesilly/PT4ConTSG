"""
Case study visualization for time series generation evaluation.

This module provides visualization utilities for comparing generated
time series against ground truth.
"""

from __future__ import annotations

import csv
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from contsg.config.schema import VizConfig

logger = logging.getLogger(__name__)


@dataclass
class VisualizationCase:
    """Single visualization case data."""

    global_id: int
    ts: np.ndarray  # (ts_len, num_vars) ground truth
    median: np.ndarray  # (ts_len, num_vars) median of multi-sample predictions
    std: np.ndarray  # (ts_len, num_vars) std of multi-sample predictions
    random_sample: np.ndarray  # (ts_len, num_vars) single random sample
    ts_len: int
    num_vars: int


class CaseStudyVisualizer:
    """
    Visualizer for case study comparison of generated vs ground truth time series.

    Features:
    - Random sample selection with seeded RNG
    - Median ± std band visualization
    - Random sample visualization
    - Combined figure output

    Usage:
        visualizer = CaseStudyVisualizer(config)
        visualizer.select_samples(test_size=1000)

        for batch in dataloader:
            visualizer.collect(batch_data, multi_preds, seen_so_far)
            seen_so_far += batch_size

        visualizer.render(output_dir)
    """

    def __init__(self, config: "VizConfig"):
        """
        Initialize visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config
        self.selected_ids: Set[int] = set()
        self.cases: List[VisualizationCase] = []
        self.random_sample_idx: int = 0
        self._rng = random.Random(config.seed)

    def reset(self) -> None:
        """Reset visualizer state."""
        self.selected_ids = set()
        self.cases = []
        self.random_sample_idx = 0
        self._rng = random.Random(self.config.seed)

    def select_samples(self, test_size: int, n_samples: int = 10) -> Set[int]:
        """
        Randomly select samples for visualization.

        Args:
            test_size: Total number of test samples
            n_samples: Number of generation samples (for random_sample_idx)

        Returns:
            Set of selected global IDs
        """
        k = min(self.config.k_cases, test_size)
        if k > 0:
            selected_list = sorted(self._rng.sample(range(test_size), k))
            self.selected_ids = set(selected_list)
            logger.info(
                f"[Viz] Selected {k} samples from {test_size}: "
                f"{selected_list[:min(5, len(selected_list))]}..."
            )
        else:
            self.selected_ids = set()

        # Generate global random sample index
        self.random_sample_idx = self._rng.randint(0, max(0, n_samples - 1)) if n_samples > 1 else 0
        logger.info(f"[Viz] Random sample index: {self.random_sample_idx}")

        return self.selected_ids

    def collect(
        self,
        batch_data: Dict[str, Any],
        multi_preds: Tensor,
        seen_so_far: int,
    ) -> None:
        """
        Collect visualization cases from batch.

        Args:
            batch_data: Batch data containing 'ts', 'pred', 'ts_len'
            multi_preds: Multi-sample predictions (n_samples, B, L, F)
            seen_so_far: Number of samples seen before this batch
        """
        if not self.selected_ids:
            return

        ts = batch_data["ts"]  # (B, L, F)
        pred = batch_data["pred"]  # (B, L, F) - median prediction
        ts_len = batch_data["ts_len"]  # (B,)

        batch_size = ts.shape[0]
        global_ids = list(range(seen_so_far, seen_so_far + batch_size))

        # Find which samples in this batch are selected
        pick_indices = [i for i, gid in enumerate(global_ids) if gid in self.selected_ids]

        if not pick_indices:
            return

        # Compute std from multi_preds
        std_pred = multi_preds.std(dim=0)  # (B, L, F)

        # Extract random sample
        random_sample_pred = multi_preds[self.random_sample_idx]  # (B, L, F)

        for i in pick_indices:
            L = int(ts_len[i].item())
            ts_np = ts[i].detach().cpu().numpy()[:L]
            median_np = pred[i].detach().cpu().numpy()[:L]
            std_np = std_pred[i].detach().cpu().numpy()[:L]
            random_np = random_sample_pred[i].detach().cpu().numpy()[:L]
            num_vars = ts_np.shape[-1]

            self.cases.append(VisualizationCase(
                global_id=global_ids[i],
                ts=ts_np,
                median=median_np,
                std=std_np,
                random_sample=random_np,
                ts_len=L,
                num_vars=num_vars,
            ))

        logger.debug(f"[Viz] Collected {len(pick_indices)} cases from batch, total: {len(self.cases)}")

    def render(self, output_dir: Path) -> None:
        """
        Render and save visualizations.

        Args:
            output_dir: Output directory for figures
        """
        if not self.cases:
            logger.warning("[Viz] No cases collected, skipping render")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine output format
        ext = self.config.output_type  # "png" or "pdf"

        # Render median mode (with std band)
        self._render_combined_figure(
            output_dir / f"all_cases_combined.{ext}",
            output_dir / "selection.csv",
            is_random=False,
        )

        # Render random sample mode
        self._render_combined_figure(
            output_dir / f"all_cases_combined_random.{ext}",
            output_dir / "selection_random.csv",
            is_random=True,
        )

        logger.info(f"[Viz] Saved {len(self.cases)} case study figures to {output_dir}")

    def _render_combined_figure(
        self,
        fig_path: Path,
        csv_path: Path,
        is_random: bool,
    ) -> None:
        """
        Render all cases in a single combined figure.

        Args:
            fig_path: Output figure path
            csv_path: Output CSV path
            is_random: If True, use random_sample; else use median with std
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_cases = len(self.cases)
        ncols = min(self.config.max_vars, max(c.num_vars for c in self.cases))
        nrows = n_cases

        fig_w = ncols * self.config.figsize_per_subplot[0]
        fig_h = nrows * self.config.figsize_per_subplot[1] + 1.0  # Extra for legend

        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

        for case_idx, case in enumerate(self.cases):
            n_vars = min(case.num_vars, ncols)
            x = np.arange(case.ts_len)

            for var_idx in range(ncols):
                ax = axes[case_idx, var_idx]

                if var_idx < n_vars:
                    real_y = case.ts[:, var_idx]
                    if is_random:
                        gen_y = case.random_sample[:, var_idx]
                        std_y = None
                    else:
                        gen_y = case.median[:, var_idx]
                        std_y = case.std[:, var_idx]

                    self._render_subplot(ax, x, real_y, gen_y, std_y, is_random)
                    ax.set_title(f"Case {case_idx} (ID {case.global_id}) - Var {var_idx}", fontsize=8)
                else:
                    ax.axis("off")

        # Add legend to first subplot
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10)

        mode_str = "Random Sample" if is_random else "Median ± Std"
        fig.suptitle(f"Case Study: Generated vs Ground Truth ({mode_str})", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        plt.savefig(fig_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)

        # Save CSV
        self._save_csv(csv_path, fig_path)

    def _render_subplot(
        self,
        ax,
        x: np.ndarray,
        real_y: np.ndarray,
        gen_y: np.ndarray,
        std_y: Optional[np.ndarray],
        is_random: bool,
    ) -> None:
        """
        Render a single subplot.

        Args:
            ax: Matplotlib axes
            x: Time axis
            real_y: Ground truth values
            gen_y: Generated values (median or random sample)
            std_y: Standard deviation (None for random mode)
            is_random: If True, no std band
        """
        linewidth = 1.0 if is_random else 1.5

        ax.plot(x, real_y, color="#1f77b4", linewidth=linewidth, label="Ground Truth")
        ax.plot(x, gen_y, color="#d62728", linewidth=linewidth, label="Generated")

        if std_y is not None:
            ax.fill_between(
                x,
                gen_y - std_y,
                gen_y + std_y,
                color="#d62728",
                alpha=self.config.alpha,
                label="Generated ±1σ",
            )

        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.tick_params(labelsize=7)

    def _save_csv(self, csv_path: Path, fig_path: Path) -> None:
        """Save case selection to CSV."""
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "global_id", "ts_len", "num_vars", "file_path"])
            for rank, case in enumerate(self.cases):
                writer.writerow([
                    rank,
                    case.global_id,
                    case.ts_len,
                    case.num_vars,
                    str(fig_path),
                ])


__all__ = ["VisualizationCase", "CaseStudyVisualizer"]
