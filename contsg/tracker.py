"""
Experiment tracker for ConTSG.

This module provides experiment tracking functionality including:
- Configuration snapshots
- Git commit tracking
- Checkpoint management
- Result logging
"""

from __future__ import annotations

import json
import os
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback

from contsg.config.model_validation import validate_model_config
from contsg.config.schema import ExperimentConfig


class ExperimentTracker:
    """
    Experiment tracker that manages:
    - Experiment directory creation
    - Configuration snapshots
    - Git information capture
    - Lightning callbacks for checkpointing and early stopping
    - Result logging

    Example:
        config = ExperimentConfig(...)
        tracker = ExperimentTracker(config)
        tracker.start()

        # Train with Lightning
        trainer = pl.Trainer(
            default_root_dir=tracker.experiment_dir,
            callbacks=tracker.get_callbacks(),
        )

        tracker.finish(trainer.checkpoint_callback.best_model_path)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize the experiment tracker.

        Args:
            config: Experiment configuration
            experiment_name: Optional custom experiment name
        """
        self.config = config
        self.experiment_id = self._generate_id(experiment_name)
        self.experiment_dir = config.output_dir / self.experiment_id

        self._started = False
        self._finished = False

    def _generate_id(self, custom_name: Optional[str] = None) -> str:
        """
        Generate unique experiment ID.

        Format: {timestamp}_{dataset}_{model}[_{custom_name}]
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{timestamp}_{self.config.data.name}_{self.config.model.name}"

        if custom_name:
            base_name = f"{base_name}_{custom_name}"

        return base_name

    def start(self) -> "ExperimentTracker":
        """
        Start the experiment.

        Creates experiment directory, saves configuration and Git info.

        Returns:
            self for chaining
        """
        if self._started:
            return self

        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.experiment_dir / "checkpoints").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)
        (self.experiment_dir / "results").mkdir(exist_ok=True)

        # Save configuration
        self._save_config()

        # Save Git information
        self._save_git_info()

        # Save experiment metadata
        self._save_metadata("started")

        self._started = True
        return self

    def _save_config(self) -> None:
        """Save experiment configuration to YAML file."""
        config_path = self.experiment_dir / "config.yaml"
        self.config.to_yaml(config_path)

    def _save_git_info(self) -> None:
        """Save Git commit information."""
        git_info = self._get_git_info()
        git_path = self.experiment_dir / "git_info.json"

        with open(git_path, "w", encoding="utf-8") as f:
            json.dump(git_info, f, indent=2, ensure_ascii=False)

    def _get_git_info(self) -> Dict[str, Any]:
        """
        Get current Git repository information.

        Returns:
            Dictionary with commit hash, branch, status, and diff
        """
        try:
            # Get commit hash
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()

            # Get branch name
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()

            # Get short status
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()

            # Get diff stat
            diff_stat = subprocess.check_output(
                ["git", "diff", "--stat"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()

            # Get remote URL (if available)
            try:
                remote_url = subprocess.check_output(
                    ["git", "remote", "get-url", "origin"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except subprocess.CalledProcessError:
                remote_url = None

            return {
                "commit": commit,
                "commit_short": commit[:8],
                "branch": branch,
                "has_uncommitted_changes": bool(status),
                "status": status if status else None,
                "diff_stat": diff_stat if diff_stat else None,
                "remote_url": remote_url,
                "captured_at": datetime.now().isoformat(),
            }

        except subprocess.CalledProcessError:
            return {
                "error": "Not a git repository or git not available",
                "captured_at": datetime.now().isoformat(),
            }
        except FileNotFoundError:
            return {
                "error": "Git command not found",
                "captured_at": datetime.now().isoformat(),
            }

    def _save_metadata(self, status: str) -> None:
        """Save experiment metadata."""
        metadata = {
            "experiment_id": self.experiment_id,
            "status": status,
            "dataset": self.config.data.name,
            "model": self.config.model.name,
            "created_at": self.config.created_at.isoformat() if self.config.created_at else None,
            "updated_at": datetime.now().isoformat(),
        }

        metadata_path = self.experiment_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def get_callbacks(self) -> List[Callback]:
        """
        Get Lightning callbacks for experiment tracking.

        Returns:
            List of callbacks including ModelCheckpoint and EarlyStopping
        """
        callbacks: List[Callback] = []

        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.experiment_dir / "checkpoints",
            filename="{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

        # Early stopping callback (if patience > 0)
        if self.config.train.early_stopping_patience > 0:
            early_stopping = EarlyStopping(
                monitor="val/loss",
                patience=self.config.train.early_stopping_patience,
                mode="min",
                verbose=True,
            )
            callbacks.append(early_stopping)

        return callbacks

    def finish(
        self,
        best_model_path: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Finish the experiment.

        Args:
            best_model_path: Path to best model checkpoint
            metrics: Optional final metrics to log
        """
        if self._finished:
            return

        # Save final summary
        summary = {
            "status": "completed",
            "finished_at": datetime.now().isoformat(),
            "best_checkpoint": str(best_model_path) if best_model_path else None,
        }

        if metrics:
            summary["final_metrics"] = metrics

        summary_path = self.experiment_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Update metadata
        self._save_metadata("completed")

        self._finished = True

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to a JSON file.

        Args:
            metrics: Dictionary of metric values
            step: Optional step/epoch number
        """
        metrics_path = self.experiment_dir / "results" / "metrics.jsonl"

        entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        if step is not None:
            entry["step"] = step

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def save_results(self, results: Dict[str, Any], filename: str = "results.json") -> None:
        """
        Save evaluation results to JSON file.

        Args:
            results: Results dictionary
            filename: Output filename
        """
        results_path = self.experiment_dir / "results" / filename

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    @classmethod
    def from_experiment_dir(
        cls,
        experiment_dir: Path | str,
    ) -> "ExperimentTracker":
        """
        Load tracker from existing experiment directory.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            ExperimentTracker instance
        """
        experiment_dir = Path(experiment_dir)

        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

        config_path = experiment_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = ExperimentConfig.from_yaml(config_path)
        config = validate_model_config(config, strict_schema=False).config

        tracker = cls(config)
        tracker.experiment_dir = experiment_dir
        tracker.experiment_id = experiment_dir.name
        tracker._started = True

        return tracker

    def get_best_checkpoint(self) -> Optional[Path]:
        """
        Get the path to the best checkpoint.

        Returns:
            Path to best checkpoint or None if not available
        """
        summary_path = self.experiment_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
                if summary.get("best_checkpoint"):
                    return Path(summary["best_checkpoint"])

        # Fallback: look for last.ckpt
        last_ckpt = self.experiment_dir / "checkpoints" / "last.ckpt"
        if last_ckpt.exists():
            return last_ckpt

        return None


def list_experiments(
    output_dir: Path | str = "experiments",
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List all experiments in the output directory.

    Args:
        output_dir: Path to experiments directory
        status: Filter by status ("completed", "running", etc.)

    Returns:
        List of experiment metadata dictionaries
    """
    output_dir = Path(output_dir)
    experiments = []

    if not output_dir.exists():
        return experiments

    for exp_dir in sorted(output_dir.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue

        metadata_path = exp_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        with open(metadata_path) as f:
            metadata = json.load(f)

        if status and metadata.get("status") != status:
            continue

        metadata["path"] = str(exp_dir)
        experiments.append(metadata)

    return experiments
