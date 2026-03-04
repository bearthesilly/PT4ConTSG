"""
Snapshot data loader for the ConTSG-Bench HF Space.

Reads the 5 snapshot files (parquet + json) from a local directory or
HF Dataset and provides them as pandas DataFrames / dicts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class SnapshotData:
    """Container for all loaded snapshot data."""

    def __init__(self, snapshot_dir: str | Path):
        self.snapshot_dir = Path(snapshot_dir)
        self._load()

    def _load(self) -> None:
        """Load all snapshot files."""
        d = self.snapshot_dir

        # Parquet files
        self.leaderboard_long: pd.DataFrame = self._read_parquet(
            d / "leaderboard_long.parquet"
        )
        self.leaderboard_wide: pd.DataFrame = self._read_parquet(
            d / "leaderboard_wide.parquet"
        )
        self.model_cards: pd.DataFrame = self._read_parquet(d / "model_cards.parquet")

        # JSON files
        self.metric_catalog: List[Dict[str, Any]] = self._read_json(
            d / "metric_catalog.json"
        )
        self.version_manifest: Dict[str, Any] = self._read_json(
            d / "version_manifest.json"
        )

        # Derived lookups
        self.metric_lookup: Dict[str, Dict[str, Any]] = {
            m["metric_name"]: m for m in self.metric_catalog
        }

        logger.info(
            "Loaded snapshot: %d long rows, %d wide rows, %d models, %d metrics",
            len(self.leaderboard_long),
            len(self.leaderboard_wide),
            len(self.model_cards),
            len(self.metric_catalog),
        )

    def _read_parquet(self, path: Path) -> pd.DataFrame:
        """Read a parquet file, returning empty DataFrame if missing."""
        if path.exists():
            return pd.read_parquet(path)
        logger.warning("Parquet file not found: %s", path)
        return pd.DataFrame()

    def _read_json(self, path: Path) -> Any:
        """Read a JSON file, returning empty structure if missing."""
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        logger.warning("JSON file not found: %s", path)
        return {}

    @property
    def version(self) -> str:
        """Current snapshot version string."""
        return self.version_manifest.get("current_version", "unknown")

    @property
    def models(self) -> List[str]:
        """List of unique model names in the leaderboard."""
        if self.leaderboard_long.empty:
            return []
        return sorted(self.leaderboard_long["model"].unique().tolist())

    @property
    def datasets(self) -> List[str]:
        """List of unique dataset names in the leaderboard."""
        if self.leaderboard_long.empty:
            return []
        return sorted(self.leaderboard_long["dataset"].unique().tolist())

    @property
    def metric_groups(self) -> List[str]:
        """List of unique metric groups."""
        return sorted(set(m["metric_group"] for m in self.metric_catalog))

    @property
    def condition_modalities(self) -> List[str]:
        """List of unique condition modalities in the data."""
        if self.leaderboard_long.empty:
            return []
        return sorted(self.leaderboard_long["condition_modality"].unique().tolist())

    @property
    def semantic_levels(self) -> List[str]:
        """List of unique semantic levels in the data."""
        if self.leaderboard_long.empty:
            return []
        return sorted(self.leaderboard_long["semantic_level"].unique().tolist())
