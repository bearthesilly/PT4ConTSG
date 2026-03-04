"""
Global application state for the ConTSG-Bench HF Space.

Manages the loaded snapshot data and provides a single point of access
for all tabs and components.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.data_loader import SnapshotData

logger = logging.getLogger(__name__)

# Global state — initialized by app.py
_data: Optional[SnapshotData] = None


def init(snapshot_dir: str | Path) -> SnapshotData:
    """Initialize the global state with snapshot data."""
    global _data
    _data = SnapshotData(snapshot_dir)
    return _data


def get_data() -> SnapshotData:
    """Get the loaded snapshot data.

    Raises:
        RuntimeError: If data has not been initialized.
    """
    if _data is None:
        raise RuntimeError(
            "Snapshot data not initialized. Call state.init(snapshot_dir) first."
        )
    return _data
