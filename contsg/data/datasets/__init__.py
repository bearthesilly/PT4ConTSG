"""Datasets package for ConTSG."""

# Import all dataset modules to trigger registration
from contsg.data.datasets import standard  # Standard datasets (synth, weather, etc.)
from contsg.data.datasets import debug  # In-memory debug dataset for smoke tests
from contsg.data.datasets import telecomts_segment  # Segment-level caption dataset
