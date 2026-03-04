"""
Base Lightning DataModule for conditional time series generation.

This module defines the base class for all dataset implementations.
It provides standardized data loading, preprocessing, and batch creation.
"""

from __future__ import annotations

import json
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from contsg.config.schema import DataConfig


class TimeSeriesDataset(Dataset):
    """
    Basic dataset for time series with conditions.

    This dataset loads pre-processed time series data and conditions
    (text embeddings, attributes, or labels) from numpy files.

    Expected file structure:
        data_folder/
        ├── meta.json              # Dataset metadata
        ├── train_ts.npy          # Training time series (N, L, C)
        ├── train_caps.npy        # Training captions (N,) or embeddings (N, D)
        ├── train_attrs_idx.npy   # Training attributes (N, A) [optional]
        ├── valid_ts.npy
        ├── valid_caps.npy
        ├── valid_attrs_idx.npy
        ├── test_ts.npy
        ├── test_caps.npy
        └── test_attrs_idx.npy
    """

    def __init__(
        self,
        data_folder: Path,
        split: str = "train",
        normalize: bool = True,
        transform: Optional[Any] = None,
        provide_bridge_example: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            data_folder: Path to dataset folder
            split: Data split ("train", "valid", "test")
            normalize: Whether to normalize time series
            transform: Optional transform to apply
            provide_bridge_example: Whether to provide example TS for Bridge model
        """
        self.data_folder = Path(data_folder)
        self.split = split
        self.normalize = normalize
        self.transform = transform
        self.provide_bridge_example = provide_bridge_example

        # Load data
        self._load_data()

        # Load metadata
        self._load_metadata()

    def _load_data(self) -> None:
        """Load time series and conditions from files."""
        prefix = self.split

        # Load time series
        ts_path = self.data_folder / f"{prefix}_ts.npy"
        if not ts_path.exists():
            raise FileNotFoundError(f"Time series file not found: {ts_path}")
        self.ts = np.load(ts_path).astype(np.float32)

        # Load captions/embeddings (support multiple naming conventions)
        # Try: {split}_caps.npy, {split}_text_caps.npy
        caps_path = self.data_folder / f"{prefix}_caps.npy"
        if not caps_path.exists():
            caps_path = self.data_folder / f"{prefix}_text_caps.npy"
        if caps_path.exists():
            self.caps = np.load(caps_path, allow_pickle=True)
        else:
            self.caps = None

        # Load text embeddings if available (pre-computed)
        # Try multiple naming patterns:
        # 1. {split}_cap_emb.npy (simple format)
        # 2. {split}_text_caps_embeddings_*.npy (benchmark format)
        self.cap_emb = None
        cap_emb_path = self.data_folder / f"{prefix}_cap_emb.npy"
        if cap_emb_path.exists():
            self.cap_emb = np.load(cap_emb_path).astype(np.float32)
        else:
            # Search for benchmark-style embedding files
            import glob
            pattern = str(self.data_folder / f"{prefix}_text_caps_embeddings_*.npy")
            emb_files = sorted(glob.glob(pattern))
            if emb_files:
                # Prefer 1024-dim embeddings if available
                for f in emb_files:
                    if "1024" in f:
                        self.cap_emb = np.load(f).astype(np.float32)
                        break
                if self.cap_emb is None:
                    # Use the first available embedding file
                    self.cap_emb = np.load(emb_files[0]).astype(np.float32)

        # Load attributes (optional)
        attrs_path = self.data_folder / f"{prefix}_attrs_idx.npy"
        if attrs_path.exists():
            self.attrs = np.load(attrs_path)
        else:
            self.attrs = None

        # Load labels (optional)
        labels_path = self.data_folder / f"{prefix}_labels.npy"
        if labels_path.exists():
            self.labels = np.load(labels_path)
        else:
            self.labels = None

        # Normalize time series
        if self.normalize:
            self._normalize_ts()

    def _normalize_ts(self) -> None:
        """Normalize time series to zero mean and unit variance."""
        # Compute statistics from training data
        if self.split == "train":
            self.ts_mean = self.ts.mean(axis=(0, 1), keepdims=True)
            self.ts_std = self.ts.std(axis=(0, 1), keepdims=True)
            self.ts_std = np.where(self.ts_std == 0, 1.0, self.ts_std)
        else:
            # Load statistics from meta or training set
            stats_path = self.data_folder / "normalization_stats.npz"
            if stats_path.exists():
                stats = np.load(stats_path)
                self.ts_mean = stats["mean"]
                self.ts_std = stats["std"]
            else:
                # Fallback: compute from this split
                self.ts_mean = self.ts.mean(axis=(0, 1), keepdims=True)
                self.ts_std = self.ts.std(axis=(0, 1), keepdims=True)
                self.ts_std = np.where(self.ts_std == 0, 1.0, self.ts_std)

        self.ts = (self.ts - self.ts_mean) / self.ts_std

    def _load_metadata(self) -> None:
        """Load dataset metadata from meta.json."""
        meta_path = self.data_folder / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.meta = json.load(f)
        else:
            # Infer metadata from data
            self.meta = {
                "n_samples": len(self.ts),
                "seq_length": self.ts.shape[1],
                "n_var": self.ts.shape[2],
            }

    def __len__(self) -> int:
        return len(self.ts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - "ts": Time series tensor (L, C)
                - "tp": Time positions (L,) - indices [0, 1, ..., L-1]
                - "cap_emb": Caption embedding (D,) if available
                - "cap": Raw caption string if available
                - "attrs": Attribute indices if available
                - "label": Class label if available
                - "idx": Sample index
                - "bridge_example_ts": Example TS for Bridge model (L, C) if enabled
        """
        ts_tensor = torch.from_numpy(self.ts[idx])
        seq_len = ts_tensor.shape[0]

        item: Dict[str, Any] = {
            "ts": ts_tensor,
            "tp": torch.arange(seq_len, dtype=torch.float32),  # Time positions
            "idx": idx,
        }

        # Add caption embedding
        if self.cap_emb is not None:
            item["cap_emb"] = torch.from_numpy(self.cap_emb[idx])

        # Add raw caption
        if self.caps is not None:
            item["cap"] = str(self.caps[idx])

        # Add attributes
        if self.attrs is not None:
            item["attrs"] = torch.from_numpy(self.attrs[idx])

        # Add label
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx])

        # Add example time series for Bridge model
        if self.provide_bridge_example:
            # Use the same sample as the prototype source during training.
            item["bridge_example_ts"] = ts_tensor

        # Apply transform
        if self.transform:
            item = self.transform(item)

        return item


class BaseDataModule(pl.LightningDataModule):
    """
    Base Lightning DataModule for time series datasets.

    This class provides a standardized interface for loading and
    preparing time series data for training and evaluation.

    Subclasses can override `_create_dataset()` to customize
    dataset creation for specific data formats.

    Example:
        @Registry.register_dataset("synth-m")
        class SynthMDataModule(BaseDataModule):
            '''Synthetic multi-shape dataset'''
            pass  # Use default implementation

        @Registry.register_dataset("custom")
        class CustomDataModule(BaseDataModule):
            '''Custom dataset with special preprocessing'''

            def _create_dataset(self, split: str) -> Dataset:
                # Custom dataset creation logic
                return CustomDataset(...)
    """

    def __init__(
        self,
        config: DataConfig,
        train_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the data module.

        Args:
            config: Data configuration
            train_config: Optional training config (for batch_size, num_workers)
        """
        super().__init__()

        self.config = config
        self.train_config = train_config or {}

        # Datasets (populated in setup)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        # Save hyperparameters
        self.save_hyperparameters({"data_config": config.model_dump(mode="json")})

    def prepare_data(self) -> None:
        """
        Prepare data (download, preprocess, etc.).

        This method is called only once on the main process.
        Override this for custom data preparation logic.
        """
        # Verify data folder exists
        if not self.config.data_folder.exists():
            raise FileNotFoundError(
                f"Data folder not found: {self.config.data_folder}. "
                "Please download the dataset first."
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets for the specified stage.

        Args:
            stage: "fit", "validate", "test", or "predict"
        """
        if stage == "fit" or stage is None:
            self.train_dataset = self._create_dataset("train")
            self.val_dataset = self._create_dataset("valid")

        if stage == "test" or stage is None:
            self.test_dataset = self._create_dataset("test")

        if stage == "predict":
            self.test_dataset = self._create_dataset("test")

    def _create_dataset(self, split: str) -> Dataset:
        """
        Create a dataset for the given split.

        Override this method in subclasses for custom dataset creation.

        Args:
            split: Data split ("train", "valid", "test")

        Returns:
            Dataset instance
        """
        # Check if Bridge model requires example time series
        model_name = self.train_config.get("model_name", "")
        provide_bridge_example = model_name.lower() == "bridge"

        return TimeSeriesDataset(
            data_folder=self.config.data_folder,
            split=split,
            normalize=self.config.normalize,
            provide_bridge_example=provide_bridge_example,
        )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        batch_size = self.config.batch_size or self.train_config.get("batch_size", 256)
        num_workers = self.train_config.get("num_workers", 4)
        pin_memory = self.train_config.get("pin_memory", True)

        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        batch_size = self.config.batch_size or self.train_config.get("batch_size", 256)
        num_workers = self.train_config.get("num_workers", 4)
        pin_memory = self.train_config.get("pin_memory", True)

        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        batch_size = self.config.batch_size or self.train_config.get("batch_size", 256)
        num_workers = self.train_config.get("num_workers", 4)
        pin_memory = self.train_config.get("pin_memory", True)

        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader."""
        return self.test_dataloader()

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @property
    def n_var(self) -> int:
        """Number of variables/channels in the time series."""
        return self.config.n_var

    @property
    def seq_length(self) -> int:
        """Sequence length of the time series."""
        return self.config.seq_length

    def _get_ts_array(self, dataset: Optional[Dataset], name: str) -> np.ndarray:
        """Extract time series array from dataset."""
        if dataset is None:
            raise RuntimeError(
                f"{name} requires setup() to be called first. "
                "Ensure DataModule.setup(stage='fit') has been executed."
            )
        if hasattr(dataset, "ts"):
            ts_array = dataset.ts
        else:
            # Fallback: iterate through dataset (less efficient)
            ts_list = [dataset[i]["ts"].numpy() for i in range(len(dataset))]
            ts_array = np.stack(ts_list, axis=0)
        if ts_array.ndim != 3:
            raise RuntimeError(f"Expected 3D array (N, L, C), got shape {ts_array.shape}")
        return ts_array.astype(np.float32)

    def _get_captions(self, dataset: Optional[Dataset], name: str) -> List[str]:
        """Extract caption strings from dataset."""
        if dataset is None:
            raise RuntimeError(
                f"{name} requires setup() to be called first. "
                "Ensure DataModule.setup(stage='fit') has been executed."
            )
        if hasattr(dataset, "caps") and dataset.caps is not None:
            return [str(c) for c in dataset.caps]
        raise RuntimeError(f"{name} requires dataset to have 'caps' attribute.")

    @property
    def train_ts_array(self) -> np.ndarray:
        """Raw training time series as numpy array (N, L, C)."""
        return self._get_ts_array(self.train_dataset, "train_ts_array")

    @property
    def train_captions(self) -> List[str]:
        """Raw training captions as list of strings."""
        return self._get_captions(self.train_dataset, "train_captions")

    @property
    def val_ts_array(self) -> np.ndarray:
        """Raw validation time series as numpy array (N, L, C)."""
        return self._get_ts_array(self.val_dataset, "val_ts_array")

    @property
    def val_captions(self) -> List[str]:
        """Raw validation captions as list of strings."""
        return self._get_captions(self.val_dataset, "val_captions")

    def get_sample_batch(self, n: int = 4) -> Dict[str, Tensor]:
        """
        Get a sample batch for debugging/visualization.

        Args:
            n: Number of samples to retrieve

        Returns:
            Batch dictionary
        """
        if self.train_dataset is None:
            self.setup("fit")

        samples = [self.train_dataset[i] for i in range(min(n, len(self.train_dataset)))]

        # Collate samples
        batch = {}
        for key in samples[0].keys():
            if isinstance(samples[0][key], Tensor):
                batch[key] = torch.stack([s[key] for s in samples])
            else:
                batch[key] = [s[key] for s in samples]

        return batch
