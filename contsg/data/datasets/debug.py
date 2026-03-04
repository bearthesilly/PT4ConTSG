"""
Lightweight synthetic dataset for CPU smoke tests.

This dataset generates small random time series and condition fields in memory,
so the training/evaluation pipeline can run without external data files.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from contsg.data.datamodule import BaseDataModule
from contsg.registry import Registry


class DebugTimeSeriesDataset(Dataset):
    """In-memory synthetic dataset for quick smoke testing."""

    def __init__(
        self,
        split: str,
        n_samples: int,
        seq_length: int,
        n_var: int,
        condition_cfg: Optional[Any] = None,
        seed: int = 0,
        provide_bridge_example: bool = False,
    ) -> None:
        self.split = split
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.n_var = n_var
        self.provide_bridge_example = provide_bridge_example

        rng = np.random.default_rng(seed)

        self.ts = rng.standard_normal((n_samples, seq_length, n_var)).astype(np.float32)
        self.tp = torch.arange(seq_length, dtype=torch.float32)

        # Optional condition fields
        self.cap_emb: Optional[np.ndarray] = None
        self.caps: Optional[np.ndarray] = None
        self.attrs_discrete: Optional[np.ndarray] = None
        self.attrs_continuous: Optional[np.ndarray] = None
        self.attrs: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None

        if condition_cfg is not None:
            text_cfg = getattr(condition_cfg, "text", None)
            if text_cfg is not None and getattr(text_cfg, "enabled", True):
                text_dim = int(getattr(text_cfg, "input_dim", 1024))
                self.cap_emb = rng.standard_normal((n_samples, text_dim)).astype(np.float32)
                self.caps = np.array(
                    [f"debug-{split}-{i}" for i in range(n_samples)], dtype=object
                )

            attr_cfg = getattr(condition_cfg, "attribute", None)
            if attr_cfg is not None and getattr(attr_cfg, "enabled", False):
                cont_dim = int(getattr(attr_cfg, "continuous_dim", 0))
                disc_cfgs = list(getattr(attr_cfg, "discrete_configs", []) or [])

                if cont_dim > 0:
                    self.attrs_continuous = rng.standard_normal(
                        (n_samples, cont_dim)
                    ).astype(np.float32)

                if disc_cfgs:
                    discrete_cols = []
                    for cfg in disc_cfgs:
                        num_classes = int(cfg.get("num_classes", 1))
                        discrete_cols.append(
                            rng.integers(0, num_classes, size=(n_samples,), endpoint=False)
                        )
                    self.attrs_discrete = np.stack(discrete_cols, axis=1).astype(np.int64)
                    self.attrs = self.attrs_discrete

            label_cfg = getattr(condition_cfg, "label", None)
            if label_cfg is not None and getattr(label_cfg, "enabled", False):
                num_classes = int(getattr(label_cfg, "num_classes", 2))
                self.labels = rng.integers(
                    0, num_classes, size=(n_samples,), endpoint=False, dtype=np.int64
                )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ts_tensor = torch.from_numpy(self.ts[idx])

        item: Dict[str, Any] = {
            "ts": ts_tensor,
            "tp": self.tp,
            "idx": idx,
        }

        if self.cap_emb is not None:
            item["cap_emb"] = torch.from_numpy(self.cap_emb[idx])
        if self.caps is not None:
            item["cap"] = str(self.caps[idx])

        if self.attrs_discrete is not None:
            item["attrs_discrete"] = torch.from_numpy(self.attrs_discrete[idx])
        if self.attrs_continuous is not None:
            item["attrs_continuous"] = torch.from_numpy(self.attrs_continuous[idx])
        if self.attrs is not None:
            item["attrs"] = torch.from_numpy(self.attrs[idx])

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.provide_bridge_example:
            example_idx = (idx + 1) % len(self.ts)
            item["bridge_example_ts"] = torch.from_numpy(self.ts[example_idx])

        return item


@Registry.register_dataset("debug")
class DebugDataModule(BaseDataModule):
    """DataModule that generates small synthetic data for smoke tests."""

    def prepare_data(self) -> None:
        # No files needed for debug dataset.
        return

    def _create_dataset(self, split: str) -> Dataset:
        debug_samples = self.train_config.get("debug_samples", {})
        if isinstance(debug_samples, dict):
            n_samples = int(
                debug_samples.get(
                    split, {"train": 32, "valid": 8, "test": 8}.get(split, 8)
                )
            )
        elif isinstance(debug_samples, int):
            n_samples = debug_samples
        else:
            n_samples = {"train": 32, "valid": 8, "test": 8}.get(split, 8)

        seed = int(self.train_config.get("seed", 0))
        seed_offset = {"train": 0, "valid": 1, "test": 2}.get(split, 0)
        condition_cfg = self.train_config.get("condition")

        model_name = self.train_config.get("model_name", "")
        provide_bridge_example = model_name.lower() == "bridge"

        return DebugTimeSeriesDataset(
            split=split,
            n_samples=n_samples,
            seq_length=self.config.seq_length,
            n_var=self.config.n_var,
            condition_cfg=condition_cfg,
            seed=seed + seed_offset,
            provide_bridge_example=provide_bridge_example,
        )
