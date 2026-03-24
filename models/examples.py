"""
Example model implementations for ConTSG.

These are placeholder/baseline implementations that demonstrate the registration
pattern and interface.

Note: Full model implementations are in separate files:
- verbalts.py - VerbalTS with multi-view noise estimation
- timeweaver.py - TimeWeaver with heterogeneous attribute conditioning
- bridge.py - Bridge with domain-unified prototyper
- t2s.py - T2S with flow matching
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from contsg.models.base import BaseGeneratorModule
from contsg.registry import Registry


@Registry.register_model("retrieval", aliases=["ret", "nn"])
class RetrievalModule(BaseGeneratorModule):
    """Retrieval baseline: Nearest neighbor search in training set.

    This is a simple baseline that retrieves the most similar time series
    from the training set based on condition similarity.
    """

    def _build_model(self) -> None:
        # No learnable parameters
        self.dummy = nn.Parameter(torch.zeros(1))  # Placeholder
        self._train_ts: Optional[Tensor] = None
        self._train_cap_emb: Optional[Tensor] = None
        self._train_cap_emb_norm: Optional[Tensor] = None
        self._cache_device: Optional[torch.device] = None

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Retrieval has no training loss
        loss = self.dummy.sum() * 0.0
        return {"loss": loss}

    def _load_train_cap_emb(self, data_folder: Path) -> Tensor:
        cap_emb_path = data_folder / "train_cap_emb.npy"
        import numpy as np
        import glob

        if cap_emb_path.exists():
            return torch.from_numpy(np.load(cap_emb_path)).float()

        pattern = str(data_folder / "train_text_caps_embeddings_*.npy")
        emb_files = sorted(glob.glob(pattern))
        if not emb_files:
            raise FileNotFoundError(
                f"No train text embeddings found in {data_folder}"
            )
        preferred = None
        for f in emb_files:
            if "1024" in f:
                preferred = f
                break
        emb_path = preferred or emb_files[0]
        return torch.from_numpy(np.load(emb_path)).float()

    def _ensure_retrieval_cache(self, device: torch.device) -> None:
        if self._train_ts is not None and self._cache_device == device:
            return

        data_folder = Path(self.config.data.data_folder)
        train_ts_path = data_folder / "train_ts.npy"
        if not train_ts_path.exists():
            raise FileNotFoundError(f"Missing train_ts.npy at {train_ts_path}")

        import numpy as np

        train_ts_np = np.load(train_ts_path).astype(np.float32)
        if self.config.data.normalize:
            stats_path = data_folder / "normalization_stats.npz"
            mean = None
            std = None
            if stats_path.exists():
                stats = np.load(stats_path)
                mean = stats.get("mean")
                std = stats.get("std")
            if mean is None or std is None:
                mean = train_ts_np.mean(axis=(0, 1), keepdims=True)
                std = train_ts_np.std(axis=(0, 1), keepdims=True)
            if mean.ndim == 1:
                mean = mean.reshape(1, 1, -1)
            if std.ndim == 1:
                std = std.reshape(1, 1, -1)
            std = np.where(std == 0, 1.0, std)
            train_ts_np = (train_ts_np - mean) / std

        train_ts = torch.from_numpy(train_ts_np).float()
        train_cap_emb = self._load_train_cap_emb(data_folder)
        if train_ts.shape[0] != train_cap_emb.shape[0]:
            raise ValueError(
                f"Train TS count {train_ts.shape[0]} != train embeddings {train_cap_emb.shape[0]}"
            )

        self._train_ts = train_ts.to(device)
        self._train_cap_emb = train_cap_emb.to(device)

        if self.config.model.similarity_metric == "cosine":
            self._train_cap_emb_norm = F.normalize(self._train_cap_emb, dim=1)
        else:
            self._train_cap_emb_norm = None

        self._cache_device = device

    def _retrieve_indices(self, condition: Tensor) -> Tensor:
        if self._train_cap_emb is None:
            raise RuntimeError("Retrieval cache not initialized.")

        metric = self.config.model.similarity_metric
        top_k = int(self.config.model.top_k)
        top_k = min(top_k, self._train_cap_emb.shape[0])

        if metric == "cosine":
            cond_norm = F.normalize(condition, dim=1)
            sims = torch.matmul(cond_norm, self._train_cap_emb_norm.t())
            _, topk = torch.topk(sims, k=top_k, dim=1, largest=True)
        elif metric == "euclidean":
            cond_sq = (condition ** 2).sum(dim=1, keepdim=True)
            train_sq = (self._train_cap_emb ** 2).sum(dim=1, keepdim=True).t()
            dists = cond_sq + train_sq - 2.0 * torch.matmul(condition, self._train_cap_emb.t())
            _, topk = torch.topk(dists, k=top_k, dim=1, largest=False)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

        return topk

    def _expand_topk(self, topk: Tensor, n_samples: int) -> Tensor:
        # Deterministic rank-based cycling/truncation
        if n_samples <= topk.shape[1]:
            return topk[:, :n_samples]
        reps = (n_samples + topk.shape[1] - 1) // topk.shape[1]
        expanded = topk.repeat(1, reps)
        return expanded[:, :n_samples]

    def generate(
        self,
        condition: Tensor,
        n_samples: int = 1,
        **kwargs: Any,
    ) -> Tensor:
        if condition.dim() != 2:
            raise ValueError(
                f"Expected condition to be 2D (B, D), got {condition.shape}"
            )

        device = condition.device
        self._ensure_retrieval_cache(device)

        topk = self._retrieve_indices(condition)
        selected = self._expand_topk(topk, n_samples)

        retrieved = self._train_ts[selected]  # (B, n_samples, L, F)
        return retrieved.permute(1, 0, 2, 3)  # (n_samples, B, L, F)
