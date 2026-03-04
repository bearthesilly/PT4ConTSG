"""
Specialized evaluation metrics.

This module implements complex, specialized metrics:
- CTTPMetric: Text-series alignment score
- RealVsFakeDiscriminatorAUCMetric: K-fold CNN discriminator
- TSNETimeSeriesVisualizationMetric: t-SNE visualization
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from contsg.eval.metrics.base import AccumulativeMetric, CollectiveMetric
from contsg.registry import Registry

logger = logging.getLogger(__name__)


@Registry.register_metric("cttp")
class CTTPMetric(AccumulativeMetric):
    """
    CTTP (Contrastive Text-Time Series Pretraining) metric.

    Computes trace(ts_gen_emb @ cap_emb.T) to measure text-series alignment.
    Higher values indicate better alignment between generated sequences
    and their conditioning text.

    Legacy source: metrics.py:107-119
    """

    def __init__(self, name: str = "cttp"):
        """
        Initialize CTTP metric.

        Args:
            name: Metric identifier
        """
        super().__init__(name)

    @property
    def requires_clip(self) -> bool:
        """CTTP requires CLIP embeddings."""
        return True

    def update(self, batch_data: Dict[str, Any]) -> None:
        """
        Accumulate CTTP scores from batch.

        Args:
            batch_data: Must contain 'ts_gen_emb' and 'cap_emb' (B, D)
        """
        ts_gen_emb: Tensor = batch_data["ts_gen_emb"]
        cap_emb: Tensor = batch_data["cap_emb"]

        # Compute trace of similarity matrix
        cttp_batch = torch.mm(ts_gen_emb, cap_emb.permute(1, 0)).trace().item()
        batch_size = ts_gen_emb.shape[0]

        self.total_value += cttp_batch
        self.total_samples += batch_size


@Registry.register_metric("disc_auc")
class RealVsFakeDiscriminatorAUCMetric(CollectiveMetric):
    """
    Real vs Fake Discriminator AUC metric.

    Trains a lightweight 1D CNN classifier to distinguish real from generated
    time series, using K-fold cross-validation. Reports mean AUC.

    Features:
    - GPU + AMP acceleration
    - Masked pooling for variable-length sequences
    - Results saved to disk

    Legacy source: metrics.py:540-806
    """

    def __init__(
        self,
        name: str = "disc_auc",
        save_dir: str = "./disc_auc_outputs",
        k_folds: int = 5,
        epochs: int = 5,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        device: str = "cuda",
        seed: int = 42,
        num_workers: int = 0,
    ):
        """
        Initialize Discriminator AUC metric.

        Args:
            name: Metric identifier
            save_dir: Directory for saving results
            k_folds: Number of cross-validation folds
            epochs: Training epochs per fold
            batch_size: Training batch size
            lr: Learning rate
            weight_decay: L2 regularization weight
            hidden_channels: CNN hidden channels
            num_layers: Number of conv layers
            dropout: Dropout rate
            device: Torch device
            seed: Random seed
            num_workers: DataLoader workers
        """
        super().__init__(name)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.k_folds = int(k_folds)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.seed = int(seed)
        self.num_workers = int(num_workers)

        try:
            self.device = torch.device(device)
        except Exception:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Runtime caches
        self._pos_list: List[Tensor] = []  # real ts
        self._neg_list: List[Tensor] = []  # generated pred
        self._len_pos: List[Tensor] = []
        self._len_neg: List[Tensor] = []

    def reset(self) -> None:
        """Reset accumulators."""
        self._pos_list = []
        self._neg_list = []
        self._len_pos = []
        self._len_neg = []

    def update(self, batch_data: Dict[str, Any]) -> None:
        """
        Collect real and generated sequences.

        Args:
            batch_data: Must contain 'ts', 'pred', 'ts_len'
        """
        ts: Tensor = batch_data["ts"].detach().cpu()
        pred: Tensor = batch_data["pred"].detach().cpu()
        ts_len: Tensor = batch_data["ts_len"].detach().cpu()

        self._pos_list.append(ts)
        self._neg_list.append(pred)
        self._len_pos.append(ts_len)
        self._len_neg.append(ts_len)

    def compute(self) -> float:
        """
        Train K-fold discriminator and compute mean AUC.

        Returns:
            Mean AUC across folds, or NaN if no data
        """
        if len(self._pos_list) == 0:
            return float("nan")

        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score

        # Assemble dataset
        X_pos = torch.cat(self._pos_list, dim=0)  # (N, L, F)
        X_neg = torch.cat(self._neg_list, dim=0)
        len_pos = torch.cat(self._len_pos, dim=0)
        len_neg = torch.cat(self._len_neg, dim=0)

        assert X_pos.shape[1:] == X_neg.shape[1:], "pos/neg shapes mismatch"

        X = torch.cat([X_pos, X_neg], dim=0)
        lengths = torch.cat([len_pos, len_neg], dim=0)
        y = torch.cat([
            torch.ones(len_pos.shape[0], dtype=torch.long),
            torch.zeros(len_neg.shape[0], dtype=torch.long),
        ], dim=0)

        # K-fold setup
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
        y_np = y.numpy()

        fold_aucs: List[float] = []

        # Discriminator model
        class TinyTSDiscriminator(torch.nn.Module):
            def __init__(self, in_ch: int, hidden: int, layers: int, dropout: float):
                super().__init__()
                chs = [in_ch] + [hidden] * layers
                modules: List[torch.nn.Module] = []
                for i in range(layers):
                    modules += [
                        torch.nn.Conv1d(chs[i], chs[i + 1], kernel_size=5, padding=2, bias=False),
                        torch.nn.BatchNorm1d(chs[i + 1]),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Dropout(dropout),
                    ]
                self.backbone = torch.nn.Sequential(*modules)
                self.head = torch.nn.Linear(hidden, 1)

            def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
                x = x.transpose(1, 2)  # (B, F, L)
                h = self.backbone(x)  # (B, C, L)
                B, C, L_out = h.shape
                device = h.device

                t = torch.arange(L_out, device=device)[None, :]
                mask = (t < lengths[:, None]).float()[:, None, :]
                denom = torch.clamp(mask.sum(dim=2), min=1.0)
                h = (h * mask).sum(dim=2) / denom
                return self.head(h).squeeze(-1)

        class TSDataset(torch.utils.data.Dataset):
            def __init__(self, X: Tensor, lengths: Tensor, y: Tensor):
                self.X = X
                self.lengths = lengths
                self.y = y

            def __len__(self) -> int:
                return self.X.shape[0]

            def __getitem__(self, idx: int):
                return self.X[idx], self.lengths[idx], self.y[idx]

        # Determinism
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

        L, F = X.shape[1], X.shape[2]

        logger.info(
            f"[disc_auc] Start K-fold training | device={self.device} cuda={use_cuda} | "
            f"samples={X.shape[0]} seq_len={L} n_vars={F} | k_folds={self.k_folds} "
            f"epochs={self.epochs}"
        )

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(y_np), y_np)):
            X_tr, len_tr, y_tr = X[tr_idx], lengths[tr_idx], y[tr_idx]
            X_va, len_va, y_va = X[va_idx], lengths[va_idx], y[va_idx]

            ds_tr = TSDataset(X_tr, len_tr, y_tr)
            ds_va = TSDataset(X_va, len_va, y_va)

            loader_tr = torch.utils.data.DataLoader(
                ds_tr, batch_size=self.batch_size, shuffle=True,
                num_workers=self.num_workers, pin_memory=use_cuda
            )
            loader_va = torch.utils.data.DataLoader(
                ds_va, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=use_cuda
            )

            model = TinyTSDiscriminator(
                in_ch=F,
                hidden=self.hidden_channels,
                layers=self.num_layers,
                dropout=self.dropout,
            ).to(self.device)

            optim = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            criterion = torch.nn.BCEWithLogitsLoss()

            for epoch in range(self.epochs):
                model.train()
                running_loss = 0.0
                running_cnt = 0

                for xb, lb, yb in loader_tr:
                    xb = xb.to(self.device)
                    lb = lb.to(self.device)
                    yb = yb.float().to(self.device)

                    optim.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda', enabled=use_cuda):
                        logits = model(xb, lb)
                        loss = criterion(logits, yb)

                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()

                    running_loss += loss.detach().float().item() * xb.size(0)
                    running_cnt += xb.size(0)

            # Validation AUC
            model.eval()
            all_logits = []
            all_ys = []

            with torch.no_grad():
                for xb, lb, yb in loader_va:
                    xb = xb.to(self.device)
                    lb = lb.to(self.device)

                    with torch.amp.autocast('cuda', enabled=use_cuda):
                        logits = model(xb, lb)

                    all_logits.append(logits.detach().cpu())
                    all_ys.append(yb)

            logits_np = torch.cat(all_logits, dim=0).numpy()
            ys_np = torch.cat(all_ys, dim=0).numpy()

            try:
                probs = 1.0 / (1.0 + np.exp(-logits_np))
                auc = float(roc_auc_score(ys_np, probs))
            except Exception as e:
                logger.warning(f"disc_auc fold {fold_idx} failed to compute AUC: {e}")
                auc = float("nan")

            fold_aucs.append(auc)
            logger.info(f"[disc_auc] Fold {fold_idx + 1}/{self.k_folds} AUC={auc:.6f}")

        mean_auc = float(np.nanmean(fold_aucs)) if fold_aucs else float("nan")

        # Save results
        self._save_results(fold_aucs, mean_auc)

        return mean_auc

    def _save_results(self, fold_aucs: List[float], mean_auc: float) -> None:
        """Save results to disk."""
        try:
            with open(self.save_dir / "fold_auc.csv", "w") as f:
                f.write("fold,auc\n")
                for i, v in enumerate(fold_aucs):
                    f.write(f"{i},{v}\n")

            results = {
                "mean_auc": mean_auc,
                "std_auc": float(np.nanstd(fold_aucs)) if len(fold_aucs) > 1 else 0.0,
                "fold_aucs": fold_aucs,
                "k_folds": self.k_folds,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "seed": self.seed,
            }
            with open(self.save_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save disc_auc results: {e}")


@Registry.register_metric("tsne_viz")
class TSNETimeSeriesVisualizationMetric(CollectiveMetric):
    """
    t-SNE visualization metric for generated time series.

    Creates t-SNE visualization comparing generated vs ground truth
    time series distributions. Returns 0.0 as metric (primarily for visualization).

    Legacy source: metrics.py:256-333
    """

    def __init__(self, name: str = "tsne_viz", save_dir: str = "./visualization_outputs"):
        """
        Initialize t-SNE visualization metric.

        Args:
            name: Metric identifier
            save_dir: Directory for saving visualizations
        """
        super().__init__(name)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def update(self, batch_data: Dict[str, Any]) -> None:
        """
        Collect predictions and ground truth for visualization.

        Args:
            batch_data: Must contain 'pred' and 'ts' (B, L, F)
        """
        pred: Tensor = batch_data["pred"]
        ts: Tensor = batch_data["ts"]

        # Flatten for t-SNE input
        pred_flat = pred.cpu().numpy().reshape(pred.shape[0], -1)
        ts_flat = ts.cpu().numpy().reshape(ts.shape[0], -1)

        self.collected_data.append({
            "pred_flat": pred_flat,
            "ts_flat": ts_flat,
            "labels": ["Generated"] * len(pred_flat) + ["Ground Truth"] * len(ts_flat),
        })

    def compute(self) -> float:
        """
        Create t-SNE visualization.

        Returns:
            0.0 (visualization metric, no numeric value)
        """
        if not self.collected_data:
            return float("nan")

        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        # Set matplotlib backend for non-interactive use
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Combine all data
        all_pred = np.concatenate([batch["pred_flat"] for batch in self.collected_data], axis=0)
        all_ts = np.concatenate([batch["ts_flat"] for batch in self.collected_data], axis=0)
        all_labels = []
        for batch in self.collected_data:
            all_labels.extend(batch["labels"])

        combined_data = np.vstack([all_pred, all_ts])

        # Standardize
        scaler = StandardScaler()
        combined_data_scaled = scaler.fit_transform(combined_data)

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        tsne_result = tsne.fit_transform(combined_data_scaled)

        # Create visualization
        self._create_tsne_plot(tsne_result, all_labels)

        return 0.0

    def _create_tsne_plot(self, tsne_result: np.ndarray, labels: List[str]) -> None:
        """Create and save t-SNE plot."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        gen_indices = [i for i, label in enumerate(labels) if label == "Generated"]
        gt_indices = [i for i, label in enumerate(labels) if label == "Ground Truth"]

        plt.scatter(
            tsne_result[gen_indices, 0],
            tsne_result[gen_indices, 1],
            c="red", marker="o", alpha=0.6, s=50, label="Generated"
        )
        plt.scatter(
            tsne_result[gt_indices, 0],
            tsne_result[gt_indices, 1],
            c="blue", marker="^", alpha=0.6, s=50, label="Ground Truth"
        )

        plt.title("t-SNE Visualization of Generated vs Ground Truth Time Series", fontsize=16)
        plt.xlabel("t-SNE Component 1", fontsize=12)
        plt.ylabel("t-SNE Component 2", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = self.save_dir / "tsne_timeseries_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"t-SNE visualization saved to: {save_path}")


__all__ = [
    "CTTPMetric",
    "RealVsFakeDiscriminatorAUCMetric",
    "TSNETimeSeriesVisualizationMetric",
]
