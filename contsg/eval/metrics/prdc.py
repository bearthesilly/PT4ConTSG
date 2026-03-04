"""
PRDC-F1 based evaluation metrics.

This module implements Precision-Recall-based metrics using k-NN manifold estimation:
- PRDCF1Metric: TS-only PRDC-F1
- JointPRDCF1Metric: Joint TS+text PRDC-F1 with normalization

Backend selection: FAISS GPU → Torch CUDA → sklearn (auto)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from contsg.eval.metrics.base import CollectiveMetric
from contsg.registry import Registry

logger = logging.getLogger(__name__)

# Optional FAISS backend for accelerated kNN
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


@Registry.register_metric("prdc_f1")
class PRDCF1Metric(CollectiveMetric):
    """
    PRDC-F1 metric (TS-only).

    Computes Precision/Recall/F1 based on k-th nearest neighbor radii
    and cross-set 1-NN distances between generated and real distributions.

    Precision: Fraction of generated points within real manifold
    Recall: Fraction of real points within generated manifold
    F1: Harmonic mean of precision and recall

    Backend selection: auto | faiss | torch | sklearn
    - auto: FAISS GPU → Torch CUDA → sklearn

    Legacy source: metrics.py:1276-1472
    """

    def __init__(
        self,
        name: str = "prdc_f1",
        k: int = 5,
        max_samples: Optional[int] = None,
        seed: int = 0,
        metric: str = "euclidean",
        backend: str = "auto",
        knn_chunk_size: int = 4096,
    ):
        """
        Initialize PRDC-F1 metric.

        Args:
            name: Metric identifier
            k: Number of nearest neighbors for radius estimation
            max_samples: Maximum samples to use (None for all)
            seed: Random seed for sampling
            metric: Distance metric (euclidean)
            backend: Backend selection (auto, faiss, torch, sklearn)
            knn_chunk_size: Chunk size for batched kNN computation
        """
        super().__init__(name)
        self.k = int(k)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.seed = int(seed)
        self.metric = str(metric)
        self.backend = str(backend)
        self.knn_chunk_size = int(knn_chunk_size)

        self._gen_list: List[np.ndarray] = []
        self._real_list: List[np.ndarray] = []

    @property
    def requires_clip(self) -> bool:
        """PRDC-F1 requires CLIP embeddings."""
        return True

    def reset(self) -> None:
        """Reset accumulators."""
        super().reset()
        self._gen_list = []
        self._real_list = []

    def update(self, batch_data: Dict[str, Any]) -> None:
        """
        Collect generated and ground truth embeddings.

        Args:
            batch_data: Must contain 'ts_gen_emb' and 'ts_gt_emb' (B, D)
        """
        ts_gen_emb: Tensor = batch_data["ts_gen_emb"]
        ts_gt_emb: Tensor = batch_data["ts_gt_emb"]

        self._gen_list.append(ts_gen_emb.detach().cpu().numpy().astype(np.float64))
        self._real_list.append(ts_gt_emb.detach().cpu().numpy().astype(np.float64))

    def _select_backend(self) -> str:
        """Select best available backend."""
        if self.backend in ("faiss", "torch", "sklearn"):
            return self.backend

        if self.metric == "euclidean" and _HAS_FAISS:
            try:
                if faiss.get_num_gpus() > 0:
                    return "faiss"
            except Exception:
                pass

        if self.metric == "euclidean" and torch.cuda.is_available():
            return "torch"

        return "sklearn"

    @staticmethod
    def _filter_finite_rows(x: np.ndarray) -> np.ndarray:
        """Filter rows with any non-finite values."""
        mask = np.isfinite(x).all(axis=1)
        return x[mask]

    @staticmethod
    def _sample_indices(n: int, k: int, seed: int) -> np.ndarray:
        """Sample k indices from n without replacement."""
        if k >= n:
            return np.arange(n, dtype=np.int64)
        rng = np.random.RandomState(seed)
        return rng.choice(n, size=k, replace=False)

    def _compute_with_sklearn(
        self, Xq: np.ndarray, Xdb: np.ndarray, n_neighbors: int
    ) -> np.ndarray:
        """Compute squared distances to k-NN using sklearn."""
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors, algorithm="auto", n_jobs=-1, metric=self.metric
        )
        nbrs.fit(Xdb)
        dists, _ = nbrs.kneighbors(Xq)
        return dists.astype(np.float64) ** 2

    def _compute_self_kth_radius(
        self, X: np.ndarray, k_eff: int, backend: str
    ) -> np.ndarray:
        """
        Compute k-th nearest neighbor radius for each point (self-set).

        Args:
            X: (N, D) feature matrix
            k_eff: Effective k for kNN
            backend: Backend to use

        Returns:
            (N,) k-th NN squared distances
        """
        n = X.shape[0]
        if k_eff < 1 or n < 2:
            return np.full(n, np.nan, dtype=np.float64)

        if backend == "faiss" and self.metric == "euclidean" and _HAS_FAISS:
            index = faiss.IndexFlatL2(X.shape[1])
            try:
                if faiss.get_num_gpus() > 0:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                pass
            X32 = X.astype(np.float32)
            index.add(X32)
            dists, _ = index.search(X32, k_eff + 1)  # includes self at [:,0]
            kth = dists[:, -1].astype(np.float64)
            return kth

        elif backend == "torch" and self.metric == "euclidean" and torch.cuda.is_available():
            device = torch.device("cuda")
            X_t = torch.from_numpy(X.astype(np.float32)).to(device)
            kth_chunks: List[Tensor] = []
            csz = max(1, min(self.knn_chunk_size, n))

            for i in range(0, n, csz):
                j = min(i + csz, n)
                Q = X_t[i:j]
                d = torch.cdist(Q, X_t, p=2.0).pow(2)  # (C, N)

                # Mask diagonal segment (self-distances)
                rows = torch.arange(j - i, device=device)
                cols = torch.arange(i, j, device=device)
                d[rows, cols] = float("inf")

                vals, _ = torch.topk(d, k_eff, dim=1, largest=False)
                kth = vals[:, -1]
                kth_chunks.append(kth.detach().cpu())

            return torch.cat(kth_chunks, dim=0).numpy().astype(np.float64)

        else:
            d2 = self._compute_with_sklearn(X, X, k_eff + 1)
            return d2[:, -1]

    def _compute_cross_1nn(
        self,
        Xq: np.ndarray,
        Xdb: np.ndarray,
        backend: str,
        return_indices: bool = False,
    ) -> Tuple[np.ndarray, ...]:
        """
        Compute cross-set 1-NN distances.

        Args:
            Xq: Query points
            Xdb: Database points
            backend: Backend to use
            return_indices: Whether to return NN indices

        Returns:
            (distances,) or (distances, indices)
        """
        if Xq.shape[0] == 0:
            if return_indices:
                return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int64)
            return (np.empty((0,), dtype=np.float64),)

        if backend == "faiss" and self.metric == "euclidean" and _HAS_FAISS:
            index = faiss.IndexFlatL2(Xdb.shape[1])
            try:
                if faiss.get_num_gpus() > 0:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                pass
            Xdb32 = Xdb.astype(np.float32)
            Xq32 = Xq.astype(np.float32)
            index.add(Xdb32)
            dists, inds = index.search(Xq32, 1)
            d1 = dists[:, 0].astype(np.float64)
            if return_indices:
                return d1, inds[:, 0].astype(np.int64)
            return (d1,)

        elif backend == "torch" and self.metric == "euclidean" and torch.cuda.is_available():
            device = torch.device("cuda")
            Q = torch.from_numpy(Xq.astype(np.float32)).to(device)
            DB = torch.from_numpy(Xdb.astype(np.float32)).to(device)
            d1_list: List[Tensor] = []
            idx_list: List[Tensor] = []
            csz = max(1, min(self.knn_chunk_size, Q.shape[0]))

            for i in range(0, Q.shape[0], csz):
                j = min(i + csz, Q.shape[0])
                qq = Q[i:j]
                d = torch.cdist(qq, DB, p=2.0).pow(2)
                minv, argmin = torch.min(d, dim=1)
                d1_list.append(minv.detach().cpu())
                idx_list.append(argmin.detach().cpu())

            d1 = torch.cat(d1_list, dim=0).numpy().astype(np.float64)
            if return_indices:
                nn_idx = torch.cat(idx_list, dim=0).numpy().astype(np.int64)
                return d1, nn_idx
            return (d1,)

        else:
            from sklearn.neighbors import NearestNeighbors

            nbrs = NearestNeighbors(
                n_neighbors=1, algorithm="auto", n_jobs=-1, metric=self.metric
            )
            nbrs.fit(Xdb)
            dists, inds = nbrs.kneighbors(Xq)
            d1 = (dists.astype(np.float64) ** 2)[:, 0]
            if return_indices:
                return d1, inds[:, 0].astype(np.int64)
            return (d1,)

    def compute(self) -> Dict[str, float]:
        """
        Compute PRDC-F1 score with precision and recall.

        Returns:
            Dictionary with "f1", "precision", "recall" keys, or NaN values if insufficient data
        """
        nan_result = {"f1": float("nan"), "precision": float("nan"), "recall": float("nan")}

        if not self._gen_list or not self._real_list:
            return nan_result

        Xg = self._filter_finite_rows(np.concatenate(self._gen_list, axis=0))
        Xr = self._filter_finite_rows(np.concatenate(self._real_list, axis=0))
        Ng, Nr = Xg.shape[0], Xr.shape[0]

        if Ng == 0 or Nr == 0:
            return nan_result

        n_cap = self.max_samples if self.max_samples is not None else min(Ng, Nr)
        n = min(Ng, Nr, n_cap)

        if n < 2:
            logger.warning(f"[{self.name}] insufficient samples n={n}")
            return nan_result

        idx_g = self._sample_indices(Ng, n, self.seed)
        idx_r = self._sample_indices(Nr, n, self.seed)
        Xg = Xg[idx_g]
        Xr = Xr[idx_r]

        k_eff = min(self.k, n - 1)
        if k_eff < 1:
            logger.warning(f"[{self.name}] invalid k_eff={k_eff} for n={n}")
            return nan_result

        backend = self._select_backend()

        try:
            r_gen = self._compute_self_kth_radius(Xg, k_eff, backend)
            r_real = self._compute_self_kth_radius(Xr, k_eff, backend)
            d_g2r, nn_g2r = self._compute_cross_1nn(Xg, Xr, backend, return_indices=True)
            d_r2g, nn_r2g = self._compute_cross_1nn(Xr, Xg, backend, return_indices=True)
        except Exception as e:
            logger.warning(f"[{self.name}] backend {backend} failed ({e}); fallback to sklearn")
            r_gen = self._compute_self_kth_radius(Xg, k_eff, "sklearn")
            r_real = self._compute_self_kth_radius(Xr, k_eff, "sklearn")
            d_g2r, nn_g2r = self._compute_cross_1nn(Xg, Xr, "sklearn", return_indices=True)
            d_r2g, nn_r2g = self._compute_cross_1nn(Xr, Xg, "sklearn", return_indices=True)

        precision = float(np.mean(d_g2r <= r_real[nn_g2r]))
        recall = float(np.mean(d_r2g <= r_gen[nn_r2g]))
        f1 = 0.0 if (precision + recall) == 0.0 else float(2.0 * precision * recall / (precision + recall))

        logger.info(
            f"[{self.name}] precision={precision:.6f} recall={recall:.6f} "
            f"f1={f1:.6f} n={n} k_eff={k_eff} backend={backend}"
        )
        return {"f1": f1, "precision": precision, "recall": recall}


@Registry.register_metric("joint_prdc_f1")
class JointPRDCF1Metric(CollectiveMetric):
    """
    Joint PRDC-F1 metric (TS + text).

    Computes PRDC-F1 on normalized joint (TS, text) embeddings.
    Each modality is standardized independently before concatenation
    and weighted by configurable weights.

    Legacy source: metrics.py:1474-1618
    """

    def __init__(
        self,
        name: str = "joint_prdc_f1",
        k: int = 5,
        max_samples: Optional[int] = None,
        seed: int = 0,
        metric: str = "euclidean",
        weights: Optional[Dict[str, float]] = None,
        normalize: str = "standard",
        backend: str = "auto",
        knn_chunk_size: int = 4096,
    ):
        """
        Initialize Joint PRDC-F1 metric.

        Args:
            name: Metric identifier
            k: Number of nearest neighbors
            max_samples: Maximum samples to use
            seed: Random seed
            metric: Distance metric
            weights: Modality weights {"ts": 1.0, "text": 1.0}
            normalize: Normalization method ("standard" or "none")
            backend: Backend selection
            knn_chunk_size: Chunk size for batched computation
        """
        super().__init__(name)
        self.k = int(k)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.seed = int(seed)
        self.metric = str(metric)
        self.weights = weights or {"ts": 1.0, "text": 1.0}
        self.normalize = str(normalize)
        self.backend = str(backend)
        self.knn_chunk_size = int(knn_chunk_size)

        self._gen_ts: List[np.ndarray] = []
        self._real_ts: List[np.ndarray] = []
        self._txt: List[np.ndarray] = []

    @property
    def requires_clip(self) -> bool:
        """Joint PRDC-F1 requires CLIP embeddings."""
        return True

    def reset(self) -> None:
        """Reset accumulators."""
        super().reset()
        self._gen_ts = []
        self._real_ts = []
        self._txt = []

    def update(self, batch_data: Dict[str, Any]) -> None:
        """
        Collect embeddings for all modalities.

        Args:
            batch_data: Must contain 'ts_gen_emb', 'ts_gt_emb', 'cap_emb'
        """
        ts_gen_emb: Tensor = batch_data["ts_gen_emb"]
        ts_gt_emb: Tensor = batch_data["ts_gt_emb"]
        cap_emb: Tensor = batch_data["cap_emb"]

        B = ts_gen_emb.shape[0]
        if ts_gt_emb.shape[0] != B or cap_emb.shape[0] != B:
            logger.warning(f"[{self.name}] batch size mismatch, skip this batch")
            return

        self._gen_ts.append(ts_gen_emb.detach().cpu().numpy().astype(np.float64))
        self._real_ts.append(ts_gt_emb.detach().cpu().numpy().astype(np.float64))
        self._txt.append(cap_emb.detach().cpu().numpy().astype(np.float64))

    @staticmethod
    def _filter_finite_rows(x: np.ndarray) -> np.ndarray:
        """Filter rows with non-finite values."""
        mask = np.isfinite(x).all(axis=1)
        return x[mask]

    @staticmethod
    def _sample_indices(n: int, k: int, seed: int) -> np.ndarray:
        """Sample k indices from n."""
        if k >= n:
            return np.arange(n, dtype=np.int64)
        rng = np.random.RandomState(seed)
        return rng.choice(n, size=k, replace=False)

    def _select_backend(self) -> str:
        """Select best available backend."""
        if self.backend in ("faiss", "torch", "sklearn"):
            return self.backend

        if self.metric == "euclidean" and _HAS_FAISS:
            try:
                if faiss.get_num_gpus() > 0:
                    return "faiss"
            except Exception:
                pass

        if self.metric == "euclidean" and torch.cuda.is_available():
            return "torch"

        return "sklearn"

    def _compute_prdc_like(
        self, Xg: np.ndarray, Xr: np.ndarray, k_eff: int, backend: str
    ) -> Dict[str, float]:
        """Compute PRDC-F1 with precision and recall using internal PRDCF1Metric."""
        base = PRDCF1Metric(
            name=f"{self.name}_inner",
            k=k_eff,
            max_samples=Xg.shape[0],
            seed=self.seed,
            metric=self.metric,
            backend=backend,
            knn_chunk_size=self.knn_chunk_size,
        )

        r_gen = base._compute_self_kth_radius(Xg, k_eff, backend)
        r_real = base._compute_self_kth_radius(Xr, k_eff, backend)
        d_g2r, nn_g2r = base._compute_cross_1nn(Xg, Xr, backend, return_indices=True)
        d_r2g, nn_r2g = base._compute_cross_1nn(Xr, Xg, backend, return_indices=True)

        precision = float(np.mean(d_g2r <= r_real[nn_g2r]))
        recall = float(np.mean(d_r2g <= r_gen[nn_r2g]))
        f1 = 0.0 if (precision + recall) == 0.0 else float(2.0 * precision * recall / (precision + recall))

        return {"f1": f1, "precision": precision, "recall": recall}

    def compute(self) -> Dict[str, float]:
        """
        Compute Joint PRDC-F1 score with precision and recall.

        Returns:
            Dictionary with "f1", "precision", "recall" keys, or NaN values if insufficient data
        """
        nan_result = {"f1": float("nan"), "precision": float("nan"), "recall": float("nan")}

        if not self._gen_ts or not self._real_ts or not self._txt:
            return nan_result

        gen_ts = self._filter_finite_rows(np.concatenate(self._gen_ts, axis=0))
        real_ts = self._filter_finite_rows(np.concatenate(self._real_ts, axis=0))
        txt = self._filter_finite_rows(np.concatenate(self._txt, axis=0))

        if gen_ts.shape[0] == 0 or real_ts.shape[0] == 0 or txt.shape[0] == 0:
            return nan_result

        # Normalize per block using global stats
        all_ts = np.concatenate([gen_ts, real_ts], axis=0)
        mu_ts = all_ts.mean(axis=0)
        std_ts = all_ts.std(axis=0)
        mu_txt = txt.mean(axis=0)
        std_txt = txt.std(axis=0)
        eps = 1e-6

        def norm(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
            if self.normalize == "standard":
                return (x - mu) / (sigma + eps)
            return x

        # Normalize and weight
        gen_ts_n = norm(gen_ts, mu_ts, std_ts) * self.weights["ts"]
        real_ts_n = norm(real_ts, mu_ts, std_ts) * self.weights["ts"]
        txt_n = norm(txt, mu_txt, std_txt) * self.weights["text"]

        # Concatenate to form joint embeddings
        gen_joint = np.concatenate([gen_ts_n, txt_n], axis=1)
        real_joint = np.concatenate([real_ts_n, txt_n], axis=1)

        Ng, Nr = gen_joint.shape[0], real_joint.shape[0]
        n_cap = self.max_samples if self.max_samples is not None else min(Ng, Nr)
        n = min(Ng, Nr, n_cap)

        if n < 2:
            logger.warning(f"[{self.name}] insufficient samples n={n}")
            return nan_result

        idx_g = self._sample_indices(Ng, n, self.seed)
        idx_r = self._sample_indices(Nr, n, self.seed)
        Xg = gen_joint[idx_g]
        Xr = real_joint[idx_r]

        k_eff = min(self.k, n - 1)
        if k_eff < 1:
            logger.warning(f"[{self.name}] invalid k_eff={k_eff} for n={n}")
            return nan_result

        backend = self._select_backend()

        try:
            result = self._compute_prdc_like(Xg, Xr, k_eff, backend)
        except Exception as e:
            logger.warning(f"[{self.name}] backend {backend} failed ({e}); fallback to sklearn")
            result = self._compute_prdc_like(Xg, Xr, k_eff, "sklearn")

        logger.info(
            f"[{self.name}] f1={result['f1']:.6f} precision={result['precision']:.6f} "
            f"recall={result['recall']:.6f} n={n} k_eff={k_eff} backend={backend}"
        )
        return result


__all__ = [
    "PRDCF1Metric",
    "JointPRDCF1Metric",
]
