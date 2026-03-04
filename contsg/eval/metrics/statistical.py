"""
Statistical evaluation metrics.

This module implements statistical distribution comparison metrics including
autocorrelation difference, skewness difference, kurtosis difference, and
marginal distribution difference.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from torch import Tensor

from contsg.eval.metrics.base import BaseMetric
from contsg.registry import Registry

logger = logging.getLogger(__name__)


@Registry.register_metric("acd")
class ACDMetric(BaseMetric):
    """
    Autocorrelation Difference (ACD) metric.

    Computes mean ACF for real and generated sequences (per feature, lag 0..max_lag),
    then averages |ACF_real - ACF_gen| over all features and lags.

    Uses GPU-vectorized computation with masking for variable-length sequences.

    Legacy source: metrics.py:831-930
    """

    def __init__(self, name: str = "acd", max_lag: int = 50):
        """
        Initialize ACD metric.

        Args:
            name: Metric identifier
            max_lag: Maximum lag for autocorrelation computation
        """
        super().__init__(name)
        self.max_lag = int(max_lag)
        self.reset()

    def reset(self) -> None:
        """Reset accumulators."""
        self.sum_acf_real: Optional[Tensor] = None  # (F, K)
        self.sum_acf_gen: Optional[Tensor] = None  # (F, K)
        self.cnt_real: Optional[Tensor] = None  # (F, K)
        self.cnt_gen: Optional[Tensor] = None  # (F, K)
        self.F: Optional[int] = None

    @torch.no_grad()
    def update(self, batch_data: Dict[str, Any]) -> None:
        """
        Accumulate ACF from batch.

        Args:
            batch_data: Must contain 'pred', 'ts', 'ts_len'
        """
        pred: Tensor = batch_data["pred"]  # (B, L, F)
        ts: Tensor = batch_data["ts"]  # (B, L, F)
        lengths: Tensor = batch_data["ts_len"]  # (B,)

        device = pred.device
        B, L, F = pred.shape
        K = min(self.max_lag, max(L - 1, 0)) + 1

        # Initialize accumulators
        if self.sum_acf_real is None:
            self.F = F
            self.sum_acf_real = torch.zeros(F, K, device=device)
            self.sum_acf_gen = torch.zeros(F, K, device=device)
            self.cnt_real = torch.zeros(F, K, device=device)
            self.cnt_gen = torch.zeros(F, K, device=device)
        elif self.sum_acf_real.shape[1] != K:
            # Handle different sequence lengths across batches
            newK = max(self.sum_acf_real.shape[1], K)

            def _pad(mat: Tensor) -> Tensor:
                pad_cols = newK - mat.shape[1]
                if pad_cols <= 0:
                    return mat[:, :newK]
                return torch.cat(
                    [mat, torch.zeros(mat.shape[0], pad_cols, device=mat.device)],
                    dim=1,
                )

            self.sum_acf_real = _pad(self.sum_acf_real)
            self.sum_acf_gen = _pad(self.sum_acf_gen)
            self.cnt_real = _pad(self.cnt_real)
            self.cnt_gen = _pad(self.cnt_gen)
            K = newK

        def _accumulate_acf(
            x: Tensor, acc_sum: Tensor, acc_cnt: Tensor
        ) -> None:
            """Accumulate ACF with masking for variable lengths."""
            B, L, F = x.shape
            t_idx = torch.arange(L, device=device)[None, :]
            mask_full = (t_idx < lengths[:, None]).unsqueeze(-1).float()  # (B, L, 1)

            # Mean and variance with masking
            denom_t = torch.clamp(mask_full.sum(dim=1), min=1.0)  # (B, 1)
            mu = (x * mask_full).sum(dim=1) / denom_t  # (B, F)
            xc = x - mu[:, None, :]  # (B, L, F) - centered
            denom = (xc.pow(2) * mask_full).sum(dim=1)  # (B, F)
            eps = 1e-8

            max_k = min(self.max_lag, max(L - 1, 0))
            for k in range(max_k + 1):
                if k == 0:
                    num = (xc.pow(2) * mask_full).sum(dim=1)  # (B, F)
                    valid = denom > eps  # (B, F)
                else:
                    mtk = (t_idx < (lengths[:, None] - k)).unsqueeze(-1)  # (B, L, 1)
                    if L - k <= 0:
                        continue
                    x0 = xc[:, : L - k, :]
                    x1 = xc[:, k:, :]
                    mk = mtk[:, : L - k, :].float()
                    num = (x0 * x1 * mk).sum(dim=1)  # (B, F)
                    valid = (denom > eps) & (lengths[:, None] > k)

                acf = torch.zeros(B, F, device=device)
                acf[valid] = num[valid] / (denom[valid] + eps)

                # Aggregate to (F, k)
                acc_sum[:, k] += acf.sum(dim=0)
                acc_cnt[:, k] += valid.float().sum(dim=0)

        _accumulate_acf(ts, self.sum_acf_real, self.cnt_real)
        _accumulate_acf(pred, self.sum_acf_gen, self.cnt_gen)

    def compute(self) -> float:
        """
        Compute mean absolute ACF difference.

        Returns:
            Mean |ACF_real - ACF_gen|, or NaN if no data
        """
        if self.sum_acf_real is None:
            return float("nan")

        # Compute means
        mean_real = self.sum_acf_real / torch.clamp(self.cnt_real, min=1.0)
        mean_gen = self.sum_acf_gen / torch.clamp(self.cnt_gen, min=1.0)

        # Only consider positions where both have valid counts
        valid = (self.cnt_real > 0) & (self.cnt_gen > 0)
        if valid.sum() == 0:
            return float("nan")

        diff = torch.abs(mean_real - mean_gen)
        return float(diff[valid].mean().detach().cpu().item())


@Registry.register_metric("sd")
class SkewnessDiffMetric(BaseMetric):
    """
    Skewness Difference (SD) metric.

    Computes |skew(reference) - skew(generated)|, averaged over features.

    Supports two modes:
    - reference_split="train": Reference stats injected via set_reference_stats()
    - reference_split="test": Reference computed from test set ground truth

    Uses GPU for accumulation, CPU float64 for final computation.

    Legacy source: metrics.py:932-1020
    """

    def __init__(self, name: str = "sd", reference_split: str = "train"):
        """
        Initialize Skewness Difference metric.

        Args:
            name: Metric identifier
            reference_split: "train" or "test" for reference source
        """
        # Set reference_split before super().__init__ because reset() needs it
        self.reference_split = reference_split
        self.ref_skew: Optional[Tensor] = None
        self.gen_sums: Optional[Dict[str, Tensor]] = None
        self.real_sums: Optional[Dict[str, Tensor]] = None
        self.F: Optional[int] = None
        super().__init__(name)

    def reset(self) -> None:
        """Reset accumulators (preserve train reference if applicable)."""
        if self.reference_split != "train":
            self.ref_skew = None
        self.gen_sums = None
        self.real_sums = None
        self.F = None

    @staticmethod
    def _init_sums(F: int) -> Dict[str, Tensor]:
        """Initialize moment accumulators."""
        return {
            "sum1": torch.zeros(F, dtype=torch.float64),
            "sum2": torch.zeros(F, dtype=torch.float64),
            "sum3": torch.zeros(F, dtype=torch.float64),
            "cnt": torch.zeros(F, dtype=torch.float64),
        }

    @staticmethod
    @torch.no_grad()
    def _accumulate_moments(
        x: Tensor, lengths: Tensor, dst: Dict[str, Tensor]
    ) -> None:
        """Accumulate moments with masking."""
        device = x.device
        B, L, F = x.shape
        t = torch.arange(L, device=device)[None, :]
        mask = (t < lengths[:, None]).unsqueeze(-1).float()  # (B, L, 1)
        x_mask = x * mask

        # Reduce to CPU float64
        dst["sum1"] += x_mask.sum(dim=(0, 1)).detach().cpu().to(torch.float64)
        dst["sum2"] += (x_mask * x).sum(dim=(0, 1)).detach().cpu().to(torch.float64)
        dst["sum3"] += (x_mask * (x ** 2)).sum(dim=(0, 1)).detach().cpu().to(torch.float64)
        dst["cnt"] += mask.sum(dim=(0, 1)).detach().cpu().to(torch.float64).squeeze(-1)

    def set_reference_stats(self, ref_skew: np.ndarray) -> None:
        """
        Inject pre-computed reference skewness.

        Args:
            ref_skew: (F,) array of reference skewness per feature
        """
        self.ref_skew = torch.from_numpy(ref_skew).to(torch.float64)

    @torch.no_grad()
    def update(self, batch_data: Dict[str, Any]) -> None:
        """
        Accumulate moments from batch.

        Args:
            batch_data: Must contain 'pred', 'ts', 'ts_len'
        """
        pred: Tensor = batch_data["pred"]
        ts: Tensor = batch_data["ts"]
        lengths: Tensor = batch_data["ts_len"]
        B, L, F = pred.shape

        if self.gen_sums is None:
            self.F = F
            self.gen_sums = self._init_sums(F)
            if self.reference_split == "test":
                self.real_sums = self._init_sums(F)

        # Generated moments
        self._accumulate_moments(pred, lengths, self.gen_sums)

        # Real moments if test reference
        if self.reference_split == "test":
            self._accumulate_moments(ts, lengths, self.real_sums)

    def _finalize_skew(self, sums: Dict[str, Tensor]) -> Tensor:
        """Compute skewness from accumulated moments."""
        eps = 1e-8
        n = torch.clamp(sums["cnt"], min=1.0)
        E1 = sums["sum1"] / n
        E2 = sums["sum2"] / n
        E3 = sums["sum3"] / n

        var = torch.clamp(E2 - E1 ** 2, min=0.0)
        sigma = torch.sqrt(var + eps)
        m3 = E3 - 3 * E1 * E2 + 2 * (E1 ** 3)
        skew = m3 / (sigma ** 3 + eps)
        return skew

    def compute(self) -> float:
        """
        Compute mean absolute skewness difference.

        Returns:
            Mean |ref_skew - gen_skew|, or NaN if no data
        """
        if self.gen_sums is None:
            return float("nan")

        gen_skew = self._finalize_skew(self.gen_sums)

        if self.reference_split == "train":
            if self.ref_skew is None:
                return float("nan")
            ref_skew = self.ref_skew
        else:
            ref_skew = self._finalize_skew(self.real_sums)

        diff = torch.abs(ref_skew - gen_skew).mean().item()
        return float(diff)


@Registry.register_metric("kd")
class KurtosisDiffMetric(BaseMetric):
    """
    Kurtosis Difference (KD) metric.

    Computes |kurt(reference) - kurt(generated)|, averaged over features.

    Supports two modes:
    - reference_split="train": Reference stats injected via set_reference_stats()
    - reference_split="test": Reference computed from test set ground truth

    Legacy source: metrics.py:1022-1111
    """

    def __init__(self, name: str = "kd", reference_split: str = "train"):
        """
        Initialize Kurtosis Difference metric.

        Args:
            name: Metric identifier
            reference_split: "train" or "test" for reference source
        """
        # Set reference_split before super().__init__ because reset() needs it
        self.reference_split = reference_split
        self.ref_kurt: Optional[Tensor] = None
        self.gen_sums: Optional[Dict[str, Tensor]] = None
        self.real_sums: Optional[Dict[str, Tensor]] = None
        self.F: Optional[int] = None
        super().__init__(name)

    def reset(self) -> None:
        """Reset accumulators (preserve train reference if applicable)."""
        if self.reference_split != "train":
            self.ref_kurt = None
        self.gen_sums = None
        self.real_sums = None
        self.F = None

    @staticmethod
    def _init_sums(F: int) -> Dict[str, Tensor]:
        """Initialize moment accumulators (up to 4th moment)."""
        return {
            "sum1": torch.zeros(F, dtype=torch.float64),
            "sum2": torch.zeros(F, dtype=torch.float64),
            "sum3": torch.zeros(F, dtype=torch.float64),
            "sum4": torch.zeros(F, dtype=torch.float64),
            "cnt": torch.zeros(F, dtype=torch.float64),
        }

    @staticmethod
    @torch.no_grad()
    def _accumulate_moments(
        x: Tensor, lengths: Tensor, dst: Dict[str, Tensor]
    ) -> None:
        """Accumulate moments up to 4th power with masking."""
        device = x.device
        B, L, F = x.shape
        t = torch.arange(L, device=device)[None, :]
        mask = (t < lengths[:, None]).unsqueeze(-1).float()
        x_mask = x * mask

        dst["sum1"] += x_mask.sum(dim=(0, 1)).detach().cpu().to(torch.float64)
        dst["sum2"] += (x_mask * x).sum(dim=(0, 1)).detach().cpu().to(torch.float64)

        x2 = x * x
        dst["sum3"] += ((x2 * x) * mask).sum(dim=(0, 1)).detach().cpu().to(torch.float64)
        dst["sum4"] += ((x2 * x2) * mask).sum(dim=(0, 1)).detach().cpu().to(torch.float64)
        dst["cnt"] += mask.sum(dim=(0, 1)).detach().cpu().to(torch.float64).squeeze(-1)

    def set_reference_stats(self, ref_kurt: np.ndarray) -> None:
        """
        Inject pre-computed reference kurtosis.

        Args:
            ref_kurt: (F,) array of reference kurtosis per feature
        """
        self.ref_kurt = torch.from_numpy(ref_kurt).to(torch.float64)

    @torch.no_grad()
    def update(self, batch_data: Dict[str, Any]) -> None:
        """
        Accumulate moments from batch.

        Args:
            batch_data: Must contain 'pred', 'ts', 'ts_len'
        """
        pred: Tensor = batch_data["pred"]
        ts: Tensor = batch_data["ts"]
        lengths: Tensor = batch_data["ts_len"]
        B, L, F = pred.shape

        if self.gen_sums is None:
            self.F = F
            self.gen_sums = self._init_sums(F)
            if self.reference_split == "test":
                self.real_sums = self._init_sums(F)

        self._accumulate_moments(pred, lengths, self.gen_sums)
        if self.reference_split == "test":
            self._accumulate_moments(ts, lengths, self.real_sums)

    def _finalize_kurt(self, sums: Dict[str, Tensor]) -> Tensor:
        """Compute kurtosis from accumulated moments."""
        eps = 1e-8
        n = torch.clamp(sums["cnt"], min=1.0)
        E1 = sums["sum1"] / n
        E2 = sums["sum2"] / n
        E3 = sums["sum3"] / n
        E4 = sums["sum4"] / n

        var = torch.clamp(E2 - E1 ** 2, min=0.0)
        sigma = torch.sqrt(var + eps)
        m4 = E4 - 4 * E1 * E3 + 6 * (E1 ** 2) * E2 - 3 * (E1 ** 4)
        kurt = m4 / (sigma ** 4 + eps)
        return kurt

    def compute(self) -> float:
        """
        Compute mean absolute kurtosis difference.

        Returns:
            Mean |ref_kurt - gen_kurt|, or NaN if no data
        """
        if self.gen_sums is None:
            return float("nan")

        gen_kurt = self._finalize_kurt(self.gen_sums)

        if self.reference_split == "train":
            if self.ref_kurt is None:
                return float("nan")
            ref_kurt = self.ref_kurt
        else:
            ref_kurt = self._finalize_kurt(self.real_sums)

        diff = torch.abs(ref_kurt - gen_kurt).mean().item()
        return float(diff)


@Registry.register_metric("mdd")
class MDDMetric(BaseMetric):
    """
    Marginal Distribution Difference (MDD) metric.

    Compares per-timestep, per-feature histograms between reference and generated
    sequences. Uses histogram bins computed from reference data.

    Supports two modes:
    - reference_split="train": Bins and reference hist injected via set_reference()
    - reference_split="test": Bins computed from test set, hists accumulated during eval

    Legacy source: metrics.py:1113-1274
    """

    def __init__(
        self, name: str = "mdd", num_bins: int = 32, reference_split: str = "train"
    ):
        """
        Initialize MDD metric.

        Args:
            name: Metric identifier
            num_bins: Number of histogram bins
            reference_split: "train" or "test" for reference source
        """
        # Set reference_split and num_bins before super().__init__ because reset() needs them
        self.reference_split = reference_split
        self.num_bins = int(num_bins)
        super().__init__(name)

    def reset(self) -> None:
        """Reset accumulators."""
        # Always initialize these attributes (they may be set via set_reference for train mode)
        # Only reset them to None if not in train mode (preserve injected reference)
        if self.reference_split != "train":
            self.bin_edges: Optional[Tensor] = None  # (L, F, B+1)
            self.ref_hist: Optional[Tensor] = None  # (L, F, B)
            self.ref_tot: Optional[Tensor] = None  # (L, F)
        elif not hasattr(self, 'bin_edges'):
            # First initialization in train mode
            self.bin_edges: Optional[Tensor] = None
            self.ref_hist: Optional[Tensor] = None
            self.ref_tot: Optional[Tensor] = None

        self.gen_hist: Optional[Tensor] = None  # (L, F, B)
        self.gen_tot: Optional[Tensor] = None  # (L, F)

        # Test mode caches
        self._cached_real: List[Tuple[Tensor, Tensor]] = []
        self._cached_pred: List[Tuple[Tensor, Tensor]] = []
        self._min: Optional[Tensor] = None
        self._max: Optional[Tensor] = None
        self.L: Optional[int] = None
        self.F: Optional[int] = None

    def set_reference(
        self, bin_edges: np.ndarray, ref_hist: np.ndarray, ref_tot: np.ndarray
    ) -> None:
        """
        Inject pre-computed reference bins and histogram.

        Args:
            bin_edges: (L, F, num_bins+1) histogram bin edges
            ref_hist: (L, F, num_bins) reference histogram counts
            ref_tot: (L, F) total counts per position
        """
        self.bin_edges = torch.from_numpy(bin_edges).to(torch.float32)
        self.ref_hist = torch.from_numpy(ref_hist).to(torch.float64)
        self.ref_tot = torch.from_numpy(ref_tot).to(torch.float64)

    @staticmethod
    def _accumulate_hist_for_batch(
        values_t_f: Tensor, edges_t_f: Tensor, out_counts: Tensor
    ) -> None:
        """Accumulate histogram counts using bucketize."""
        idx = torch.bucketize(values_t_f, edges_t_f) - 1
        idx = idx.clamp(min=0, max=out_counts.shape[0] - 1)
        ones = torch.ones_like(idx, dtype=out_counts.dtype)
        out_counts.scatter_add_(0, idx, ones)

    @torch.no_grad()
    def update(self, batch_data: Dict[str, Any]) -> None:
        """
        Accumulate histogram data from batch.

        Args:
            batch_data: Must contain 'pred', 'ts', 'ts_len'
        """
        pred: Tensor = batch_data["pred"]  # (B, L, F)
        ts: Tensor = batch_data["ts"]
        lengths: Tensor = batch_data["ts_len"]
        Bbatch, L, F = pred.shape
        device = pred.device

        if self.L is None:
            self.L = L
            self.F = F

        # Train reference mode: accumulate generated histogram
        if self.reference_split == "train" and self.bin_edges is not None:
            if self.gen_hist is None:
                self.gen_hist = torch.zeros(L, F, self.num_bins, dtype=torch.float64)
                self.gen_tot = torch.zeros(L, F, dtype=torch.float64)

            edges = self.bin_edges  # CPU float32
            for t in range(L):
                valid_b = lengths > t
                if valid_b.any():
                    vals_tf = pred[valid_b, t, :]  # (Nv, F) on device
                    for f in range(F):
                        v = vals_tf[:, f].detach().cpu().to(torch.float32)
                        e = edges[t, f]  # (B+1,)
                        cnt = torch.zeros(self.num_bins, dtype=torch.float64)
                        self._accumulate_hist_for_batch(v, e, cnt)
                        self.gen_hist[t, f] += cnt
                        self.gen_tot[t, f] += float(v.shape[0])
        else:
            # Test reference mode: cache data for final computation
            self._cached_real.append(
                (ts.detach().cpu().to(torch.float16), lengths.detach().cpu())
            )
            self._cached_pred.append(
                (pred.detach().cpu().to(torch.float16), lengths.detach().cpu())
            )

            # Update min/max for bin edge computation
            t_idx = torch.arange(L, device=ts.device)[None, :]
            mask = (t_idx < lengths[:, None]).unsqueeze(-1)
            ts_masked = ts.masked_fill(~mask, float("inf"))
            ts_masked_max = ts.masked_fill(~mask, float("-inf"))
            min_b = ts_masked.min(dim=0).values  # (L, F)
            max_b = ts_masked_max.max(dim=0).values
            min_b = min_b.detach().cpu()
            max_b = max_b.detach().cpu()

            if self._min is None:
                self._min = min_b
                self._max = max_b
            else:
                self._min = torch.minimum(self._min, min_b)
                self._max = torch.maximum(self._max, max_b)

    def _compute_hist_from_cache(self) -> None:
        """Build histograms from cached test data."""
        eps = 1e-6
        L, F = int(self._min.shape[0]), int(self._min.shape[1])

        # Build bin edges from min/max
        edges = torch.empty(L, F, self.num_bins + 1, dtype=torch.float32)
        for t in range(L):
            for f in range(F):
                lo = float(self._min[t, f].item())
                hi = float(self._max[t, f].item())
                if not np.isfinite(lo) or not np.isfinite(hi):
                    lo, hi = -0.5, 0.5
                if abs(hi - lo) < eps:
                    lo, hi = lo - 0.5, lo + 0.5
                edges[t, f] = torch.linspace(
                    lo, hi, self.num_bins + 1, dtype=torch.float32
                )

        ref_hist = torch.zeros(L, F, self.num_bins, dtype=torch.float64)
        ref_tot = torch.zeros(L, F, dtype=torch.float64)
        gen_hist = torch.zeros(L, F, self.num_bins, dtype=torch.float64)
        gen_tot = torch.zeros(L, F, dtype=torch.float64)

        # Compute histograms for real data
        for ts16, len_cpu in self._cached_real:
            ts = ts16.to(torch.float32)
            B, Lc, Fc = ts.shape
            for t in range(Lc):
                valid = (len_cpu > t).numpy()
                if valid.any():
                    vals = ts[valid, t, :].to(torch.float32)
                    for f in range(Fc):
                        v = vals[:, f]
                        e = edges[t, f]
                        cnt = torch.zeros(self.num_bins, dtype=torch.float64)
                        self._accumulate_hist_for_batch(v, e, cnt)
                        ref_hist[t, f] += cnt
                        ref_tot[t, f] += float(v.shape[0])

        # Compute histograms for generated data
        for pr16, len_cpu in self._cached_pred:
            pr = pr16.to(torch.float32)
            B, Lc, Fc = pr.shape
            for t in range(Lc):
                valid = (len_cpu > t).numpy()
                if valid.any():
                    vals = pr[valid, t, :].to(torch.float32)
                    for f in range(Fc):
                        v = vals[:, f]
                        e = edges[t, f]
                        cnt = torch.zeros(self.num_bins, dtype=torch.float64)
                        self._accumulate_hist_for_batch(v, e, cnt)
                        gen_hist[t, f] += cnt
                        gen_tot[t, f] += float(v.shape[0])

        self.bin_edges = edges
        self.ref_hist = ref_hist
        self.ref_tot = ref_tot
        self.gen_hist = gen_hist
        self.gen_tot = gen_tot

    def compute(self) -> float:
        """
        Compute mean absolute histogram difference.

        Returns:
            Mean |ref_prob - gen_prob|, or NaN if no data
        """
        if self.reference_split == "test":
            if len(self._cached_real) == 0:
                return float("nan")
            self._compute_hist_from_cache()
        else:
            # Train reference requires pre-injected bins/hist
            if self.bin_edges is None or self.ref_hist is None or self.gen_hist is None:
                return float("nan")

        # Normalize to probabilities
        eps = 1e-12
        ref_p = self.ref_hist / (self.ref_tot.unsqueeze(-1) + eps)
        gen_p = self.gen_hist / (self.gen_tot.unsqueeze(-1) + eps)

        # Mean absolute difference
        diff = torch.abs(ref_p - gen_p).mean().item()
        return float(diff)


__all__ = [
    "ACDMetric",
    "SkewnessDiffMetric",
    "KurtosisDiffMetric",
    "MDDMetric",
]
