"""
Segment-level parameter classification metrics.

This module implements segment-wise shapelet parameter classification
for evaluating local generation capability on synthetic datasets.

Two-stage classification:
- Stage 1: Presence classification (none vs present)
- Stage 2: Parameter classification (width/amp/skew) for present segments
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from contsg.eval.metrics.base import CollectiveMetric
from contsg.registry import Registry

logger = logging.getLogger(__name__)


# ==============================================================================
# Classifier Models
# ==============================================================================


class PresenceClassifier1D(nn.Module):
    """
    Binary classifier for detecting shapelet presence in time series segments.

    Architecture:
    - Conv Block 1: 1→32 channels, kernel=7, pool=2
    - Conv Block 2: 32→64 channels, kernel=5, pool=2
    - Conv Block 3: 64→128 channels, kernel=3
    - Global Average Pooling
    - FC: 128→64→2

    Legacy source: segment_classifier_models.py:14-88
    """

    def __init__(self, segment_len: int):
        """
        Initialize presence classifier.

        Args:
            segment_len: Length of input segment (42 or 43 timesteps typically)
        """
        super().__init__()
        self.segment_len = segment_len

        # Conv Block 1: Capture local patterns
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        # Conv Block 2: Capture intermediate features
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        # Conv Block 3: Capture high-level features
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Global pooling and classification
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: (batch, 1, seg_len) input segments

        Returns:
            logits: (batch, 2) classification logits for [none, present]
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Global pooling
        x = self.gap(x)  # (batch, 128, 1)
        x = x.squeeze(-1)  # (batch, 128)

        # Classification head
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ParameterClassifier1D(nn.Module):
    """
    Multi-task classifier for predicting shapelet parameters.

    Predicts three properties:
    - Width: sharp/narrow/broad
    - Amplitude: subtle/moderate/prominent
    - Skewness: left-leaning/symmetric/right-leaning

    Architecture:
    - Deeper Conv Backbone (4 layers): 1→64→128→256→256
    - Global Average Pooling
    - Shared FC: 256→128
    - Three independent heads: width/amp/skew (each 128→3)

    Legacy source: segment_classifier_models.py:91-180
    """

    def __init__(self, segment_len: int):
        """
        Initialize parameter classifier.

        Args:
            segment_len: Length of input segment (42 or 43 timesteps typically)
        """
        super().__init__()
        self.segment_len = segment_len

        # Deeper backbone for fine-grained parameter discrimination
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)

        # Global pooling and shared feature extraction
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc_shared = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.4)

        # Three independent classification heads
        self.fc_width = nn.Linear(128, 3)  # sharp/narrow/broad
        self.fc_amp = nn.Linear(128, 3)    # subtle/moderate/prominent
        self.fc_skew = nn.Linear(128, 3)   # left-leaning/symmetric/right-leaning

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, 1, seg_len) input segments

        Returns:
            width_logits: (batch, 3) logits for width classification
            amp_logits: (batch, 3) logits for amplitude classification
            skew_logits: (batch, 3) logits for skewness classification
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Conv Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        # Global pooling
        x = self.gap(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)

        # Shared feature layer
        x = self.fc_shared(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Three independent heads
        width_logits = self.fc_width(x)
        amp_logits = self.fc_amp(x)
        skew_logits = self.fc_skew(x)

        return width_logits, amp_logits, skew_logits


# ==============================================================================
# Segment Parameter Accuracy Metric
# ==============================================================================


@Registry.register_metric("segment_param_acc")
class SegmentParameterAccuracyMetric(CollectiveMetric):
    """
    Segment-level parameter classification accuracy metric.

    Evaluates generated time series by classifying each segment's parameters
    (width, amplitude, skewness) using pre-trained two-stage classifiers.

    Stage 1: Presence classification (none vs present)
    Stage 2: Parameter classification (width/amp/skew) - only for present segments

    This metric is specific to synth-u-parametric dataset which has:
    - 3 segments (beginning, middle, end)
    - 17 attributes including presence and parameter labels

    Legacy source: metrics.py:1621-1953
    """

    def __init__(
        self,
        name: str = "segment_param_acc",
        save_dir: str = "./segment_acc_outputs",
        checkpoint_dir: str = "./segment_classifiers",
        segment_len: int = 128,
        n_segments: int = 3,
        device: str = "cuda",
        seed: int = 42,
    ):
        """
        Initialize segment parameter accuracy metric.

        Args:
            name: Metric identifier
            save_dir: Directory to save detailed results
            checkpoint_dir: Directory containing classifier checkpoints
            segment_len: Total sequence length (default: 128)
            n_segments: Number of segments (default: 3)
            device: Device for inference
            seed: Random seed
        """
        super().__init__(name)
        self.save_dir = Path(save_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.segment_len = int(segment_len)
        self.n_segments = int(n_segments)
        self.seed = int(seed)

        try:
            self.device = torch.device(device)
        except Exception:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data accumulation containers
        self._pred_list: List[np.ndarray] = []
        self._ts_list: List[np.ndarray] = []
        self._attrs_list: List[np.ndarray] = []
        self._ts_len_list: List[np.ndarray] = []

        # Compute segment slices
        self.segment_slices = self._compute_segment_slices()

        # Attribute index mapping (synth-u-parametric has 17 attributes)
        # Indices 5/6/7: shapelet_begin/middle/end_presence
        # Indices 8-10: shapelet_begin width/amp/skew
        # Indices 11-13: shapelet_middle width/amp/skew
        # Indices 14-16: shapelet_end width/amp/skew
        self.attr_indices = {
            "beginning": {"presence": 5, "width": 8, "amp": 9, "skew": 10},
            "middle": {"presence": 6, "width": 11, "amp": 12, "skew": 13},
            "end": {"presence": 7, "width": 14, "amp": 15, "skew": 16},
        }

        logger.info(f"Initialized {name} with checkpoint_dir: {checkpoint_dir}")

    def reset(self) -> None:
        """Reset accumulated data."""
        self._pred_list = []
        self._ts_list = []
        self._attrs_list = []
        self._ts_len_list = []

    def update(self, batch_data: Dict[str, Any]) -> None:
        """
        Accumulate batch data for later evaluation.

        Args:
            batch_data: Must contain:
                - 'pred': (B, L, F) median prediction
                - 'ts': (B, L, F) ground truth
                - 'ts_len': (B,) sequence lengths
                - 'attrs_idx': (B, 17) attribute indices
        """
        if "attrs_idx" not in batch_data:
            logger.warning(f"[{self.name}] attrs_idx not in batch_data, skipping update")
            return

        pred = batch_data["pred"].detach().cpu().numpy()
        ts = batch_data["ts"].detach().cpu().numpy()
        ts_len = batch_data["ts_len"].detach().cpu().numpy()
        attrs_idx = batch_data["attrs_idx"].detach().cpu().numpy()

        self._pred_list.append(pred)
        self._ts_list.append(ts)
        self._attrs_list.append(attrs_idx)
        self._ts_len_list.append(ts_len)

    def compute(self) -> float:
        """
        Compute segment-level classification accuracy.

        Returns:
            Average joint accuracy across all segments, or NaN if no data
        """
        if len(self._pred_list) == 0:
            logger.warning(f"[{self.name}] No data collected, returning NaN")
            return float("nan")

        # Merge all batches
        all_pred = np.concatenate(self._pred_list, axis=0)  # (N, L, F)
        all_ts = np.concatenate(self._ts_list, axis=0)
        all_attrs = np.concatenate(self._attrs_list, axis=0)  # (N, 17)
        all_ts_len = np.concatenate(self._ts_len_list, axis=0)

        n_samples = all_pred.shape[0]
        logger.info(f"[{self.name}] Evaluating {n_samples} samples")

        # Load classifier models
        try:
            models = self._load_classifiers()
        except FileNotFoundError as e:
            logger.error(f"[{self.name}] Failed to load classifiers: {e}")
            return float("nan")

        # Evaluate each segment
        results: Dict[str, Any] = {}
        all_joint_accs: List[float] = []

        for seg_name in ["beginning", "middle", "end"]:
            seg_result = self._evaluate_segment(
                seg_name, all_pred, all_ts, all_attrs, models
            )
            results[seg_name] = seg_result
            all_joint_accs.append(seg_result["joint_acc"])

            logger.info(
                f"[{self.name}] {seg_name}: "
                f"presence={seg_result['presence_acc']:.4f}, "
                f"joint={seg_result['joint_acc']:.4f}"
            )

        # Compute average metrics
        avg_joint_acc = float(np.mean(all_joint_accs))
        results["average"] = {
            "joint_acc": avg_joint_acc,
            "presence_acc": float(np.mean([
                r["presence_acc"] for r in results.values()
                if isinstance(r, dict) and "presence_acc" in r
            ])),
            "width_acc": float(np.mean([
                r.get("width_acc", 0) for r in results.values()
                if isinstance(r, dict) and "width_acc" in r
            ])),
            "amp_acc": float(np.mean([
                r.get("amp_acc", 0) for r in results.values()
                if isinstance(r, dict) and "amp_acc" in r
            ])),
            "skew_acc": float(np.mean([
                r.get("skew_acc", 0) for r in results.values()
                if isinstance(r, dict) and "skew_acc" in r
            ])),
        }

        # Save detailed results
        self._save_results(results)

        logger.info(f"[{self.name}] Average joint accuracy: {avg_joint_acc:.4f}")
        return avg_joint_acc

    def _compute_segment_slices(self) -> Dict[str, tuple]:
        """Compute segment slices based on sequence length."""
        base = self.segment_len // self.n_segments
        remainder = self.segment_len % self.n_segments

        slices: Dict[str, tuple] = {}
        start = 0
        for idx, seg_name in enumerate(["beginning", "middle", "end"]):
            seg_len = base + (1 if idx < remainder else 0)
            end = start + seg_len
            slices[seg_name] = (start, end, seg_len)
            start = end

        return slices

    def _split_segments(self, ts: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split time series into 3 segments.

        Args:
            ts: (N, L, F) time series array

        Returns:
            Dict mapping segment names to (N, seg_len, F) arrays
        """
        segments: Dict[str, np.ndarray] = {}
        for seg_name, (start, end, seg_len) in self.segment_slices.items():
            segments[seg_name] = ts[:, start:end, :]
        return segments

    def _load_classifiers(self) -> Dict[str, nn.Module]:
        """
        Load all 6 classifier checkpoints.

        Returns:
            Dict mapping model names to loaded models

        Raises:
            FileNotFoundError: If any checkpoint is missing
        """
        models: Dict[str, nn.Module] = {}

        for seg_name in ["beginning", "middle", "end"]:
            seg_len = self.segment_slices[seg_name][2]

            # Stage 1: Presence classifier
            presence_model = PresenceClassifier1D(seg_len)
            presence_path = self.checkpoint_dir / f"presence_{seg_name}_best.pth"

            if not presence_path.exists():
                raise FileNotFoundError(
                    f"Presence classifier checkpoint not found: {presence_path}\n"
                    f"Please train classifiers first using the segment classifier training script."
                )

            checkpoint = torch.load(presence_path, map_location=self.device)
            presence_model.load_state_dict(checkpoint)
            presence_model.eval()
            models[f"{seg_name}_presence"] = presence_model.to(self.device)

            # Stage 2: Parameter classifier
            params_model = ParameterClassifier1D(seg_len)
            params_path = self.checkpoint_dir / f"params_{seg_name}_best.pth"

            if not params_path.exists():
                raise FileNotFoundError(
                    f"Parameter classifier checkpoint not found: {params_path}\n"
                    f"Please train classifiers first using the segment classifier training script."
                )

            checkpoint = torch.load(params_path, map_location=self.device)
            params_model.load_state_dict(checkpoint)
            params_model.eval()
            models[f"{seg_name}_params"] = params_model.to(self.device)

        logger.info(f"[{self.name}] Loaded 6 classifier checkpoints")
        return models

    def _evaluate_segment(
        self,
        seg_name: str,
        all_pred: np.ndarray,
        all_ts: np.ndarray,
        all_attrs: np.ndarray,
        models: Dict[str, nn.Module],
    ) -> Dict[str, float]:
        """
        Evaluate a single segment.

        Args:
            seg_name: Segment name (beginning/middle/end)
            all_pred: (N, L, F) predictions
            all_ts: (N, L, F) ground truth
            all_attrs: (N, 17) attribute labels
            models: Loaded classifier models

        Returns:
            Dict with accuracy metrics and log probabilities
        """
        # Extract segment
        start, end, seg_len = self.segment_slices[seg_name]
        seg_pred = all_pred[:, start:end, :]  # (N, seg_len, F)

        # Get ground truth labels
        attr_idx = self.attr_indices[seg_name]
        presence_gt = all_attrs[:, attr_idx["presence"]]  # (N,)
        width_gt = all_attrs[:, attr_idx["width"]]
        amp_gt = all_attrs[:, attr_idx["amp"]]
        skew_gt = all_attrs[:, attr_idx["skew"]]

        # Convert to torch tensors for inference
        # Reshape to (N, 1, seg_len) for 1D CNN
        seg_pred_torch = torch.from_numpy(seg_pred).float().permute(0, 2, 1).to(self.device)

        # Stage 1: Presence classification
        presence_model = models[f"{seg_name}_presence"]
        with torch.no_grad():
            presence_logits = presence_model(seg_pred_torch)  # (N, 2)
            presence_preds = torch.argmax(presence_logits, dim=1).cpu().numpy()

        presence_acc = float((presence_preds == presence_gt).mean())

        # Stage 2: Parameter classification (only for present segments)
        present_mask = presence_gt == 1
        n_present = int(present_mask.sum())

        if n_present == 0:
            logger.warning(f"[{self.name}] No present segments for {seg_name}")
            return {
                "presence_acc": presence_acc,
                "width_acc": 0.0,
                "amp_acc": 0.0,
                "skew_acc": 0.0,
                "joint_acc": 0.0,
                "width_log_prob": 0.0,
                "amp_log_prob": 0.0,
                "skew_log_prob": 0.0,
                "n_present": 0,
            }

        # Extract present samples
        seg_pred_present = seg_pred_torch[present_mask]
        width_gt_present = width_gt[present_mask]
        amp_gt_present = amp_gt[present_mask]
        skew_gt_present = skew_gt[present_mask]

        params_model = models[f"{seg_name}_params"]
        with torch.no_grad():
            width_logits, amp_logits, skew_logits = params_model(seg_pred_present)

            width_preds = torch.argmax(width_logits, dim=1).cpu().numpy()
            amp_preds = torch.argmax(amp_logits, dim=1).cpu().numpy()
            skew_preds = torch.argmax(skew_logits, dim=1).cpu().numpy()

            # Compute accuracies
            width_acc = float((width_preds == width_gt_present).mean())
            amp_acc = float((amp_preds == amp_gt_present).mean())
            skew_acc = float((skew_preds == skew_gt_present).mean())

            # Joint accuracy: all three parameters correct
            joint_correct = (
                (width_preds == width_gt_present) &
                (amp_preds == amp_gt_present) &
                (skew_preds == skew_gt_present)
            )
            joint_acc = float(joint_correct.mean())

            # Compute log probabilities (soft metrics)
            width_log_probs = F.log_softmax(width_logits, dim=1)
            amp_log_probs = F.log_softmax(amp_logits, dim=1)
            skew_log_probs = F.log_softmax(skew_logits, dim=1)

            # Extract log prob of correct class
            width_gt_torch = torch.from_numpy(width_gt_present).long().to(self.device)
            amp_gt_torch = torch.from_numpy(amp_gt_present).long().to(self.device)
            skew_gt_torch = torch.from_numpy(skew_gt_present).long().to(self.device)

            width_log_prob = float(
                width_log_probs.gather(1, width_gt_torch.unsqueeze(1)).mean().item()
            )
            amp_log_prob = float(
                amp_log_probs.gather(1, amp_gt_torch.unsqueeze(1)).mean().item()
            )
            skew_log_prob = float(
                skew_log_probs.gather(1, skew_gt_torch.unsqueeze(1)).mean().item()
            )

        return {
            "presence_acc": presence_acc,
            "width_acc": width_acc,
            "amp_acc": amp_acc,
            "skew_acc": skew_acc,
            "joint_acc": joint_acc,
            "width_log_prob": width_log_prob,
            "amp_log_prob": amp_log_prob,
            "skew_log_prob": skew_log_prob,
            "n_present": n_present,
        }

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save detailed results to JSON."""
        self.save_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.save_dir / "segment_classifier_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"[{self.name}] Saved detailed results to {output_path}")


__all__ = [
    "PresenceClassifier1D",
    "ParameterClassifier1D",
    "SegmentParameterAccuracyMetric",
]
