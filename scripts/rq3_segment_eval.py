# -*- coding: utf-8 -*-
"""
RQ3: Fine-grained Segment-level Control Evaluation

Evaluates whether PTFG can follow fine-grained segment-level specifications
by comparing segment classification accuracy against baseline models.

This script:
  1. Loads generated samples from experiment checkpoints (via predictions_cache.pkl
     or by regenerating from checkpoint)
  2. Trains segment classifiers on REAL data if not already trained
  3. Evaluates segment-level classification accuracy on generated samples
  4. Compares PTFG (with/without patch_cond_modulate) against baselines
  5. Reports joint accuracy per segment (all 3 params correct)

Usage:
    # Step 1: Train segment classifiers (only needed once)
    python scripts/rq3_segment_eval.py --train-classifiers \
        --data-folder ./datasets/synth-u

    # Step 2: Evaluate experiments
    python scripts/rq3_segment_eval.py --eval \
        --experiments exp1_dir exp2_dir exp3_dir \
        --labels "PTFG-best" "PTFG-no-patch-mod" "VerbalTS" \
        --data-folder ./datasets/synth-u

    # Step 3: Compare against retrieval baseline
    python scripts/rq3_segment_eval.py --eval --retrieval-baseline \
        --data-folder ./datasets/synth-u
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Classifier Architectures (same as contsg/eval/metrics/segment.py)
# ============================================================

class PresenceClassifier1D(nn.Module):
    def __init__(self, segment_len: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x).squeeze(-1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class ParameterClassifier1D(nn.Module):
    def __init__(self, segment_len: int):
        super().__init__()
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
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc_shared = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc_width = nn.Linear(128, 3)
        self.fc_amp = nn.Linear(128, 3)
        self.fc_skew = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.gap(x).squeeze(-1)
        x = self.dropout(F.relu(self.fc_shared(x)))
        return self.fc_width(x), self.fc_amp(x), self.fc_skew(x)


# ============================================================
# Segment Attribute Mapping (synth-u: 17 attributes)
# ============================================================

SEGMENT_NAMES = ["beginning", "middle", "end"]
ATTR_INDICES = {
    "beginning": {"presence": 5, "width": 8, "amp": 9, "skew": 10},
    "middle":    {"presence": 6, "width": 11, "amp": 12, "skew": 13},
    "end":       {"presence": 7, "width": 14, "amp": 15, "skew": 16},
}


def compute_segment_slices(seq_len: int = 128, n_segments: int = 3):
    """Compute segment boundaries."""
    base = seq_len // n_segments
    remainder = seq_len % n_segments
    slices = {}
    start = 0
    for idx, name in enumerate(SEGMENT_NAMES):
        seg_len = base + (1 if idx < remainder else 0)
        slices[name] = (start, start + seg_len, seg_len)
        start += seg_len
    return slices


# ============================================================
# Step 1: Train Segment Classifiers on REAL data
# ============================================================

def train_classifiers(data_folder: str, output_dir: str = "./segment_classifiers",
                      epochs: int = 100, lr: float = 1e-3, batch_size: int = 64,
                      device: str = "cuda"):
    """Train presence and parameter classifiers on real training data."""
    data_folder = Path(data_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load real training data
    train_ts = np.load(data_folder / "train_ts.npy").astype(np.float32)   # (N, L, C)
    train_attrs = np.load(data_folder / "train_attrs_idx.npy")            # (N, 17)

    slices = compute_segment_slices(train_ts.shape[1])

    for seg_name in SEGMENT_NAMES:
        start, end, seg_len = slices[seg_name]
        seg_data = train_ts[:, start:end, 0:1]  # (N, seg_len, 1) — univariate
        seg_data_t = torch.from_numpy(seg_data).permute(0, 2, 1)  # (N, 1, seg_len)

        attr_idx = ATTR_INDICES[seg_name]
        presence_labels = torch.from_numpy(train_attrs[:, attr_idx["presence"]]).long()

        # --- Train Presence Classifier ---
        logger.info(f"Training presence classifier for [{seg_name}] (seg_len={seg_len})...")
        presence_model = PresenceClassifier1D(seg_len).to(dev)
        _train_classifier(
            presence_model, seg_data_t, presence_labels,
            epochs=epochs, lr=lr, batch_size=batch_size, device=dev
        )
        torch.save(presence_model.state_dict(),
                    output_dir / f"presence_{seg_name}_best.pth")
        logger.info(f"  Saved: {output_dir / f'presence_{seg_name}_best.pth'}")

        # --- Train Parameter Classifier (only on present samples) ---
        present_mask = train_attrs[:, attr_idx["presence"]] == 1
        if present_mask.sum() < 10:
            logger.warning(f"  Too few present samples for [{seg_name}], skipping params")
            continue

        seg_present = seg_data_t[present_mask]
        width_labels = torch.from_numpy(train_attrs[present_mask, attr_idx["width"]]).long()
        amp_labels = torch.from_numpy(train_attrs[present_mask, attr_idx["amp"]]).long()
        skew_labels = torch.from_numpy(train_attrs[present_mask, attr_idx["skew"]]).long()

        logger.info(f"Training parameter classifier for [{seg_name}] "
                     f"({present_mask.sum()} present samples)...")
        params_model = ParameterClassifier1D(seg_len).to(dev)
        _train_param_classifier(
            params_model, seg_present, width_labels, amp_labels, skew_labels,
            epochs=epochs, lr=lr, batch_size=batch_size, device=dev
        )
        torch.save(params_model.state_dict(),
                    output_dir / f"params_{seg_name}_best.pth")
        logger.info(f"  Saved: {output_dir / f'params_{seg_name}_best.pth'}")

    logger.info("All segment classifiers trained.")


def _train_classifier(model, data, labels, epochs, lr, batch_size, device):
    """Generic binary/multiclass classifier training loop."""
    dataset = TensorDataset(data.to(device), labels.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"    Epoch {epoch+1}/{epochs}  "
                         f"loss={total_loss/total:.4f}  acc={correct/total:.4f}")


def _train_param_classifier(model, data, w_labels, a_labels, s_labels,
                            epochs, lr, batch_size, device):
    """Multi-task parameter classifier training."""
    dataset = TensorDataset(data.to(device), w_labels.to(device),
                            a_labels.to(device), s_labels.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss, total = 0.0, 0
        for x, w, a, s in loader:
            w_logits, a_logits, s_logits = model(x)
            loss = (F.cross_entropy(w_logits, w) +
                    F.cross_entropy(a_logits, a) +
                    F.cross_entropy(s_logits, s))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"    Epoch {epoch+1}/{epochs}  loss={total_loss/total:.4f}")


# ============================================================
# Step 2: Evaluate Generated Samples
# ============================================================

def evaluate_experiment(exp_dir: str, data_folder: str,
                        classifier_dir: str = "./segment_classifiers",
                        device: str = "cuda") -> dict:
    """Evaluate segment accuracy on generated samples from one experiment."""
    exp_path = Path(exp_dir)
    data_path = Path(data_folder)
    clf_path = Path(classifier_dir)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load predictions cache
    cache_file = exp_path / "results" / "predictions_cache.pkl"
    if not cache_file.exists():
        logger.error(f"predictions_cache.pkl not found in {exp_path / 'results'}")
        logger.info("Run: contsg evaluate <exp_dir> --use-cache  first.")
        return {}

    with open(cache_file, "rb") as f:
        cache = pickle.load(f)

    # Cache format: dict with "all_batches_data" key containing list of batch dicts
    batches = cache["all_batches_data"] if isinstance(cache, dict) and "all_batches_data" in cache else cache

    # Reconstruct arrays from cache batches (may contain torch Tensors)
    def _to_np(x):
        if hasattr(x, "cpu"):
            return x.cpu().numpy()
        return np.asarray(x)

    def _get_pred(b):
        """Get median prediction: cache stores multi_preds, not pred."""
        if "pred" in b:
            return _to_np(b["pred"])
        mp = b["multi_preds"]  # (n_samples, B, L, C)
        if hasattr(mp, "median"):
            return mp.median(dim=0).values.cpu().numpy()
        return np.median(mp, axis=0)

    all_pred = np.concatenate([_get_pred(b) for b in batches], axis=0)   # (N, L, C)
    all_attrs = np.concatenate([_to_np(b["attrs_idx"]) for b in batches], axis=0)  # (N, 17)

    logger.info(f"Loaded {all_pred.shape[0]} generated samples from {exp_dir}")

    slices = compute_segment_slices(all_pred.shape[1])
    results = {}

    for seg_name in SEGMENT_NAMES:
        start, end, seg_len = slices[seg_name]
        seg_pred = all_pred[:, start:end, 0:1]  # (N, seg_len, 1)
        seg_pred_t = torch.from_numpy(seg_pred.astype(np.float32)).permute(0, 2, 1).to(dev)

        attr_idx = ATTR_INDICES[seg_name]
        presence_gt = all_attrs[:, attr_idx["presence"]]

        # Load presence classifier
        presence_model = PresenceClassifier1D(seg_len).to(dev)
        presence_model.load_state_dict(
            torch.load(clf_path / f"presence_{seg_name}_best.pth", map_location=dev))
        presence_model.eval()

        with torch.no_grad():
            presence_preds = presence_model(seg_pred_t).argmax(1).cpu().numpy()
        presence_acc = float((presence_preds == presence_gt).mean())

        # Load parameter classifier
        params_model = ParameterClassifier1D(seg_len).to(dev)
        params_model.load_state_dict(
            torch.load(clf_path / f"params_{seg_name}_best.pth", map_location=dev))
        params_model.eval()

        present_mask = presence_gt == 1
        n_present = int(present_mask.sum())

        if n_present == 0:
            results[seg_name] = {"presence_acc": presence_acc, "joint_acc": 0.0,
                                 "width_acc": 0.0, "amp_acc": 0.0, "skew_acc": 0.0}
            continue

        seg_present = seg_pred_t[present_mask]
        w_gt = all_attrs[present_mask, attr_idx["width"]]
        a_gt = all_attrs[present_mask, attr_idx["amp"]]
        s_gt = all_attrs[present_mask, attr_idx["skew"]]

        with torch.no_grad():
            w_logits, a_logits, s_logits = params_model(seg_present)
            w_pred = w_logits.argmax(1).cpu().numpy()
            a_pred = a_logits.argmax(1).cpu().numpy()
            s_pred = s_logits.argmax(1).cpu().numpy()

        w_acc = float((w_pred == w_gt).mean())
        a_acc = float((a_pred == a_gt).mean())
        s_acc = float((s_pred == s_gt).mean())
        joint_acc = float(((w_pred == w_gt) & (a_pred == a_gt) & (s_pred == s_gt)).mean())

        results[seg_name] = {
            "presence_acc": presence_acc,
            "width_acc": w_acc, "amp_acc": a_acc, "skew_acc": s_acc,
            "joint_acc": joint_acc, "n_present": n_present,
        }

    # Average
    joint_accs = [results[s]["joint_acc"] for s in SEGMENT_NAMES]
    results["average_joint_acc"] = float(np.mean(joint_accs))

    return results


def evaluate_retrieval_baseline(data_folder: str, device: str = "cuda") -> dict:
    """Evaluate the retrieval baseline: for each test condition, return the
    nearest training sample. This is the trivial baseline from the ConTSG paper."""
    data_path = Path(data_folder)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    train_ts = np.load(data_path / "train_ts.npy").astype(np.float32)
    train_emb = np.load(data_path / "train_cap_emb.npy").astype(np.float32)
    test_ts = np.load(data_path / "test_ts.npy").astype(np.float32)
    test_emb = np.load(data_path / "test_cap_emb.npy").astype(np.float32)
    test_attrs = np.load(data_path / "test_attrs_idx.npy")

    # For each test sample, find nearest training sample by embedding cosine sim
    train_emb_norm = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-8)
    test_emb_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-8)
    sim = test_emb_norm @ train_emb_norm.T  # (N_test, N_train)
    nearest_idx = sim.argmax(axis=1)
    retrieved_ts = train_ts[nearest_idx]  # (N_test, L, C)

    # Now evaluate segment accuracy on retrieved samples
    # (reuse the same evaluation logic)
    logger.info(f"Retrieval baseline: {len(test_ts)} test samples")

    clf_path = Path("./segment_classifiers")
    slices = compute_segment_slices(retrieved_ts.shape[1])
    results = {}

    for seg_name in SEGMENT_NAMES:
        start, end, seg_len = slices[seg_name]
        seg_data = retrieved_ts[:, start:end, 0:1]
        seg_data_t = torch.from_numpy(seg_data).permute(0, 2, 1).to(dev)

        attr_idx = ATTR_INDICES[seg_name]
        presence_gt = test_attrs[:, attr_idx["presence"]]

        presence_model = PresenceClassifier1D(seg_len).to(dev)
        presence_model.load_state_dict(
            torch.load(clf_path / f"presence_{seg_name}_best.pth", map_location=dev))
        presence_model.eval()

        with torch.no_grad():
            presence_preds = presence_model(seg_data_t).argmax(1).cpu().numpy()
        presence_acc = float((presence_preds == presence_gt).mean())

        present_mask = presence_gt == 1
        n_present = int(present_mask.sum())
        if n_present == 0:
            results[seg_name] = {"presence_acc": presence_acc, "joint_acc": 0.0}
            continue

        params_model = ParameterClassifier1D(seg_len).to(dev)
        params_model.load_state_dict(
            torch.load(clf_path / f"params_{seg_name}_best.pth", map_location=dev))
        params_model.eval()

        seg_present = seg_data_t[present_mask]
        w_gt = test_attrs[present_mask, attr_idx["width"]]
        a_gt = test_attrs[present_mask, attr_idx["amp"]]
        s_gt = test_attrs[present_mask, attr_idx["skew"]]

        with torch.no_grad():
            w_l, a_l, s_l = params_model(seg_present)
            w_p = w_l.argmax(1).cpu().numpy()
            a_p = a_l.argmax(1).cpu().numpy()
            s_p = s_l.argmax(1).cpu().numpy()

        joint_acc = float(((w_p == w_gt) & (a_p == a_gt) & (s_p == s_gt)).mean())
        results[seg_name] = {
            "presence_acc": presence_acc, "joint_acc": joint_acc,
            "n_present": n_present,
        }

    joint_accs = [results[s]["joint_acc"] for s in SEGMENT_NAMES]
    results["average_joint_acc"] = float(np.mean(joint_accs))
    return results


# ============================================================
# Step 3: Comparative Report
# ============================================================

def print_comparison(all_results: dict):
    """Pretty-print comparison table."""
    print("\n" + "=" * 80)
    print("RQ3: Fine-grained Segment-level Control — Comparison")
    print("=" * 80)

    header = "{:<25s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "Model", "Seg-Begin", "Seg-Mid", "Seg-End", "Avg Joint")
    print(header)
    print("-" * 80)

    for label, res in all_results.items():
        if not res:
            continue
        vals = []
        for seg_name in SEGMENT_NAMES:
            if seg_name in res:
                vals.append(res[seg_name].get("joint_acc", float("nan")))
            else:
                vals.append(float("nan"))
        avg = res.get("average_joint_acc", float("nan"))
        print("{:<25s} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            label, *vals, avg))

    print("=" * 80)
    print("Higher is better. Joint acc = all 3 params (width, amp, skew) correct.")
    print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RQ3: Segment-level control eval")
    parser.add_argument("--train-classifiers", action="store_true",
                        help="Train segment classifiers on real data")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate experiments")
    parser.add_argument("--retrieval-baseline", action="store_true",
                        help="Include retrieval baseline")
    parser.add_argument("--data-folder", type=str, default="./datasets/synth-u",
                        help="Path to synth-u dataset")
    parser.add_argument("--classifier-dir", type=str, default="./segment_classifiers")
    parser.add_argument("--experiments", nargs="*", default=[],
                        help="Experiment directories to evaluate")
    parser.add_argument("--labels", nargs="*", default=[],
                        help="Labels for each experiment (same order)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Epochs for classifier training")
    parser.add_argument("--output", type=str, default="./results_rq3.json",
                        help="Save results JSON")
    args = parser.parse_args()

    if args.train_classifiers:
        train_classifiers(args.data_folder, args.classifier_dir,
                          epochs=args.epochs, device=args.device)

    if args.eval:
        all_results = {}

        # Evaluate each experiment
        for i, exp_dir in enumerate(args.experiments):
            label = args.labels[i] if i < len(args.labels) else Path(exp_dir).name
            logger.info(f"\nEvaluating: {label} ({exp_dir})")
            res = evaluate_experiment(exp_dir, args.data_folder,
                                       args.classifier_dir, args.device)
            all_results[label] = res

        # Retrieval baseline
        if args.retrieval_baseline:
            logger.info("\nEvaluating: Retrieval Baseline")
            res = evaluate_retrieval_baseline(args.data_folder, args.device)
            all_results["Retrieval-Baseline"] = res

        # Print comparison
        print_comparison(all_results)

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
