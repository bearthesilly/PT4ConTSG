# -*- coding: utf-8 -*-
"""
RQ4: Compositional Generalization Evaluation

Tests whether PTFG can generalize to novel attribute combinations unseen
during training, using the Head-Tail split methodology from the ConTSG paper.

Methodology (aligned with ConTSG paper):
  1. For each test sample, compute k-NN average Hamming distance to training
     neighbors in attribute space
  2. Split test samples into Head (20% closest to training) and Tail (20%
     farthest from training)
  3. Compute CTTP retrieval accuracy on generated vs real samples
  4. Normalized retrieval accuracy: Acc_norm = Acc_gen / Acc_ref
  5. Compare Head vs Tail degradation across models

This script:
  - Loads test attributes and computes k-NN Hamming distance to training set
  - Splits into Head/Tail partitions
  - Uses pre-trained CTTP model to compute retrieval accuracy
  - Reports accuracy gap (Tail - Head) as measure of OOD sensitivity

Usage:
    python scripts/rq4_compositional_eval.py --eval \
        --experiments exp1_dir exp2_dir \
        --labels "PTFG-best" "PTFG-no-cross-talk" \
        --data-folder ./datasets/synth-m \
        --clip-config ./configs/cttp/cttp_synth-m.yaml \
        --clip-model ./checkpoints/cttp/clip_model_synth-m.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# k-NN Average Hamming Distance (aligned with ConTSG paper)
# ============================================================

def compute_knn_hamming_distances(test_attrs: np.ndarray,
                                  train_attrs: np.ndarray,
                                  k: int = 5) -> np.ndarray:
    """For each test sample, compute average Hamming distance to k nearest
    training neighbors in attribute space.

    Follows ConTSG paper:
        d_knn(c_test) = (1/k) * sum_{c in KNN_k(c_test)} HD(c_test, c)

    Args:
        test_attrs: (N_test, A) integer attribute indices
        train_attrs: (N_train, A) integer attribute indices
        k: number of nearest neighbors (default 5)

    Returns:
        (N_test,) average Hamming distance to k nearest training neighbors
    """
    n_test = test_attrs.shape[0]
    knn_dists = np.zeros(n_test, dtype=np.float64)

    # Chunk to avoid memory explosion
    chunk_size = 500
    for i in range(0, n_test, chunk_size):
        end = min(i + chunk_size, n_test)
        test_chunk = test_attrs[i:end]  # (chunk, A)
        # (chunk, 1, A) != (1, N_train, A) -> sum -> (chunk, N_train)
        dists = (test_chunk[:, None, :] != train_attrs[None, :, :]).sum(axis=2)
        # For each test sample, find k smallest distances and average
        k_actual = min(k, dists.shape[1])
        # np.partition is O(n) vs O(n log n) for full sort
        partitioned = np.partition(dists, k_actual - 1, axis=1)[:, :k_actual]
        knn_dists[i:end] = partitioned.mean(axis=1)

    return knn_dists


# ============================================================
# Head-Tail Split
# ============================================================

def head_tail_split(knn_dists: np.ndarray, head_ratio: float = 0.2,
                    tail_ratio: float = 0.2):
    """Split indices into Head (closest to train) and Tail (farthest).

    Returns:
        head_indices, tail_indices: arrays of sample indices
    """
    n = len(knn_dists)
    sorted_idx = np.argsort(knn_dists)

    n_head = max(1, int(n * head_ratio))
    n_tail = max(1, int(n * tail_ratio))

    head_indices = sorted_idx[:n_head]
    tail_indices = sorted_idx[-n_tail:]

    logger.info(f"  Head: {n_head} samples (avg Hamming dist <= {knn_dists[head_indices[-1]]:.2f})")
    logger.info(f"  Tail: {n_tail} samples (avg Hamming dist >= {knn_dists[tail_indices[0]]:.2f})")

    return head_indices, tail_indices


# ============================================================
# CTTP Retrieval Accuracy (aligned with ConTSG paper)
# ============================================================

def compute_retrieval_accuracy(ts_emb: torch.Tensor,
                               cap_emb: torch.Tensor,
                               batch_size: int = 256) -> float:
    """Compute top-1 retrieval accuracy using CTTP embeddings.

    For each time series embedding, find the most similar text embedding.
    Accuracy = fraction where the matched text is the correct one.

    To handle large N, we evaluate in batches where the candidate pool
    is the batch itself (consistent with ConTSG CTTP evaluation).

    Args:
        ts_emb: (N, D) time series embeddings
        cap_emb: (N, D) text caption embeddings
        batch_size: evaluation batch size

    Returns:
        top-1 retrieval accuracy (float)
    """
    N = ts_emb.shape[0]
    if N == 0:
        return 0.0

    total_correct = 0
    total_samples = 0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ts_batch = ts_emb[start:end]   # (B, D)
        cap_batch = cap_emb[start:end]  # (B, D)
        B = ts_batch.shape[0]

        # (B, D) @ (D, B) -> (B, B) similarity matrix
        sim = torch.mm(ts_batch, cap_batch.t())  # (B, B)
        # Top-1 retrieval: for each TS, which text is most similar?
        preds = sim.argmax(dim=-1)  # (B,)
        gt = torch.arange(B, device=preds.device)
        total_correct += (preds == gt).sum().item()
        total_samples += B

    return total_correct / total_samples


# ============================================================
# Evaluate Compositional Generalization
# ============================================================

def _to_np(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _get_pred(b):
    """Get median prediction from cache batch."""
    if "pred" in b:
        return _to_np(b["pred"])
    mp = b["multi_preds"]  # (n_samples, B, L, C)
    if hasattr(mp, "median"):
        return mp.median(dim=0).values.cpu().numpy()
    return np.median(mp, axis=0)


def evaluate_experiment(exp_dir: str, data_folder: str,
                        clip_config_path: str, clip_model_path: str,
                        k: int = 5, device: str = "cuda") -> dict:
    """Evaluate compositional generalization for one experiment.

    Uses CTTP retrieval accuracy (aligned with ConTSG paper).
    """
    exp_path = Path(exp_dir)
    data_path = Path(data_folder)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load training and test attributes for Hamming distance
    train_attrs = np.load(data_path / "train_attrs_idx.npy")
    test_attrs = np.load(data_path / "test_attrs_idx.npy")

    # Compute k-NN Hamming distances and Head-Tail split
    logger.info(f"Computing k-NN Hamming distances (k={k})...")
    knn_dists = compute_knn_hamming_distances(test_attrs, train_attrs, k=k)
    head_idx, tail_idx = head_tail_split(knn_dists)

    # Load predictions cache
    cache_file = exp_path / "results" / "predictions_cache.pkl"
    if not cache_file.exists():
        logger.error(f"predictions_cache.pkl not found in {exp_path / 'results'}")
        return {}

    with open(cache_file, "rb") as f:
        cache = pickle.load(f)

    batches = cache["all_batches_data"] if isinstance(cache, dict) and "all_batches_data" in cache else cache

    # Collect generated TS and real TS
    gen_ts = np.concatenate([_get_pred(b) for b in batches], axis=0)
    real_ts = np.concatenate([_to_np(b["ts"]) for b in batches], axis=0)

    # Check if CTTP embeddings are pre-cached
    has_cached_emb = all("cap_emb" in b and "ts_gen_emb" in b for b in batches)

    if has_cached_emb:
        logger.info("  Using pre-cached CTTP embeddings from predictions cache")
        gen_emb = torch.cat([b["ts_gen_emb"] for b in batches], dim=0).to(dev)
        ref_emb = torch.cat([b["ts_gt_emb"] for b in batches], dim=0).to(dev)
        cap_emb = torch.cat([b["cap_emb"] for b in batches], dim=0).to(dev)
    else:
        # Compute CTTP embeddings on the fly
        logger.info("  Computing CTTP embeddings on the fly...")
        from contsg.eval.embedder import CLIPEmbedder
        embedder = CLIPEmbedder(clip_config_path, clip_model_path, dev)

        gen_ts_t = torch.from_numpy(gen_ts.astype(np.float32)).to(dev)
        real_ts_t = torch.from_numpy(real_ts.astype(np.float32)).to(dev)

        # Embed in chunks to avoid OOM
        chunk = 256
        gen_emb_list, ref_emb_list, cap_emb_list = [], [], []
        for i in range(0, len(gen_ts_t), chunk):
            end = min(i + chunk, len(gen_ts_t))
            gen_emb_list.append(embedder.get_ts_embedding(gen_ts_t[i:end]))
            ref_emb_list.append(embedder.get_ts_embedding(real_ts_t[i:end]))
            # Reconstruct cap batch for text embedding
            cap_batch = {"cap": []}
            for b in batches:
                if "cap" in b:
                    if isinstance(b["cap"], list):
                        cap_batch["cap"].extend(b["cap"])
                    else:
                        cap_batch["cap"].append(b["cap"])
        # Re-collect all captions properly
        all_caps = []
        for b in batches:
            if "cap" in b:
                if isinstance(b["cap"], list):
                    all_caps.extend(b["cap"])
                else:
                    all_caps.append(b["cap"])

        cap_emb_list = []
        for i in range(0, len(all_caps), chunk):
            end = min(i + chunk, len(all_caps))
            cap_sub = {"cap": all_caps[i:end]}
            cap_emb_list.append(embedder.get_text_embedding(cap_sub))

        gen_emb = torch.cat(gen_emb_list, dim=0)
        ref_emb = torch.cat(ref_emb_list, dim=0)
        cap_emb = torch.cat(cap_emb_list, dim=0)

    # Normalize embeddings for retrieval
    gen_emb = F.normalize(gen_emb, dim=-1)
    ref_emb = F.normalize(ref_emb, dim=-1)
    cap_emb = F.normalize(cap_emb, dim=-1)

    # Evaluate on Head / Tail / All partitions
    results = {}
    for partition_name, indices in [("head", head_idx), ("tail", tail_idx),
                                     ("all", np.arange(len(test_attrs)))]:
        idx_t = torch.from_numpy(indices).long()

        gen_acc = compute_retrieval_accuracy(gen_emb[idx_t], cap_emb[idx_t])
        ref_acc = compute_retrieval_accuracy(ref_emb[idx_t], cap_emb[idx_t])
        acc_norm = gen_acc / max(ref_acc, 1e-8)

        results[partition_name] = {
            "gen_acc": gen_acc,
            "ref_acc": ref_acc,
            "acc_norm": acc_norm,
            "n_samples": len(indices),
        }
        logger.info(f"  {partition_name}: gen_acc={gen_acc:.4f}  "
                     f"ref_acc={ref_acc:.4f}  acc_norm={acc_norm:.4f}")

    # Head-Tail gap
    results["head_tail_gap"] = {
        "acc_norm_gap": results["tail"]["acc_norm"] - results["head"]["acc_norm"],
        "gen_acc_gap": results["tail"]["gen_acc"] - results["head"]["gen_acc"],
    }
    logger.info(f"  Head-Tail gap (acc_norm): "
                f"{results['head_tail_gap']['acc_norm_gap']:.4f}")

    # Hamming distance statistics
    results["hamming_stats"] = {
        "k": k,
        "mean": float(knn_dists.mean()),
        "std": float(knn_dists.std()),
        "head_mean": float(knn_dists[head_idx].mean()),
        "tail_mean": float(knn_dists[tail_idx].mean()),
    }

    return results


# ============================================================
# Comparative Report
# ============================================================

def print_comparison(all_results: dict):
    """Pretty-print comparison table."""
    print("\n" + "=" * 92)
    print("RQ4: Compositional Generalization -- Head-Tail Analysis (CTTP Retrieval)")
    print("=" * 92)

    header = "{:<25s} {:>10s} {:>10s} {:>10s} {:>10s} {:>12s}".format(
        "Model", "Head Acc", "Tail Acc", "Head Norm", "Tail Norm", "Gap (Norm)")
    print(header)
    print("-" * 92)

    for label, res in all_results.items():
        if not res or "head" not in res:
            continue
        print("{:<25s} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>12.4f}".format(
            label,
            res["head"]["gen_acc"],
            res["tail"]["gen_acc"],
            res["head"]["acc_norm"],
            res["tail"]["acc_norm"],
            res["head_tail_gap"]["acc_norm_gap"],
        ))

    print("=" * 92)
    print("Retrieval: CTTP top-1 retrieval accuracy (TS -> Text matching)")
    print("Acc_norm = gen_acc / ref_acc (higher = better condition following)")
    print("Gap = Tail_norm - Head_norm (less negative = better compositional generalization)")
    print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RQ4: Compositional generalization eval")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate experiments")
    parser.add_argument("--data-folder", type=str, default="./datasets/synth-m")
    parser.add_argument("--clip-config", type=str,
                        default="./configs/cttp/cttp_synth-m.yaml",
                        help="Path to CTTP config YAML")
    parser.add_argument("--clip-model", type=str,
                        default="./checkpoints/cttp/clip_model_synth-m.pth",
                        help="Path to CTTP model checkpoint")
    parser.add_argument("--experiments", nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--k", type=int, default=5,
                        help="k for k-NN Hamming distance (default: 5)")
    parser.add_argument("--head-ratio", type=float, default=0.2)
    parser.add_argument("--tail-ratio", type=float, default=0.2)
    parser.add_argument("--output", type=str, default="./results_rq4.json")
    args = parser.parse_args()

    if args.eval:
        all_results = {}
        for i, exp_dir in enumerate(args.experiments):
            label = args.labels[i] if i < len(args.labels) else Path(exp_dir).name
            logger.info(f"\nEvaluating: {label} ({exp_dir})")
            res = evaluate_experiment(
                exp_dir, args.data_folder,
                args.clip_config, args.clip_model,
                k=args.k, device=args.device,
            )
            all_results[label] = res

        print_comparison(all_results)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
