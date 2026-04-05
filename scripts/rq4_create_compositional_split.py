# -*- coding: utf-8 -*-
"""
RQ4 Data Preparation: Create a compositional split of synth-m.

Holds out ~25% of attribute combinations for testing. The remaining
~75% are used for training. This guarantees that the test set contains
novel attribute combinations unseen during training, enabling
meaningful Head-Tail analysis for compositional generalization.

Produces a new dataset folder: datasets/synth-m-compo/
with the same file structure as synth-m but with the new split.

Usage:
    python scripts/rq4_create_compositional_split.py \
        --src ./datasets/synth-m \
        --dst ./datasets/synth-m-compo \
        --holdout-ratio 0.25 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def combo_key(attrs_row: np.ndarray) -> tuple:
    return tuple(attrs_row.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="./datasets/synth-m",
                        help="Source synth-m dataset folder")
    parser.add_argument("--dst", type=str, default="./datasets/synth-m-compo",
                        help="Output dataset folder for compositional split")
    parser.add_argument("--holdout-ratio", type=float, default=0.25,
                        help="Fraction of unique combinations to hold out for test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of training combos' samples for validation")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    src = Path(args.src)
    dst = Path(args.dst)

    # ---- Load ALL data (merge original train+valid+test) ----
    all_ts, all_caps, all_cap_emb, all_attrs = [], [], [], []

    for split in ["train", "valid", "test"]:
        ts_path = src / f"{split}_ts.npy"
        if not ts_path.exists():
            print(f"  [skip] {ts_path} not found")
            continue

        ts = np.load(ts_path)
        all_ts.append(ts)
        print(f"  Loaded {split}_ts: {ts.shape}")

        caps_path = src / f"{split}_caps.npy"
        if caps_path.exists():
            all_caps.append(np.load(caps_path, allow_pickle=True))

        cap_emb_path = src / f"{split}_cap_emb.npy"
        if cap_emb_path.exists():
            all_cap_emb.append(np.load(cap_emb_path))

        attrs_path = src / f"{split}_attrs_idx.npy"
        if attrs_path.exists():
            all_attrs.append(np.load(attrs_path))

    all_ts = np.concatenate(all_ts, axis=0)
    all_attrs = np.concatenate(all_attrs, axis=0)
    has_caps = len(all_caps) > 0
    has_cap_emb = len(all_cap_emb) > 0
    if has_caps:
        all_caps = np.concatenate(all_caps, axis=0)
    if has_cap_emb:
        all_cap_emb = np.concatenate(all_cap_emb, axis=0)

    N = len(all_ts)
    A = all_attrs.shape[1]
    print(f"\nTotal samples: {N}, Attributes: {A}")

    # ---- Enumerate unique attribute combinations ----
    combo_to_indices = {}
    for i in range(N):
        key = combo_key(all_attrs[i])
        combo_to_indices.setdefault(key, []).append(i)

    unique_combos = sorted(combo_to_indices.keys())
    n_combos = len(unique_combos)
    print(f"Unique attribute combinations: {n_combos}")

    # ---- Stratified holdout: ensure test combos span various Hamming distances ----
    # We hold out ~25% of combos, trying to get a range of distances
    n_holdout = max(1, int(n_combos * args.holdout_ratio))
    n_train_combos = n_combos - n_holdout

    # Shuffle and split
    perm = rng.permutation(n_combos)
    train_combo_idx = perm[:n_train_combos]
    test_combo_idx = perm[n_train_combos:]

    train_combos = set(unique_combos[i] for i in train_combo_idx)
    test_combos = set(unique_combos[i] for i in test_combo_idx)

    print(f"Training combos: {len(train_combos)}")
    print(f"Test (held-out) combos: {len(test_combos)}")

    # ---- Verify Hamming distances of test combos to train combos ----
    train_combos_arr = np.array(sorted(train_combos))  # (n_train_combos, A)
    test_combos_arr = np.array(sorted(test_combos))    # (n_test_combos, A)

    # Min Hamming distance for each test combo to nearest train combo
    min_dists = []
    for tc in test_combos_arr:
        dists = (tc[None, :] != train_combos_arr).sum(axis=1)
        min_dists.append(dists.min())
    min_dists = np.array(min_dists)

    print(f"\nHeld-out combos Hamming distance to training (min per combo):")
    for d in range(A + 1):
        count = (min_dists == d).sum()
        if count > 0:
            print(f"  dist={d}: {count} combos")

    if (min_dists == 0).any():
        print("\n  WARNING: Some held-out combos have dist=0 (exist in training).")
        print("  This should not happen. Check for duplicates.")

    # ---- Collect sample indices ----
    train_sample_idx = []
    for combo in sorted(train_combos):
        train_sample_idx.extend(combo_to_indices[combo])

    test_sample_idx = []
    for combo in sorted(test_combos):
        test_sample_idx.extend(combo_to_indices[combo])

    # Shuffle within splits
    rng.shuffle(train_sample_idx)
    rng.shuffle(test_sample_idx)

    # Split off validation from training samples
    n_val = max(1, int(len(train_sample_idx) * args.val_ratio))
    val_sample_idx = train_sample_idx[:n_val]
    train_sample_idx = train_sample_idx[n_val:]

    print(f"\nFinal split:")
    print(f"  Train: {len(train_sample_idx)} samples ({len(train_combos)} combos)")
    print(f"  Valid: {len(val_sample_idx)} samples (subset of train combos)")
    print(f"  Test:  {len(test_sample_idx)} samples ({len(test_combos)} held-out combos)")

    # ---- Write new dataset ----
    dst.mkdir(parents=True, exist_ok=True)

    def save_split(name, indices):
        indices = np.array(indices)
        np.save(dst / f"{name}_ts.npy", all_ts[indices])
        np.save(dst / f"{name}_attrs_idx.npy", all_attrs[indices])
        if has_caps:
            np.save(dst / f"{name}_caps.npy", all_caps[indices])
        if has_cap_emb:
            np.save(dst / f"{name}_cap_emb.npy", all_cap_emb[indices])

    save_split("train", train_sample_idx)
    save_split("valid", val_sample_idx)
    save_split("test", test_sample_idx)

    # Copy meta.json if exists, add split info
    meta_path = src / "meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    meta["compositional_split"] = {
        "seed": args.seed,
        "holdout_ratio": args.holdout_ratio,
        "n_train_combos": len(train_combos),
        "n_test_combos": len(test_combos),
        "n_train_samples": len(train_sample_idx),
        "n_val_samples": len(val_sample_idx),
        "n_test_samples": len(test_sample_idx),
        "test_hamming_dist_distribution": {
            str(d): int((min_dists == d).sum()) for d in range(A + 1)
        },
    }
    with open(dst / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save train combo list for reference (used by evaluation script)
    np.save(dst / "train_combos.npy", train_combos_arr)

    print(f"\nDataset saved to: {dst}")
    print(f"Meta saved to: {dst / 'meta.json'}")


if __name__ == "__main__":
    main()
