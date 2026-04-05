# -*- coding: utf-8 -*-
"""
RQ4 Data Preparation: Create a structured compositional split of synth-m.

Instead of random holdout (which produces only dist=1 in a dense 128-combo
space), we use a STRUCTURED holdout strategy:

  - Define "novel values" for 3 of 4 attributes: attr0=3, attr2=3, attr3=3
  - Training set: combos where NONE of these novel values appear
    → 3*2*3*3 = 54 combos (42% of 128)
  - Test set: combos where AT LEAST ONE novel value appears
    → 74 combos (58%)

This creates a natural Hamming distance gradient:
  - 1 novel attr → dist=1 from training (54 combos) → Head
  - 2 novel attrs → dist=2 from training (18 combos) → Tail
  - 3 novel attrs → dist=3 from training (2 combos)  → Tail

Produces: datasets/synth-m-compo/

Usage:
    python scripts/rq4_create_compositional_split.py \
        --src ./datasets/synth-m \
        --dst ./datasets/synth-m-compo \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# Novel attribute values: these values will NOT appear in training
# synth-m attributes: trend(4), volatility(2), periodicity(4), shape(4)
# We pick the last class of 3 attributes as "novel"
NOVEL_VALUES = {
    0: 3,   # trend type: class 3 is novel
    2: 3,   # periodicity: class 3 is novel
    3: 3,   # shape: class 3 is novel
}
# Note: attr1 (volatility) only has 2 classes, not used as novel dimension


def count_novel(attrs_row: np.ndarray) -> int:
    """Count how many novel attribute values are present in a sample."""
    count = 0
    for dim, novel_val in NOVEL_VALUES.items():
        if attrs_row[dim] == novel_val:
            count += 1
    return count


def combo_key(attrs_row: np.ndarray) -> tuple:
    return tuple(attrs_row.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="./datasets/synth-m",
                        help="Source synth-m dataset folder")
    parser.add_argument("--dst", type=str, default="./datasets/synth-m-compo",
                        help="Output dataset folder for compositional split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of training samples for validation")
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
    print(f"Attribute ranges: {[len(np.unique(all_attrs[:, i])) for i in range(A)]}")

    # ---- Classify each sample by novelty level ----
    novelty_levels = np.array([count_novel(all_attrs[i]) for i in range(N)])

    print(f"\nNovel values: {NOVEL_VALUES}")
    print(f"Samples by novelty level:")
    for lv in range(4):
        n = (novelty_levels == lv).sum()
        if n > 0:
            print(f"  level {lv} (Hamming dist={lv}): {n} samples")

    # ---- Split: train = novelty 0, test = novelty >= 1 ----
    train_mask = novelty_levels == 0
    test_mask = novelty_levels >= 1

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    # Shuffle
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    # Split off validation from training
    n_val = max(1, int(len(train_indices) * args.val_ratio))
    val_indices = train_indices[:n_val]
    train_indices = train_indices[n_val:]

    # ---- Compute combo statistics ----
    train_combos = set(combo_key(all_attrs[i]) for i in train_indices)
    val_combos = set(combo_key(all_attrs[i]) for i in val_indices)
    test_combos = set(combo_key(all_attrs[i]) for i in test_indices)

    print(f"\nFinal split:")
    print(f"  Train: {len(train_indices)} samples ({len(train_combos)} unique combos)")
    print(f"  Valid: {len(val_indices)} samples ({len(val_combos)} unique combos)")
    print(f"  Test:  {len(test_indices)} samples ({len(test_combos)} unique combos)")

    # ---- Verify Hamming distances ----
    train_attrs_arr = all_attrs[np.concatenate([train_indices, val_indices])]
    test_attrs_arr = all_attrs[test_indices]

    # For each test sample, compute min Hamming distance to training
    print(f"\nTest set Hamming distance distribution to training:")
    test_novelty = novelty_levels[test_indices]
    for lv in sorted(np.unique(test_novelty)):
        n = (test_novelty == lv).sum()
        print(f"  dist={lv}: {n} samples")

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
        print(f"  Saved {name}: {len(indices)} samples")

    save_split("train", train_indices)
    save_split("valid", val_indices)
    save_split("test", test_indices)

    # ---- Save metadata ----
    meta_path = src / "meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    meta["compositional_split"] = {
        "strategy": "structured_novel_values",
        "novel_values": {str(k): int(v) for k, v in NOVEL_VALUES.items()},
        "seed": args.seed,
        "n_train_combos": len(train_combos),
        "n_test_combos": len(test_combos),
        "n_train_samples": len(train_indices),
        "n_val_samples": len(val_indices),
        "n_test_samples": len(test_indices),
        "test_distance_distribution": {
            str(lv): int((test_novelty == lv).sum())
            for lv in sorted(np.unique(test_novelty))
        },
    }
    with open(dst / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDataset saved to: {dst}")
    print(f"Meta saved to: {dst / 'meta.json'}")


if __name__ == "__main__":
    main()
