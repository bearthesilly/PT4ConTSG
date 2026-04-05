# -*- coding: utf-8 -*-
"""
RQ4 Data Preparation: Create a structured compositional split of synth-m.

Strategy:
  1. Copy the ENTIRE synth-m folder to synth-m-compo (preserving all files)
  2. Re-split: hold out attribute combos with "novel values" for testing
  3. Overwrite train/valid/test npy files with the new split

This preserves ALL auxiliary files (cap_emb, caps, meta.json, etc.) that
the evaluation pipeline needs, avoiding missing-file errors.

Novel values: attr0=3 (trend), attr2=3 (periodicity), attr3=3 (shape)
  - Training: combos where NONE of these appear (54 combos, ~42%)
  - Test: combos with AT LEAST ONE novel value (74 combos, ~58%)
    - dist=1: 54 combos, dist=2: 18 combos, dist=3: 2 combos

Usage:
    python scripts/rq4_create_compositional_split.py \
        --src ./datasets/synth-m \
        --dst ./datasets/synth-m-compo \
        --seed 42
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np


# Novel attribute values: these will NOT appear in training
# synth-m: trend(4), volatility(2), periodicity(4), shape(4)
NOVEL_VALUES = {
    0: 3,   # trend: class 3 is novel
    2: 3,   # periodicity: class 3 is novel
    3: 3,   # shape: class 3 is novel
}


def count_novel(attrs_row: np.ndarray) -> int:
    """Count how many novel attribute values are present."""
    return sum(1 for dim, val in NOVEL_VALUES.items() if attrs_row[dim] == val)


def combo_key(attrs_row: np.ndarray) -> tuple:
    return tuple(attrs_row.tolist())


def find_split_npy_files(folder: Path, split: str) -> dict:
    """Find all npy files for a given split, return {suffix: path} dict.

    E.g. for split='train': {'ts.npy': Path(...), 'caps.npy': Path(...), ...}
    Also matches benchmark-style names like train_text_caps_embeddings_1024.npy.
    """
    files = {}
    for p in sorted(folder.glob(f"{split}_*.npy")):
        suffix = p.name[len(split) + 1:]  # strip "{split}_"
        files[suffix] = p
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="./datasets/synth-m")
    parser.add_argument("--dst", type=str, default="./datasets/synth-m-compo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    src = Path(args.src)
    dst = Path(args.dst)

    # ================================================================
    # Step 0: Copy entire source folder
    # ================================================================
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"Copied {src} -> {dst}")
    print(f"Files preserved:")
    for f in sorted(dst.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")
    print()

    # ================================================================
    # Step 1: Load all samples (merge train+valid+test)
    # ================================================================
    # Discover all per-split npy files so we can re-split ALL of them
    all_split_files = {}  # suffix -> list of arrays (one per original split)
    all_suffixes = set()
    total_samples = 0

    for split in ["train", "valid", "test"]:
        npy_files = find_split_npy_files(dst, split)
        if "ts.npy" not in npy_files:
            print(f"  [skip] {split}_ts.npy not found")
            continue

        n = np.load(npy_files["ts.npy"]).shape[0]
        print(f"  {split}: {n} samples, files: {list(npy_files.keys())}")
        total_samples += n

        for suffix, path in npy_files.items():
            all_suffixes.add(suffix)
            data = np.load(path, allow_pickle=True)
            all_split_files.setdefault(suffix, []).append(data)

    # Concatenate all splits — only keep files present in ALL splits
    n_splits_loaded = sum(1 for s in ["train", "valid", "test"]
                         if (dst / f"{s}_ts.npy").exists())
    merged = {}
    for suffix in sorted(all_suffixes):
        parts = all_split_files[suffix]
        if len(parts) != n_splits_loaded:
            total_len = sum(p.shape[0] for p in parts)
            print(f"  [skip] {suffix}: only in {len(parts)}/{n_splits_loaded} splits "
                  f"({total_len} samples, expected {total_samples})")
            continue
        merged[suffix] = np.concatenate(parts, axis=0)

    all_ts = merged["ts.npy"]
    all_attrs = merged["attrs_idx.npy"]
    N = len(all_ts)
    A = all_attrs.shape[1]

    print(f"\nTotal: {N} samples, {A} attributes")
    print(f"Attribute ranges: {[len(np.unique(all_attrs[:, i])) for i in range(A)]}")
    print(f"All per-sample files: {sorted(all_suffixes)}")

    # ================================================================
    # Step 2: Classify by novelty level & split
    # ================================================================
    novelty_levels = np.array([count_novel(all_attrs[i]) for i in range(N)])

    print(f"\nNovel values: {NOVEL_VALUES}")
    print(f"Samples by novelty level:")
    for lv in range(4):
        n = (novelty_levels == lv).sum()
        if n > 0:
            print(f"  level {lv} (Hamming dist={lv}): {n} samples")

    train_indices = np.where(novelty_levels == 0)[0]
    test_indices = np.where(novelty_levels >= 1)[0]

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    # Split off validation from training
    n_val = max(1, int(len(train_indices) * args.val_ratio))
    val_indices = train_indices[:n_val]
    train_indices = train_indices[n_val:]

    train_combos = set(combo_key(all_attrs[i]) for i in train_indices)
    test_combos = set(combo_key(all_attrs[i]) for i in test_indices)

    print(f"\nFinal split:")
    print(f"  Train: {len(train_indices)} samples ({len(train_combos)} combos)")
    print(f"  Valid: {len(val_indices)} samples")
    print(f"  Test:  {len(test_indices)} samples ({len(test_combos)} combos)")

    test_novelty = novelty_levels[test_indices]
    print(f"\nTest Hamming distance distribution:")
    for lv in sorted(np.unique(test_novelty)):
        print(f"  dist={lv}: {(test_novelty == lv).sum()} samples")

    # ================================================================
    # Step 3: Overwrite ALL per-split npy files with new split
    # ================================================================
    print(f"\nOverwriting split files in {dst}:")

    def save_split(name, indices):
        indices = np.array(indices)
        for suffix, full_data in merged.items():
            out_path = dst / f"{name}_{suffix}"
            np.save(out_path, full_data[indices])
        print(f"  {name}: {len(indices)} samples x {len(merged)} files")

    save_split("train", train_indices)
    save_split("valid", val_indices)
    save_split("test", test_indices)

    # ================================================================
    # Step 4: Update meta.json
    # ================================================================
    meta_path = dst / "meta.json"
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
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Dataset: {dst}")


if __name__ == "__main__":
    main()
