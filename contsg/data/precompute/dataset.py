"""
Dataset embedding precomputation utilities.

Functions for precomputing embeddings for entire datasets.
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from .base import EmbeddingPrecomputer


def load_captions(
    dataset_dir: Path,
    split: str,
    variant: str = "",
) -> Optional[np.ndarray]:
    """
    Load captions from dataset.

    Supports multiple naming conventions:
    - {split}_caps.npy (simple)
    - {split}_text_caps.npy (multi-segment)
    - {split}_text_caps_{variant}.npy (variant)

    Args:
        dataset_dir: Path to dataset directory
        split: Data split (train, valid, test)
        variant: Caption variant (empty for base)

    Returns:
        Captions array or None if not found
    """
    # Try different naming conventions
    candidates = []

    if variant and variant != "base":
        candidates.append(f"{split}_text_caps_{variant}.npy")
        candidates.append(f"{split}_caps_{variant}.npy")
    else:
        candidates.append(f"{split}_text_caps.npy")
        candidates.append(f"{split}_caps.npy")

    for filename in candidates:
        path = dataset_dir / filename
        if path.exists():
            return np.load(path, allow_pickle=True)

    return None


def merge_segments(caps_array: np.ndarray) -> List[str]:
    """
    Merge multi-segment captions into single strings.

    Args:
        caps_array: Shape (N,) or (N, max_segments)

    Returns:
        List of N merged text strings
    """
    merged_texts = []

    for i in range(caps_array.shape[0]):
        item = caps_array[i]

        if isinstance(item, np.ndarray):
            # Multi-segment: join non-empty segments
            segments = [str(seg).strip() for seg in item if str(seg).strip()]
            merged_text = " ".join(segments)
        else:
            # Single segment
            merged_text = str(item).strip()

        merged_texts.append(merged_text)

    return merged_texts


def precompute_dataset_embeddings(
    dataset_dir: Path,
    precomputer: EmbeddingPrecomputer,
    splits: List[str] = ["train", "valid", "test"],
    variants: List[str] = [""],
    batch_size: int = 64,
    overwrite: bool = False,
    output_suffix: Optional[str] = None,
) -> List[Path]:
    """
    Precompute embeddings for an entire dataset.

    Args:
        dataset_dir: Path to dataset directory
        precomputer: Embedding precomputer instance
        splits: List of splits to process
        variants: List of caption variants
        batch_size: Batch size for encoding
        overwrite: Whether to overwrite existing files
        output_suffix: Custom suffix for output files

    Returns:
        List of paths to created embedding files
    """
    dataset_dir = Path(dataset_dir)
    created_files = []

    # Build output suffix
    suffix = output_suffix or f"{precomputer.model_name}_{precomputer.embed_dim}"

    for split in splits:
        for variant in variants:
            variant_display = variant or "base"
            print(f"\n{'='*60}")
            print(f"Processing: {split} | variant: {variant_display}")
            print(f"{'='*60}")

            # Build output path
            variant_prefix = f"{variant}_" if variant and variant != "base" else ""
            out_filename = f"{split}_cap_emb_{variant_prefix}{suffix}.npy"
            out_path = dataset_dir / out_filename

            # Check if exists
            if out_path.exists() and not overwrite:
                print(f"[{split}][{variant_display}] Already exists, skipping: {out_path}")
                continue

            # Load captions
            caps = load_captions(dataset_dir, split, variant)
            if caps is None:
                print(f"[{split}][{variant_display}] Captions not found, skipping")
                continue

            print(f"[{split}][{variant_display}] Loaded: {caps.shape}")

            # Merge segments
            texts = merge_segments(caps)
            n = len(texts)

            if n > 0:
                print(f"[{split}][{variant_display}] Example (truncated):")
                print(f"  {texts[0][:150]}...")

            # Compute embeddings
            print(f"[{split}][{variant_display}] Encoding {n} texts...")
            start = time.time()

            embeddings = precomputer.compute(texts, batch_size=batch_size)

            # Validate
            if embeddings.shape != (n, precomputer.embed_dim):
                raise ValueError(
                    f"Shape mismatch: got {embeddings.shape}, "
                    f"expected ({n}, {precomputer.embed_dim})"
                )

            # Save
            np.save(out_path, embeddings.astype(np.float32))
            created_files.append(out_path)

            elapsed = time.time() - start
            print(f"[{split}][{variant_display}] ✓ Saved: {out_path}")
            print(f"[{split}][{variant_display}] ✓ Shape: {embeddings.shape}")
            print(f"[{split}][{variant_display}] ✓ Time: {elapsed:.1f}s")

    return created_files


def check_embeddings_exist(
    dataset_dir: Path,
    embed_dim: int = 1024,
    model_name: str = "qwen3-embedding",
    splits: List[str] = ["train", "valid", "test"],
) -> Tuple[bool, List[str]]:
    """
    Check if precomputed embeddings exist for a dataset.

    Args:
        dataset_dir: Path to dataset
        embed_dim: Expected embedding dimension
        model_name: Model name in filename
        splits: Splits to check

    Returns:
        (all_exist, list_of_missing_files)
    """
    dataset_dir = Path(dataset_dir)
    missing = []

    suffix = f"{model_name}_{embed_dim}"

    for split in splits:
        # Check for embedding file
        emb_path = dataset_dir / f"{split}_cap_emb_{suffix}.npy"
        if not emb_path.exists():
            # Also check without model name suffix (legacy format)
            legacy_path = dataset_dir / f"{split}_cap_emb.npy"
            if not legacy_path.exists():
                missing.append(str(emb_path))

    return len(missing) == 0, missing
