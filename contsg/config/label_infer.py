"""
Helpers for inferring label classes from attribute combinations.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np


def infer_attr_label_map(
    data_folder: Path | str,
    num_attr_ops: Iterable[int],
    splits: Iterable[str] = ("train", "valid", "test"),
) -> dict[tuple[int, ...], int]:
    """
    Build an attribute-combo -> label index map based on observed combinations.

    Unknown values (-1) are mapped to the last class per attribute, matching
    LabelExtractionMixin behavior.
    """
    combos = _infer_attr_combos_cached(
        str(data_folder),
        tuple(int(n) for n in num_attr_ops),
        tuple(splits),
    )
    return {combo: idx for idx, combo in enumerate(combos)}


def infer_num_classes_from_attrs(
    data_folder: Path | str,
    num_attr_ops: Iterable[int],
    splits: Iterable[str] = ("train", "valid", "test"),
) -> int:
    """Return the number of observed attribute combinations across splits."""
    combos = _infer_attr_combos_cached(
        str(data_folder),
        tuple(int(n) for n in num_attr_ops),
        tuple(splits),
    )
    return len(combos)


@lru_cache(maxsize=32)
def _infer_attr_combos_cached(
    data_folder: str,
    num_attr_ops: tuple[int, ...],
    splits: tuple[str, ...],
) -> tuple[tuple[int, ...], ...]:
    if not num_attr_ops:
        raise ValueError("num_attr_ops is empty; cannot infer label combos.")

    folder = Path(data_folder)
    all_unique: np.ndarray | None = None
    found = False

    for split in splits:
        attrs_path = folder / f"{split}_attrs_idx.npy"
        if not attrs_path.exists():
            continue
        found = True

        attrs = np.load(attrs_path, mmap_mode="r")
        if attrs.ndim == 1:
            attrs = attrs.reshape(-1, 1)

        if attrs.shape[1] != len(num_attr_ops):
            raise ValueError(
                f"{attrs_path} has {attrs.shape[1]} columns, but num_attr_ops has "
                f"{len(num_attr_ops)} entries."
            )

        sanitized = np.array(attrs, copy=True)
        for idx, n_ops in enumerate(num_attr_ops):
            sanitized[:, idx] = np.where(sanitized[:, idx] == -1, n_ops - 1, sanitized[:, idx])

        unique = np.unique(sanitized, axis=0)
        if all_unique is None:
            all_unique = unique
        else:
            all_unique = np.unique(np.vstack([all_unique, unique]), axis=0)

    if not found or all_unique is None:
        expected = ", ".join(f"{folder}/{split}_attrs_idx.npy" for split in splits)
        raise FileNotFoundError(
            "No attrs_idx files found for label inference. Expected one of: "
            f"{expected}"
        )

    return tuple(tuple(int(v) for v in row.tolist()) for row in all_unique)
