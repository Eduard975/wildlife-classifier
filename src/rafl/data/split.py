"""Location-based train/val/test split with zero-overlap guarantee.

Camera-trap data leaks if split randomly: the same background appears in train
and test, inflating accuracy (Beery et al., Recognition in Terra Incognita).
We hold out whole camera locations instead.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict


def assign_split(location_id: str, fractions: dict[str, float], salt: str = "rafl") -> str:
    """Deterministically map a location_id to a split by hashing.

    Hashing the location (not the row) guarantees every row of a location lands
    in the same split, so no location can straddle two splits.
    """
    h = hashlib.sha256(f"{salt}:{location_id}".encode()).hexdigest()
    x = int(h[:16], 16) / float(1 << 64)
    train, val = fractions["train"], fractions["val"]
    if x < train:
        return "train"
    if x < train + val:
        return "val"
    return "test"


def split_locations(location_ids, fractions: dict[str, float], salt: str = "rafl"):
    """Return {location_id: split}. Pure function of the id set and fractions."""
    return {loc: assign_split(loc, fractions, salt) for loc in set(location_ids)}


def no_location_overlap(split_map: dict[str, str]) -> bool:
    """True iff every location maps to exactly one split (invariant we assert)."""
    seen: dict[str, set[str]] = defaultdict(set)
    for loc, sp in split_map.items():
        seen[loc].add(sp)
    return all(len(v) == 1 for v in seen.values())
