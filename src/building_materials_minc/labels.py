"""Facade-material label scheme for the MINC-trained classifier.

MINC-2500 ships 23 in-the-wild material classes. We keep a facade-relevant
subset and fold a couple of near-duplicates together:

    polishedstone -> stone   (marble/granite cladding read as stone)
    ceramic       -> tile

Everything non-architectural (carpet, foliage, food, skin, sky, water, ...) is
dropped from training. The classifier therefore outputs the 8 target classes
below.
"""
from __future__ import annotations

# MINC-2500 class order (the dataset's ClassLabel.names).
MINC23: list[str] = [
    "brick", "carpet", "ceramic", "fabric", "foliage", "food", "glass", "hair",
    "leather", "metal", "mirror", "other", "painted", "paper", "plastic",
    "polishedstone", "skin", "sky", "stone", "tile", "wallpaper", "water", "wood",
]

# Target facade materials the model predicts (contiguous, sorted).
TARGET_CLASSES: list[str] = [
    "brick", "glass", "metal", "painted", "plastic", "stone", "tile", "wood",
]
NUM_CLASSES: int = len(TARGET_CLASSES)
TARGET_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(TARGET_CLASSES)}

# MINC-23 class name -> target class (or absent = dropped from training).
MINC_TO_TARGET: dict[str, str] = {
    "brick": "brick",
    "glass": "glass",
    "metal": "metal",
    "painted": "painted",
    "plastic": "plastic",
    "stone": "stone",
    "polishedstone": "stone",  # merge
    "tile": "tile",
    "ceramic": "tile",         # merge
    "wood": "wood",
}


def minc_index_to_target_idx(minc_idx: int) -> int | None:
    """Map a MINC-23 label index to a target class index, or None if dropped."""
    target = MINC_TO_TARGET.get(MINC23[minc_idx])
    return TARGET_TO_IDX[target] if target is not None else None
