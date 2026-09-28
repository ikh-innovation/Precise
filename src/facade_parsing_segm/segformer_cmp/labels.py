"""12-class CMP Facade label scheme + standard colors for the SegFormer model.

This reproduces the *full* CMP Facade Database class set (the scheme used by
the published SegFormer-on-CMP recipe), not the 4-class pipeline scheme. CMP
masks store ids 1..12; we map them to contiguous 0-indexed ids 0..11 (simply
``raw - 1``) as HuggingFace expects.
"""
from __future__ import annotations

import numpy as np

# 0-indexed CMP classes, in native CMP order (raw id 1..12 -> 0..11).
CLASS_NAMES: list[str] = [
    "background",  # 0  (CMP 1)
    "facade",      # 1  (CMP 2)
    "window",      # 2  (CMP 3)
    "door",        # 3  (CMP 4)
    "cornice",     # 4  (CMP 5)
    "sill",        # 5  (CMP 6)
    "balcony",     # 6  (CMP 7)
    "blind",       # 7  (CMP 8)
    "deco",        # 8  (CMP 9)
    "molding",     # 9  (CMP 10)
    "pillar",      # 10 (CMP 11)
    "shop",        # 11 (CMP 12)
]
NUM_CLASSES: int = len(CLASS_NAMES)
ID2LABEL: dict[int, str] = {i: n for i, n in enumerate(CLASS_NAMES)}
LABEL2ID: dict[str, int] = {n: i for i, n in enumerate(CLASS_NAMES)}

# Standard color per class in BGR (OpenCV). facade/window/door reuse the
# pipeline palette (red/blue/green); the rest get distinct, stable colors.
COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "background": (0, 0, 0),
    "facade": (0, 0, 255),       # red
    "window": (255, 0, 0),       # blue
    "door": (0, 255, 0),         # green
    "cornice": (0, 255, 255),    # yellow
    "sill": (255, 255, 0),       # cyan
    "balcony": (255, 0, 255),    # magenta
    "blind": (0, 128, 255),      # orange
    "deco": (128, 0, 128),       # purple
    "molding": (0, 128, 128),    # olive
    "pillar": (128, 128, 0),     # teal
    "shop": (192, 192, 192),     # light gray
}


# Pixel frequency of each class over the CMP training split (485 images,
# 303.3 M pixels; `split_pairs(discover_pairs("base"))`, the same seed the
# model was trained with). `door` is the *rarest* class in the set — 31x rarer
# than `facade` and 2.4x rarer than `shop`, the class it loses glass entrances
# to — so a model trained with plain cross-entropy carries a prior that argues
# against it at every pixel. `prior_ratio` turns that into the correction.
CMP_PIXEL_FREQ: dict[str, float] = {
    "facade": 0.38592,
    "background": 0.18125,
    "window": 0.13053,
    "molding": 0.10084,
    "balcony": 0.03775,
    "cornice": 0.03512,
    "shop": 0.02984,
    "sill": 0.02683,
    "deco": 0.02309,
    "pillar": 0.02159,
    "blind": 0.01469,
    "door": 0.01256,
}


def prior_ratio(rare: str, common: str, tau: float = 1.0) -> float:
    """How much to favour `rare` over `common` to undo their prior imbalance.

    This is the logit-adjustment correction: scoring by `p(c|x) / prior(c)**tau`
    instead of `p(c|x)` removes the training prior from the decision, so the
    argmax compares how well each class *explains the pixel* rather than how
    often it was seen. `tau = 1` removes the prior entirely, `tau = 0` leaves
    the model's own posterior alone, and values between interpolate.

    Returns:
        `(freq[common] / freq[rare]) ** tau`, or 1.0 for an unknown class.
    """
    a, b = CMP_PIXEL_FREQ.get(rare), CMP_PIXEL_FREQ.get(common)
    if not a or not b:
        return 1.0
    return float((b / a) ** tau)


def build_remap_lut() -> np.ndarray:
    """256-entry uint8 LUT mapping raw CMP ids (1..12) to 0-indexed ids (0..11).

    Raw 0 (unused in CMP masks) and out-of-range ids map to 0 (background).
    """
    lut = np.zeros(256, dtype=np.uint8)
    for raw in range(1, 13):
        lut[raw] = np.uint8(raw - 1)
    return lut


def colors_bgr_array() -> np.ndarray:
    """``[12, 3]`` uint8 BGR table indexed by 0-indexed class id."""
    table = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
    for i, name in enumerate(CLASS_NAMES):
        table[i] = COLORS_BGR[name]
    return table


def colorize_mask(label_map: np.ndarray) -> np.ndarray:
    """Map a ``[H, W]`` 0-indexed label map to a ``[H, W, 3]`` BGR image."""
    return colors_bgr_array()[label_map]
