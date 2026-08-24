"""Facade-8 material taxonomy and per-source label maps.

The DINOv3 facade backend predicts eight exterior building materials chosen to
match real facades and to fix the two gaps the MINC backend left — it had no
``concrete`` and no ``render``/plaster class, and ``painted`` was a finish, not
a material.

    Facade-8: brick, concrete, render, stone, glass, metal, wood, tile

`render` covers plaster / stucco / EIFS coatings. `metal` folds steel and
aluminium together (visually near-identical on a facade). Training data is
pooled from several facade-domain sources, each mapped into this single
taxonomy below; labels absent from a map are dropped from that source.
"""
from __future__ import annotations

TARGET_CLASSES: list[str] = [
    "brick", "concrete", "render", "stone", "glass", "metal", "wood", "tile",
]
NUM_CLASSES: int = len(TARGET_CLASSES)
TARGET_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(TARGET_CLASSES)}

# --- MINC-2500 (23 in-the-wild classes) -> Facade-8 -------------------------
# Material-centric patches. We keep only unambiguous exterior materials and
# drop "painted"/"plastic"/non-architectural classes. MINC is our main source
# of `tile` and adds patch-level texture for brick/stone/metal/glass/wood.
MINC_TO_TARGET: dict[str, str] = {
    "brick": "brick",
    "glass": "glass",
    "metal": "metal",
    "stone": "stone",
    "polishedstone": "stone",   # marble/granite cladding
    "tile": "tile",
    "ceramic": "tile",          # ceramic ~ glazed tile
    "wood": "wood",
}

# --- OpenFACADES (surface_material) -> Facade-8 -----------------------------
# ~19k single-label building images; our primary source of `concrete` and
# `render` (plaster). Whole-building crops -> weakly-labeled material crops.
OPENFACADES_TO_TARGET: dict[str, str] = {
    "brick": "brick",
    "concrete": "concrete",
    "plaster": "render",
    "stucco": "render",
    "glass": "glass",
    "metal": "metal",
    "wood": "wood",
    "stone": "stone",
}

# --- Urban Resource Cadastre (multi-label columns) -> Facade-8 --------------
# Single-label images only. `stucco` = render; `rustication` = rusticated stone
# masonry. `siding` is dropped (ambiguous vinyl/wood/metal), as are null/other.
URC_TO_TARGET: dict[str, str] = {
    "brick": "brick",
    "metal": "metal",
    "stucco": "render",
    "wood": "wood",
    "rustication": "stone",
}

# --- London/Scotland cladding-material subset -> Facade-8 -------------------
# Class folders are Brick / Concrete / Stone / Curtain-Wall / Mixed / Others.
# `curtain-wall` is a glass facade. `mixed`/`others` are dropped (not a single
# material).
LONDONSCOT_TO_TARGET: dict[str, str] = {
    "brick": "brick",
    "concrete": "concrete",
    "stone": "stone",
    "curtain-wall": "glass",
}


def map_label(source: str, raw: str) -> str | None:
    """Map a raw `source` label to a Facade-8 class, or None to drop it."""
    table = {
        "minc": MINC_TO_TARGET,
        "openfacades": OPENFACADES_TO_TARGET,
        "urc": URC_TO_TARGET,
        "londonscot": LONDONSCOT_TO_TARGET,
    }[source]
    return table.get(raw.strip().lower())
