"""SegFormer (MiT-B0) reproduction on the full 12-class CMP Facade Database.

A faithful reproduction of the documented HuggingFace SegFormer-on-CMP recipe
(AdamW lr 6e-5, 50 epochs, batch 2, mean-IoU selection), kept separate from the
4-class pipeline module. Training/inference are the ``train.py`` / ``inference.py``
scripts; the lightweight (transformers-free) symbols are exported here.
"""
from .data import CmpSegformerDataset, discover_pairs, split_pairs
from .labels import (
    CLASS_NAMES,
    COLORS_BGR,
    ID2LABEL,
    LABEL2ID,
    NUM_CLASSES,
    build_remap_lut,
    colorize_mask,
)

__all__ = [
    "CLASS_NAMES",
    "NUM_CLASSES",
    "ID2LABEL",
    "LABEL2ID",
    "COLORS_BGR",
    "build_remap_lut",
    "colorize_mask",
    "CmpSegformerDataset",
    "discover_pairs",
    "split_pairs",
]
