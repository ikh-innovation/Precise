"""Module 1 — facade parsing (semantic segmentation via SEEM)."""
from .parser import BACKBONE_REGISTRY, FacadeParser, ensure_backbone_assets
from .schemas import (
    BUILDING_LABELS,
    OPENING_LABELS,
    BBox,
    ClassMask,
    FacadeParsingResult,
    ImageInfo,
    Metadata,
    Polygon,
    ViewType,
)

__all__ = [
    "FacadeParser",
    "FacadeParsingResult",
    "ClassMask",
    "BBox",
    "Polygon",
    "ImageInfo",
    "Metadata",
    "ViewType",
    "BUILDING_LABELS",
    "OPENING_LABELS",
    "BACKBONE_REGISTRY",
    "ensure_backbone_assets",
]
