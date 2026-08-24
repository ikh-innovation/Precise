"""Module 3 — building material classification."""
from .classifier import Bbox, BuildingMaterials
from .schemas import (
    ImageInfo,
    MaterialClassificationResult,
    MaterialProperties,
    MaterialScore,
    Metadata,
    ObjectMaterial,
    PerObjectMaterialResult,
    ViewType,
)

__all__ = [
    "BuildingMaterials",
    "Bbox",
    "MaterialClassificationResult",
    "PerObjectMaterialResult",
    "ObjectMaterial",
    "MaterialScore",
    "MaterialProperties",
    "Metadata",
    "ImageInfo",
    "ViewType",
]
