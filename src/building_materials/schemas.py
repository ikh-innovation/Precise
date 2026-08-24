"""Pydantic schemas for material classification output."""
from typing import Literal

from pydantic import BaseModel, Field

# Mechanical reference data lives in the config layer (single source of truth);
# the output schema re-uses the same model so JSON dumps stay in lock-step.
from config import MaterialProperties

ViewType = Literal["facade", "topdown"]


class ImageInfo(BaseModel):
    """Image identifier and pixel dimensions."""

    id: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class MaterialScore(BaseModel):
    """A material label paired with its similarity score."""

    label: str
    score: float = Field(ge=0.0, le=1.0)


class Metadata(BaseModel):
    """Model and configuration metadata."""

    candidate_materials: list[str]
    model: str


class ObjectMaterial(BaseModel):
    """Per-instance material classification result."""

    bbox: list[float] = Field(min_length=4, max_length=4)
    dominant_material: MaterialScore
    properties: MaterialProperties | None = Field(
        default=None,
        description="Fixed mechanical properties of `dominant_material`, "
        "or null if the label has no config entry.",
    )


class MaterialClassificationResult(BaseModel):
    """Structured result for a building-region material classification.

    The material is read strictly from Module 1's building region (red
    facade/house minus window/door openings); `classified_region` records how
    those pixels were obtained. When Module 1 finds no building, the result is
    `classified_region="none"` with `dominant_material.label == "none"` — the
    pipeline never falls back to classifying the whole image.
    """

    image: ImageInfo
    view_type: ViewType
    materials: list[MaterialScore]
    dominant_material: MaterialScore
    dominant_material_properties: MaterialProperties | None = Field(
        default=None,
        description="Fixed mechanical properties of `dominant_material`, "
        "or null if the label has no config entry.",
    )
    classified_region: Literal["patches", "masked_bbox", "none"] = Field(
        default="patches",
        description="Provenance of the classified pixels: 'patches' = tiled "
        "wall patches, 'masked_bbox' = a single region-bbox crop with "
        "non-wall pixels neutralised, 'none' = no building region found "
        "(no whole-image fallback).",
    )
    metadata: Metadata


class PerObjectMaterialResult(BaseModel):
    """Structured result for per-instance material classification.

    `objects` is keyed by SEEM class label (`house`, `window`, `door`) and
    holds one entry per detected instance.
    """

    image: ImageInfo
    view_type: ViewType
    objects: dict[str, list[ObjectMaterial]]
    metadata: Metadata
