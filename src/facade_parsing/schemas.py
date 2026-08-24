"""Pydantic schemas for facade parsing output (semantic per-class)."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_serializer


ViewType = Literal["facade", "topdown"]

# Canonical Module 1 label vocabulary, shared with the pipeline + Module 3.
# The RED building class is "house" under the SEEM backend and "facade" under
# the trained SegFormer backend; both denote the same building surface. Defined
# here (the contract owner) so client.py and the material backends agree on what
# counts as the building region and as openings — no magic strings scattered
# across modules.
BUILDING_LABELS: tuple[str, ...] = ("house", "facade")
OPENING_LABELS: tuple[str, ...] = ("window", "door")


class ImageInfo(BaseModel):
    """Image identifier and pixel dimensions."""

    id: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class BBox(BaseModel):
    """Axis-aligned bounding box in both pixel (xyxy) and normalized (0-1) form."""

    pixel: list[float] = Field(min_length=4, max_length=4)
    normalized: list[float] = Field(min_length=4, max_length=4)


class Polygon(BaseModel):
    """Contour points as a list of [x, y], in both pixel and normalized form."""

    pixel: list[list[float]]
    normalized: list[list[float]]


class ClassMask(BaseModel):
    """One semantic class's mask, summary geometry, and confidence."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    pixel_area: int = Field(ge=0)
    bbox: BBox | None = None
    polygons: list[Polygon] = Field(default_factory=list)
    mask_path: str | None = None


class Metadata(BaseModel):
    """Model and inference metadata."""

    model_version: str
    backbone: str
    prompts: list[str]
    threshold: float
    timestamp: datetime

    @field_serializer("timestamp")
    def _serialize_timestamp(self, value: datetime) -> str:
        return value.isoformat().replace("+00:00", "Z")


class FacadeParsingResult(BaseModel):
    """Structured result for semantic facade parsing."""

    image: ImageInfo
    view_type: ViewType
    classes: list[ClassMask]
    metadata: Metadata
