"""Pydantic schemas for building feature extraction output."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_serializer


ViewType = Literal["facade", "topdown"]
HeightSource = Literal["door_scale", "fallback", "none"]


class ImageInfo(BaseModel):
    """Image identifier and pixel dimensions."""

    id: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class FloatPrediction(BaseModel):
    """Continuous-valued prediction with confidence."""

    value: float
    confidence: float = Field(ge=0.0, le=1.0)


class IntPrediction(BaseModel):
    """Integer-valued prediction with confidence."""

    value: int
    confidence: float = Field(ge=0.0, le=1.0)


class Predictions(BaseModel):
    """Bundle of building feature predictions."""

    building_height_m: FloatPrediction
    floor_count: IntPrediction


class Metadata(BaseModel):
    """Model and inference metadata."""

    model_version: str
    timestamp: datetime

    @field_serializer("timestamp")
    def _serialize_timestamp(self, value: datetime) -> str:
        return value.isoformat().replace("+00:00", "Z")


class BuildingFeaturesResult(BaseModel):
    """Structured result for building feature extraction."""

    image: ImageInfo
    view_type: ViewType
    predictions: Predictions
    height_source: HeightSource
    metadata: Metadata
