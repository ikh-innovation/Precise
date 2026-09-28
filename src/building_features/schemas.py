"""Pydantic schemas for building feature extraction output."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_serializer


ViewType = Literal["facade", "topdown"]

# How the metric scale behind `building_height_m` was obtained.
#   door_scale   a single anchor, the ground-floor door (as before)
#   floor_pitch  a single anchor, the measured storey pitch
#   window_scale a single anchor, the median window height
#   fused        two or more anchors combined (see `scale.anchors`)
#   fallback     no anchor survived; floor_count * assumed_floor_height_m
#   none         nothing to go on
HeightSource = Literal[
    "door_scale", "floor_pitch", "window_scale", "fused", "fallback", "none"
]


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


class ScaleAnchorInfo(BaseModel):
    """One reference length that was turned into a metric scale."""

    source: str
    reference_m: float
    metres_per_pixel_at_base: float
    sigma_rel: float
    used: bool
    detail: dict[str, float] = Field(default_factory=dict)


class ScaleReport(BaseModel):
    """How image pixels were converted to metres, and by which cues.

    `metres_per_pixel_at_base` is the *local* scale at the foot of the facade.
    Under perspective the scale shrinks toward the top of the image, which is
    why the height is computed in the projective coordinate rather than by
    multiplying a single metres-per-pixel by the pixel height.
    """

    metres_per_pixel_at_base: float
    sigma_rel: float
    perspective_corrected: bool
    vertical_vanishing_point_y: float | None = None
    storey_pitch_m: float | None = None
    storey_pitch_residual: float | None = None
    facade_span_px: float
    rows_detected: int
    anchors: list[ScaleAnchorInfo] = Field(default_factory=list)


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
    scale: ScaleReport | None = None
    metadata: Metadata
