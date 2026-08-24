"""Module 2 — building feature extraction (floor count + height).

Approach: deliberately heuristic, not learned. Working from Module 1's
window/door boxes, floors are counted by clustering window Y-centers into
horizontal rows — a new row starts when the vertical gap exceeds a fraction of
the median window height. Height is recovered with a physical ruler: the
ground-level front door is taken to be ~2.05 m, which sets a metres-per-pixel
scale applied to the building's pixel height, accepted only when the implied
per-floor height is plausible (else a floor-count × assumed-floor-height
fallback). Simple geometry plus a couple of real-world constants beats a
data-hungry regressor here: it needs no training data, is fully interpretable,
and degrades gracefully when detections are sparse.
"""
from .extractor import Bbox, BuildingFeatures, Detection
from .schemas import (
    BuildingFeaturesResult,
    FloatPrediction,
    HeightSource,
    ImageInfo,
    IntPrediction,
    Metadata,
    Predictions,
    ViewType,
)

__all__ = [
    "BuildingFeatures",
    "Detection",
    "Bbox",
    "BuildingFeaturesResult",
    "FloatPrediction",
    "IntPrediction",
    "Predictions",
    "Metadata",
    "ImageInfo",
    "ViewType",
    "HeightSource",
]
