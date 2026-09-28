"""Module 2 — building feature extraction (floor count + height).

Approach: deliberately heuristic, not learned. Working from Module 1's masks,
floors are counted by clustering window Y-centers into horizontal rows — a new
row starts when the vertical gap exceeds a fraction of the median window height
— and gaps that span two storeys are counted as two.

Height is *measured* rather than assumed. `scale.py` first solves for the
facade's vertical vanishing point from the storey rhythm itself, which removes
the foreshortening of a street-level photo taken looking up; then it fuses
every reference length the image offers into a single metres-per-pixel scale:
the ground-floor door (~2.05 m) when one is visible, the storey pitch (~3.0 m)
whenever the facade shows enough window rows, and the median window (~1.45 m)
as a weak third opinion. The height is the facade mask's full extent through
that scale, so the taller shopfront storey and the roof parapet count.

`floor_count × assumed_floor_height_m` remains only as the last resort, for
images where no anchor at all can be formed. Simple geometry plus a couple of
real-world constants beats a data-hungry regressor here: it needs no training
data, is fully interpretable — the `scale` block of the output names the cues
that carried the estimate — and degrades gracefully when detections are sparse.
"""
from .extractor import Bbox, BuildingFeatures, Detection
from .scale import (
    FusedScale,
    ScaleAnchor,
    VerticalModel,
    fit_vertical_model,
    fuse,
)
from .schemas import (
    BuildingFeaturesResult,
    FloatPrediction,
    HeightSource,
    ImageInfo,
    IntPrediction,
    Metadata,
    Predictions,
    ScaleAnchorInfo,
    ScaleReport,
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
    "ScaleReport",
    "ScaleAnchorInfo",
    "VerticalModel",
    "ScaleAnchor",
    "FusedScale",
    "fit_vertical_model",
    "fuse",
]
