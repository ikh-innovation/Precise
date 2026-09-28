"""Metric scale recovery for Module 2 — what one image row is worth in metres.

Module 2 used to hang on a ground-floor door: the door fixed metres-per-pixel
and the building's pixel height was multiplied up. On this dataset the door is
usually absent — ground floors are shopfronts, awnings, porticos or garage
bands — so 16 of 20 packages fell back to ``floor_count * 3.2 m``, a "height"
that carries no information the floor count did not already carry.

This module replaces "one anchor, else a constant" with two independent steps.

**1. A vertical model.** Street-level photographs of tall blocks look *up*, so
metres-per-pixel is not constant down the image: the top floors are
foreshortened. For a point at real height ``Y`` on a vertical facade plane the
image row ``y`` is a projective function of ``Y``, hence

    t(y) = 1 / (y - y_vp)

is *affine* in ``Y``, where ``y_vp`` is the vertical vanishing point. Equally
spaced floor lines therefore land on an arithmetic progression in ``t``, and
that is enough to solve for ``y_vp`` from the facade's own repeating structure
— no line labelling, no calibration target, no camera metadata (these images
are screenshots, so there is no EXIF to read). Once ``t`` is known the t-span
of a feature is proportional to its true height *wherever* it sits on the
facade, so a window on floor 10 measures the same as one on floor 1.

**2. Scale anchors, fused.** Each cue below yields the same quantity — metres
per unit of ``t`` — so they combine instead of competing:

===========  ==========================================  ==================
source       reference length                            typical sigma_rel
===========  ==========================================  ==================
door         door height (2.05 m) over its t-span        0.08
floor_pitch  storey pitch (3.0 m) over one storey's      0.10
             t-span; available whenever the facade
             shows >= 3 window rows
window       window height (1.45 m) over the median      0.25
             window t-span
===========  ==========================================  ==================

They are merged by inverse-variance weighting in log space (scale errors are
multiplicative, not additive) after anchors that disagree with the median are
dropped. Building height is then ``metres_per_t * t-span of the facade mask``,
which — unlike ``floor_count * 3.2`` — measures the real extent: the taller
commercial ground floor, the roof parapet and any setback storey are inside
the mask and so are counted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EPS = 1e-9

Bbox = tuple[float, float, float, float]


@dataclass(frozen=True)
class VerticalModel:
    """Maps an image row to a coordinate that is affine in real-world height.

    Attributes:
        y_vp: Vertical vanishing point row, always above the facade
            (`y_vp < y` for every row of interest). `None` selects the affine
            model `t = -y`, used when the facade shows too few rows to solve
            for perspective or when doing so does not fit better.
        pitch_t: t-span of one storey, or `None` when it could not be measured.
        row_count: Number of floor rows the fit was based on.
        residual: Relative scatter of the storey pitch after the fit; lower is
            a more regular facade. `inf` when nothing was fitted.
    """

    y_vp: float | None = None
    pitch_t: float | None = None
    row_count: int = 0
    residual: float = math.inf

    @property
    def perspective_corrected(self) -> bool:
        """True when a vertical vanishing point was solved for."""
        return self.y_vp is not None

    def t(self, y: float) -> float:
        """Projective row coordinate; affine in real height on the facade."""
        if self.y_vp is None:
            return -float(y)
        return 1.0 / max(float(y) - self.y_vp, EPS)

    def span(self, y_a: float, y_b: float) -> float:
        """t-distance between two image rows (order-independent)."""
        return abs(self.t(y_a) - self.t(y_b))

    def row_at(self, t: float) -> float:
        """Inverse of `t()`: the image row a projective coordinate maps back to.

        Returns `inf` for a coordinate at or below the vanishing point's
        horizon, which is not a row on the facade at all — callers clamp
        against an observed bound rather than using such a value.
        """
        if self.y_vp is None:
            return -float(t)
        if t <= EPS:
            return math.inf
        return self.y_vp + 1.0 / float(t)

    def metres_per_pixel(self, metres_per_t: float, y: float) -> float:
        """Local vertical scale at row `y`, i.e. `|dY/dy|`.

        Only meaningful as a *local* figure: under perspective it shrinks
        toward the top of the image. Reported so the JSON carries a number a
        human can sanity-check against the picture.
        """
        if self.y_vp is None:
            return metres_per_t
        d = max(float(y) - self.y_vp, EPS)
        return metres_per_t / (d * d)


@dataclass(frozen=True)
class ScaleAnchor:
    """One estimate of metres-per-t from a reference length in the image."""

    source: str
    metres_per_t: float
    sigma_rel: float
    detail: dict[str, float]


@dataclass(frozen=True)
class FusedScale:
    """Inverse-variance combination of the available anchors."""

    metres_per_t: float
    sigma_rel: float
    used: tuple[str, ...]
    rejected: tuple[str, ...]


def _robust_unit(diffs: np.ndarray) -> tuple[float, float]:
    """Recover the single-storey step from gaps that may skip storeys.

    A row detector that misses one floor leaves a gap of ~2 storeys, so the
    unit is refined by snapping each gap to its nearest integer multiple
    rather than by averaging raw gaps.

    Args:
        diffs: Consecutive differences between sorted row coordinates.

    Returns:
        `(unit, relative_scatter)`; `(0.0, inf)` when nothing usable.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[d > EPS]
    if d.size == 0:
        return 0.0, math.inf
    unit = float(np.median(d))
    for _ in range(3):
        if unit <= EPS:
            return 0.0, math.inf
        k = np.maximum(np.round(d / unit), 1.0)
        unit = float(np.median(d / k))
    if unit <= EPS:
        return 0.0, math.inf
    k = np.maximum(np.round(d / unit), 1.0)
    normalised = d / k
    med = float(np.median(normalised))
    if med <= EPS:
        return 0.0, math.inf
    mad = float(np.median(np.abs(normalised - med)))
    return med, mad / med


def _score(rows: np.ndarray, y_vp: float | None) -> tuple[float, float]:
    """Storey pitch and its relative scatter under a candidate vanishing point."""
    if y_vp is None:
        t = -rows
    else:
        t = 1.0 / np.maximum(rows - y_vp, EPS)
    return _robust_unit(np.diff(np.sort(t)))


def fit_vertical_model(
    rows: list[float],
    extent: tuple[float, float],
    max_scale_ratio: float = 8.0,
    max_extra_top: float = 2.5,
    max_extra_bottom: float = 3.0,
    min_rows: int = 4,
    improvement: float = 0.85,
    samples: int = 128,
) -> VerticalModel:
    """Solve for the vertical vanishing point from equally spaced floor rows.

    Candidate vanishing points are swept above the facade; the winner is the
    one making the floor rows most nearly an arithmetic progression in `t`.
    Perspective is adopted only when it fits clearly better than the affine
    model, so a near-orthographic photo is not over-fitted.

    Two things keep the fit honest, because a vanishing point close above the
    image is numerically attractive to noisy rows but physically absurd — it
    claims the roof is tens of storeys up:

    * The sweep is parameterised by the scale ratio across the *detected rows*,
      the part of the facade the data actually supports, not across the whole
      mask. A camera across a street from a 30 m block gives about 3; a narrow
      street and a steep look-up can legitimately reach 8.
    * A candidate is rejected outright when it implies the facade sticks out
      past the top or bottom window row by more than a couple of storeys. That
      is where a bad vanishing point does its damage — extrapolation beyond
      the rows — and it is bounded by something real: above the top row sits
      half a storey plus a parapet or a setback penthouse, below the bottom
      row half a storey plus the (taller) shopfront storey.

    Args:
        rows: Image-row coordinates of the floor lines (window-row centres).
        extent: `(y_top, y_bottom)` of the facade — the range the fitted model
            must stay sane over, normally the facade mask's bbox.
        max_scale_ratio: Largest scale ratio across the detected rows that a
            candidate vanishing point may imply.
        max_extra_top: Storeys of facade allowed above the topmost row.
        max_extra_bottom: Storeys of facade allowed below the bottommost row.
        min_rows: Rows required before attempting the perspective fit.
        improvement: Fraction of the affine residual the perspective fit must
            beat to be accepted.
        samples: Candidates swept across the allowed ratio range.

    Returns:
        The fitted `VerticalModel`; affine when the fit is not worthwhile.
    """
    ys = np.sort(np.asarray(list(rows), dtype=float))
    if ys.size < 2:
        return VerticalModel(row_count=int(ys.size))

    pitch_a, res_a = _score(ys, None)
    best = VerticalModel(
        y_vp=None,
        pitch_t=pitch_a if pitch_a > EPS else None,
        row_count=int(ys.size),
        residual=res_a,
    )
    if ys.size < min_rows or max_scale_ratio <= 1.0:
        return best

    row_top, row_bot = float(ys[0]), float(ys[-1])
    if row_bot - row_top < 1.0:
        return best
    facade_top = min(float(extent[0]), row_top)
    facade_bot = max(float(extent[1]), row_bot)

    # `s` is the linear (not area) scale ratio across the detected rows, so the
    # vanishing point follows from requiring
    # (row_bot - y_vp) / (row_top - y_vp) == s.
    for s in np.linspace(1.02, math.sqrt(max_scale_ratio), samples):
        y_vp = (float(s) * row_top - row_bot) / (float(s) - 1.0)
        pitch, res = _score(ys, y_vp)
        if pitch <= EPS or res >= min(best.residual, res_a * improvement):
            continue
        candidate = VerticalModel(
            y_vp=y_vp, pitch_t=pitch, row_count=int(ys.size), residual=res
        )
        if (
            candidate.span(facade_top, row_top) / pitch > max_extra_top
            or candidate.span(row_bot, facade_bot) / pitch > max_extra_bottom
        ):
            continue
        best = candidate
    return best


# ----- anchors ---------------------------------------------------------------
def door_anchor(
    model: VerticalModel,
    door_box: Bbox,
    assumed_height_m: float,
    sigma_rel: float,
    snapped_px: float = 0.0,
) -> ScaleAnchor | None:
    """Metres-per-t from the front door's t-span.

    Args:
        model: Fitted vertical model.
        door_box: xyxy box of the chosen door, after any correction.
        assumed_height_m: Reference door height.
        sigma_rel: Relative 1-sigma for this anchor.
        snapped_px: Pixels the door's foot was moved down to reach the
            building's base. Reported rather than used, so a span that owes
            part of itself to a correction can be told from one that does not.
    """
    span = model.span(float(door_box[1]), float(door_box[3]))
    if span <= EPS:
        return None
    return ScaleAnchor(
        source="door",
        metres_per_t=assumed_height_m / span,
        sigma_rel=sigma_rel,
        detail={
            "reference_m": assumed_height_m,
            "span_px": round(float(door_box[3] - door_box[1]), 2),
            "snapped_px": round(float(snapped_px), 2),
        },
    )


def floor_pitch_anchor(
    model: VerticalModel, assumed_pitch_m: float, sigma_rel: float
) -> ScaleAnchor | None:
    """Metres-per-t from one storey's t-span.

    Available on any facade with enough window rows to measure a pitch, which
    is exactly the case the door anchor cannot serve. A facade whose rows are
    irregular is a poor ruler, so the scatter of the fit inflates sigma.
    """
    if model.pitch_t is None or model.pitch_t <= EPS:
        return None
    penalty = 1.0 + min(model.residual, 1.0)
    return ScaleAnchor(
        source="floor_pitch",
        metres_per_t=assumed_pitch_m / model.pitch_t,
        sigma_rel=sigma_rel * penalty,
        detail={"reference_m": assumed_pitch_m, "rows": float(model.row_count)},
    )


def window_anchor(
    model: VerticalModel,
    window_boxes: list[Bbox],
    assumed_height_m: float,
    sigma_rel: float,
) -> ScaleAnchor | None:
    """Metres-per-t from the median window t-span.

    Weak on its own — window heights vary far more than door heights or storey
    pitches — but it is an independent third opinion and costs nothing.
    """
    spans = [model.span(float(b[1]), float(b[3])) for b in window_boxes]
    spans = [s for s in spans if s > EPS]
    if not spans:
        return None
    median = float(np.median(spans))
    return ScaleAnchor(
        source="window",
        metres_per_t=assumed_height_m / median,
        sigma_rel=sigma_rel,
        detail={"reference_m": assumed_height_m, "count": float(len(spans))},
    )


def fuse(anchors: list[ScaleAnchor], log_tol: float = 0.35) -> FusedScale | None:
    """Combine anchors by inverse-variance weighting in log space.

    Scale errors are multiplicative, so the average is taken over `log`
    metres-per-t. Anchors further than `log_tol` in log space from the median
    are dropped first: a "door" mask that actually caught a shopfront shutter
    is an order-of-magnitude outlier, and averaging it in would corrupt an
    otherwise sound estimate.

    Args:
        anchors: Available anchors; may be empty.
        log_tol: Agreement tolerance in log space (0.35 ~ +/-42%).

    Returns:
        The fused scale, or `None` when no anchor was supplied.
    """
    usable = [a for a in anchors if a.metres_per_t > EPS]
    if not usable:
        return None
    logs = np.array([math.log(a.metres_per_t) for a in usable])
    deviation = np.abs(logs - float(np.median(logs)))
    keep = deviation <= log_tol
    if not keep.any():
        keep = deviation == deviation.min()

    weights = np.array([1.0 / max(a.sigma_rel, 1e-3) ** 2 for a in usable])[keep]
    mean_log = float(np.average(logs[keep], weights=weights))
    return FusedScale(
        metres_per_t=math.exp(mean_log),
        sigma_rel=float(1.0 / math.sqrt(float(weights.sum()))),
        used=tuple(a.source for a, k in zip(usable, keep) if k),
        rejected=tuple(a.source for a, k in zip(usable, keep) if not k),
    )
