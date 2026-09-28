"""Module 2 — building feature extraction (floor count + height).

Height is measured, not assumed. `scale.py` recovers a vertical model of the
facade (a vanishing point solved from the storey rhythm itself) plus a metric
scale fused from every reference length the image offers — the door when it is
visible, the storey pitch and the median window otherwise. The height is then
the facade mask's extent converted through that model, so the taller
commercial ground floor, the parapet and any setback storey are all included.

`floor_count * assumed_floor_height_m` survives only as the last resort for
images where no anchor at all could be formed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from config import BuildingFeaturesConfig
from .scale import (
    FusedScale,
    ScaleAnchor,
    VerticalModel,
    door_anchor,
    fit_vertical_model,
    floor_pitch_anchor,
    fuse,
    window_anchor,
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


Detection = tuple[np.ndarray, float]
Bbox = tuple[float, float, float, float]

# `height_source` reported when exactly one anchor carried the estimate.
_SINGLE_SOURCE: dict[str, HeightSource] = {
    "door": "door_scale",
    "floor_pitch": "floor_pitch",
    "window": "window_scale",
}


class BuildingFeatures:
    """Estimate floor count and building height from facade detections."""

    def __init__(
        self,
        image_path: str | Path,
        cfg: BuildingFeaturesConfig,
        view_type: ViewType = "facade",
    ) -> None:
        """Bind the input image and configuration.

        Args:
            image_path: Path to the facade image (used only for size + result id).
            cfg: Module 2 configuration section.
            view_type: Pipeline view type, propagated into the result schema.
        """
        self.image_path = Path(image_path)
        self.cfg = cfg
        self.view_type: ViewType = view_type
        self.last_detection_counts: dict[str, int] = {"window": 0, "door": 0}
        # Largest correction `_snap_doors_to_base` applied on the last run,
        # reported on the door anchor so the span can be traced.
        self.last_door_snap_px: float = 0.0

    # ----- row structure -----------------------------------------------------
    def _cluster_rows(
        self, boxes: list[np.ndarray], prune: bool = True
    ) -> list[list[int]]:
        """Greedy 1-D clustering of box Y-centers into horizontal rows.

        Args:
            boxes: List of xyxy boxes.
            prune: Drop rows too sparse to be a storey. Wanted when the rows
                are about to be used as a ruler, not when they are being
                counted — see `_drop_sparse_rows`.

        Returns:
            List of clusters, each cluster being indices into `boxes`.
        """
        if not boxes:
            return []
        y_centers = np.array([(b[1] + b[3]) / 2.0 for b in boxes])
        heights = np.array([b[3] - b[1] for b in boxes])
        order = np.argsort(y_centers)
        median_h = float(np.median(heights))
        tol = max(median_h * self.cfg.window_cluster_tol_ratio, 10.0)

        rows: list[list[int]] = []
        current: list[int] = [int(order[0])]
        last_y = float(y_centers[order[0]])
        for idx in order[1:]:
            y = float(y_centers[idx])
            if abs(y - last_y) <= tol:
                current.append(int(idx))
            else:
                rows.append(current)
                current = [int(idx)]
            last_y = y
        rows.append(current)
        return self._drop_sparse_rows(rows) if prune else rows

    def _drop_sparse_rows(self, rows: list[list[int]]) -> list[list[int]]:
        """Discard rows holding far fewer boxes than a real storey does.

        A storey spans the whole facade, so its row collects several windows.
        One stray polygon — a reflection, a vent, half a window split off by
        the segmentation — otherwise becomes a row of its own, and a spurious
        row halves the apparent storey pitch, which corrupts every anchor
        derived from it. Rows are only thinned when the facade actually has
        enough windows per storey for the count to mean something.
        """
        if len(rows) < 3:
            return rows
        counts = np.array([len(r) for r in rows], dtype=float)
        median = float(np.median(counts))
        if median <= 2.0:
            return rows
        keep = [r for r in rows if len(r) >= median * self.cfg.min_row_members_ratio]
        return keep if len(keep) >= 2 else rows

    def _row_centres(self, boxes: list[np.ndarray], prune: bool = True) -> list[float]:
        """Median Y-centre of each detected row, ascending."""
        centres = [
            float(np.median([(boxes[i][1] + boxes[i][3]) / 2.0 for i in cluster]))
            for cluster in self._cluster_rows(boxes, prune=prune)
        ]
        return sorted(centres)

    def _estimate_floor_count(
        self, windows: list[Detection], model: VerticalModel, from_windows: bool
    ) -> tuple[int, float]:
        """Compute floor count and a row-regularity confidence.

        Rows are counted in the vertical model's coordinate rather than in
        pixels, so a storey whose windows were all missed (shuttered, occluded
        by a tree) still shows up as a double-width gap and is counted.

        Args:
            windows: List of window detections.
            model: Fitted vertical model; supplies the storey pitch.
            from_windows: Whether the model was fitted on these window rows.
                When it was fitted on `sill`/`balcony` rows instead its pitch
                is offset from the window rhythm, and filling gaps with it
                would invent storeys.

        Returns:
            `(floor_count, confidence)`.
        """
        if not windows:
            return 0, 0.0
        boxes = [w[0] for w in windows]
        rows = self._cluster_rows(boxes)
        if not rows:
            return 0, 0.0

        # Count across the *unpruned* rows: pruning protects the ruler from
        # stray polygons, but a thinly-detected top storey (a setback
        # penthouse, or simply the furthest windows) is a real storey, and
        # dropping it at the end of the range would lose a floor with no gap
        # left behind for the pitch to recover.
        #
        # Gaps are converted storey-by-storey rather than by dividing the
        # whole span: a locally-wrong pitch then costs one floor instead of
        # compounding over the facade. A gap under half a storey is a row the
        # segmentation split in two, and contributes nothing.
        n_floors = len(rows)
        centres = self._row_centres(boxes, prune=False)
        if (
            from_windows
            and model.pitch_t is not None
            and model.pitch_t > 0
            and len(centres) > 1
        ):
            t = np.sort(np.array([model.t(c) for c in centres]))
            steps = np.round(np.diff(t) / model.pitch_t)
            n_floors = max(n_floors, int(1 + steps.clip(min=0.0).sum()))

        counts = np.array([len(r) for r in rows], dtype=float)
        mean_count = float(counts.mean())
        if mean_count <= 0:
            return n_floors, 0.3
        std_count = float(counts.std())
        regularity = max(0.0, 1.0 - (std_count / (mean_count + 1e-6)))
        mean_score = float(np.mean([w[1] for w in windows]))
        confidence = float(np.clip(0.5 * regularity + 0.5 * mean_score, 0.0, 1.0))
        return n_floors, round(confidence, 4)

    # ----- geometry ----------------------------------------------------------
    def _pick_door_anchor(
        self, doors: list[Detection], building_pixel_h: float
    ) -> int | None:
        """Pick the front door used as the metric scale anchor.

        Front-door heuristic: among detected doors, prefer those whose
        bottom edge sits near the bottom of the building (within
        `bottom_door_tol_ratio` of the building's pixel height), and from
        that subset pick the tallest one. Falls back to the overall
        tallest door if no door sits near the bottom.

        Args:
            doors: List of door detections.
            building_pixel_h: Pixel height of the building region.

        Candidates narrower than `door_max_aspect` allows are discarded before
        the choice. A door leaf is about 0.9 m wide and 2.05 m tall, so even a
        narrow one photographed obliquely stays under about 3; a mask five or
        six times taller than it is wide is a shadow down a reveal, a drainpipe
        or a strip of dark reveal beside a shopfront, not a door. The test
        matters because the choice is "tallest wins", which is exactly the
        ranking a sliver is best at: on package 19 a 12 px-wide strip beside
        the ATM outranked the real Santander entrance next to it.

        Only the upper bound is applied. A wide candidate is a double door, a
        garage or an entrance whose lower half has not been recovered yet —
        package 19's entrance is 76 px wide and 43 px tall before its foot is
        put back on the ground — and rejecting those would throw away the very
        doors the rest of this module exists to repair.

        Returns:
            Index into `doors` of the chosen detection. `None` when `doors` is
            empty or nothing in it is shaped like a door — in which case there
            is no door to anchor on, and Module 2 falls back to the storey
            pitch rather than measuring 2.05 m against a drainpipe. An index
            rather than the detection itself, so the caller can also recover
            what `_snap_door_to_base` did to it.
        """
        if not doors:
            return None
        plausible = [
            i
            for i, (b, _) in enumerate(doors)
            if float(b[3] - b[1]) <= self.cfg.door_max_aspect * max(float(b[2] - b[0]), 1.0)
        ]
        if not plausible:
            return None
        bottom_y = max(float(doors[i][0][3]) for i in plausible)
        tol = max(building_pixel_h * self.cfg.bottom_door_tol_ratio, 5.0)
        candidates = [
            i for i in plausible if bottom_y - float(doors[i][0][3]) <= tol
        ] or plausible
        return max(candidates, key=lambda i: float(doors[i][0][3] - doors[i][0][1]))

    def _snap_door_to_base(
        self, box: np.ndarray, model: VerticalModel, base_row: float
    ) -> tuple[Bbox, float]:
        """Close the gap between a door's detected foot and the building's.

        A door reaches the ground; nothing else about it is as certain. So a
        door whose mask stops short of the base row did not end there — either
        something is parked in front of it, or the segmentation lost its lower
        half to the shopfront glazing it is set into, CMP's `shop` class being
        a fair description of a glass entrance. Both truncate the span Module 2
        divides 2.05 m by, and a short span makes the building *tall*: package
        19's entrance survived as 42 px of lintel, implied a 0.72 m door, and
        returned 42.7 m for a facade the storey rhythm puts at about 25 m.

        The base row is the one `_base_row` already resolved, so it is the
        scene gate's ground bracket after the storey pitch has vouched for it,
        not the raw ground pixels — those run on to the road well below the
        footing.

        Only a door that is *implausibly short* is corrected, and the storey
        pitch says what short means: 2.05 m in a 3 m storey is about 0.68 of a
        pitch, so a door already past `door_short_storeys` is a door, a
        shopfront or a garage band and is left exactly as detected, while one
        under it is missing part of itself. That gate is what keeps the
        correction off the packages it could only damage. The result is then
        refused if it would exceed `door_max_storeys`, which is the case where
        the base row sits far below the door for a reason of its own.

        The pitch only decides *whether* the foot can be trusted. What replaces
        it is the base row — image evidence — so the door anchor still measures
        lintel-to-ground and keeps its independence from the pitch anchor it is
        later fused with.

        Only the foot moves. There is no matching argument for the lintel, and
        a door allowed to grow upward would be unfalsifiable. And only the
        chosen door moves, after `_pick_door_anchor` has run: correcting every
        candidate first would let a snapped-up sliver outgrow the real door and
        change which one is picked, which is a different decision from the one
        this is meant to make.

        Args:
            box: xyxy box of the door chosen as the scale anchor.
            model: Fitted vertical model; supplies the storey pitch, and the
                projective coordinate the test is applied in so that it means
                the same thing anywhere on the facade.
            base_row: Image row of the facade's foot.

        Returns:
            `(box, snapped_px)` — the box with its foot moved where warranted,
            and by how many pixels.
        """
        x1, top, x2, bottom = (float(v) for v in box)
        gap = float(base_row) - bottom
        if gap <= 0.0 or not self._foot_is_truncated(model, top, bottom, base_row):
            return (x1, top, x2, bottom), 0.0
        return (x1, top, x2, float(base_row)), gap

    def _foot_is_truncated(
        self, model: VerticalModel, top: float, bottom: float, base_row: float
    ) -> bool:
        """Whether a door's foot is short of the ground rather than on it.

        With a storey pitch the test is scale-free: measure the door against
        the storey it sits in, in the projective coordinate, so a door on a
        foreshortened ground floor is judged like one on a flat elevation.
        Without a pitch — a facade too irregular to fit one — fall back to
        bounding the move by the door's own detected height, which finishes
        off a nearly-complete mask and refuses a sliver.
        """
        pitch = model.pitch_t
        if pitch is None or pitch <= 0.0:
            ratio = self.cfg.door_base_snap_ratio
            return ratio > 0.0 and base_row - bottom <= ratio * max(bottom - top, 1.0)
        if model.span(top, bottom) >= self.cfg.door_short_storeys * pitch:
            return False  # already a plausible door; its foot is where it looks
        return model.span(top, base_row) <= self.cfg.door_max_storeys * pitch

    def _facade_extent(
        self,
        windows: list[Detection],
        doors: list[Detection],
        house_bbox: Bbox | None,
    ) -> tuple[float, float]:
        """Top and bottom image rows of the building.

        Prefers the `house`/`facade` mask bbox from Module 1 when available;
        otherwise falls back to the union of window+door detections. The
        former is the correct anchor because a single misdetected window high
        or low in the scene can otherwise inflate the extent — and because it
        is the only thing that sees the parapet and the shopfront band, which
        carry real height but contain no windows.

        Args:
            windows: Window detections.
            doors: Door detections.
            house_bbox: xyxy bbox of the house mask, or `None`.

        Returns:
            `(y_top, y_bottom)`, at least 1 px apart.
        """
        if house_bbox is not None:
            top, bottom = float(house_bbox[1]), float(house_bbox[3])
        else:
            all_boxes = [b for b, _ in windows] + [b for b, _ in doors]
            if not all_boxes:
                return 0.0, 1.0
            top = float(min(b[1] for b in all_boxes))
            bottom = float(max(b[3] for b in all_boxes))
        return top, max(bottom, top + 1.0)

    def _base_row(
        self,
        model: VerticalModel,
        rows: list[float],
        visible_bottom: float,
        bracket: tuple[float, float] | None,
    ) -> float:
        """Place the building's base inside the scene gate's bracket.

        The gate (`scene_parsing`) reports two rows: the lowest wall actually
        seen, and the row where the ground in front of the building starts.
        Where a hedge or a parked car stands in front of the plinth the true
        base is between them — and cutting at the visible edge, which is what
        deleting the occluder leaves behind, measures the building short by
        however tall the hedge is.

        The ground row is the better of the two ends. It sits slightly *below*
        the base, because pavement a few metres nearer the camera images lower
        than the building's own footing, but that offset is a couple of percent
        of facade height where the visible edge can be out by a third. It is
        still only taken as far as the storey rhythm allows, so a run of ground
        pixels far from the building cannot drag the base down with it.

        Returns:
            The image row to use as the facade's lower edge.
        """
        if bracket is None:
            return visible_bottom
        visible, limit = bracket
        bottom = max(visible_bottom, float(visible))
        if limit <= bottom:
            return bottom
        if model.pitch_t is None or model.pitch_t <= 0 or not rows:
            return bottom
        budget_t = model.t(max(rows)) - (
            self.cfg.max_storeys_below_bottom_row * model.pitch_t
        )
        allowed = model.row_at(budget_t)
        return float(min(float(limit), allowed)) if allowed > bottom else bottom

    def _rows_for_model(
        self, windows: list[Detection], floor_lines: list[Detection]
    ) -> tuple[list[float], bool]:
        """Row coordinates to fit the vertical model on.

        Window rows are the primary source: exactly one per storey. The
        `sill`/`balcony`/`cornice`/`molding` classes also mark storey lines but
        sit slightly off the window centres, so mixing them in would double the
        apparent row count and halve the pitch. They are therefore used only
        when the windows alone cannot support a perspective fit.

        Returns:
            `(rows, from_windows)`.
        """
        rows = self._row_centres([w[0] for w in windows])
        if len(rows) >= self.cfg.min_rows_for_perspective or not floor_lines:
            return rows, True
        lines = self._row_centres([f[0] for f in floor_lines])
        return (lines, False) if len(lines) > len(rows) else (rows, True)

    # ----- scale + height ----------------------------------------------------
    def _collect_anchors(
        self,
        model: VerticalModel,
        windows: list[Detection],
        doors: list[Detection],
        extent: tuple[float, float],
    ) -> list[ScaleAnchor]:
        """Build every metric anchor the image supports, strongest first."""
        anchors: list[ScaleAnchor] = []
        chosen = self._pick_door_anchor(doors, extent[1] - extent[0])
        if chosen is not None:
            box, snapped = self._snap_door_to_base(doors[chosen][0], model, extent[1])
            self.last_door_snap_px = snapped
            anchor = door_anchor(
                model,
                box,
                self.cfg.assumed_door_height_m,
                self.cfg.sigma_rel_door,
                snapped_px=snapped,
            )
            if anchor is not None:
                anchors.append(anchor)

        pitch = floor_pitch_anchor(
            model, self.cfg.assumed_floor_pitch_m, self.cfg.sigma_rel_floor_pitch
        )
        if pitch is not None:
            anchors.append(pitch)

        win = window_anchor(
            model,
            [tuple(float(v) for v in w[0]) for w in windows],
            self.cfg.assumed_window_height_m,
            self.cfg.sigma_rel_window,
        )
        if win is not None:
            anchors.append(win)
        return anchors

    def _plausible_pitch(self, model: VerticalModel, metres_per_t: float) -> bool:
        """True when the implied storey pitch sits inside the configured range.

        The check is applied to the *pitch*, not to `height / floor_count`:
        the facade extent legitimately exceeds `floor_count * pitch` on these
        blocks, because the shopfront storey is taller and the parapet is not
        a storey at all.
        """
        if model.pitch_t is None or model.pitch_t <= 0:
            return True
        lo, hi = self.cfg.plausible_floor_height_m
        return lo <= metres_per_t * model.pitch_t <= hi

    def _plausible_height(
        self, height_m: float, floor_count: int, pitch_m: float
    ) -> bool:
        """True when a measured height is consistent with the storeys counted.

        Stated as a *storey budget* rather than as a range of metres: the
        facade may legitimately exceed `floor_count * pitch` — a taller
        shopfront storey, a parapet, a setback penthouse — but only by the
        same couple of storeys the vertical model is allowed to extrapolate.
        Expressing the bound in metres instead would multiply that allowance
        by the maximum plausible pitch as well, leaving it too slack to catch
        anything.

        It catches the two ways the measurement goes wrong on this dataset: a
        facade mask that swallowed a taller neighbour, and a building whose
        only "window rows" are something else entirely — a glazed stair tower,
        a painted gable — where the storey pitch is not a storey pitch at all.
        """
        if floor_count <= 0:
            return True
        lo, _ = self.cfg.plausible_floor_height_m
        if height_m < floor_count * lo:
            return False
        budget = (
            floor_count
            + self.cfg.max_storeys_above_top_row
            + self.cfg.max_storeys_below_bottom_row
        )
        return height_m / max(pitch_m, lo) <= budget

    def _estimate_height(
        self,
        model: VerticalModel,
        anchors: list[ScaleAnchor],
        extent: tuple[float, float],
        floor_count: int,
    ) -> tuple[float, float, HeightSource, FusedScale | None]:
        """Convert the facade extent to metres through the fused scale.

        Args:
            model: Fitted vertical model.
            anchors: Available metric anchors.
            extent: `(y_top, y_bottom)` of the facade.
            floor_count: Output of `_estimate_floor_count`, for the fallback.

        Returns:
            `(height_m, confidence, height_source, fused_scale)`.
        """
        fallback = (
            (round(floor_count * self.cfg.assumed_floor_height_m, 1), 0.45, "fallback")
            if floor_count > 0
            else (0.0, 0.0, "none")
        )

        fused = fuse(anchors, self.cfg.scale_agreement_log_tol)
        if fused is None or not self._plausible_pitch(model, fused.metres_per_t):
            return (*fallback, None)

        height_m = fused.metres_per_t * model.span(*extent)
        if not np.isfinite(height_m) or height_m <= 0:
            return (*fallback, None)
        pitch_m = (
            fused.metres_per_t * model.pitch_t
            if model.pitch_t is not None and model.pitch_t > 0
            else self.cfg.assumed_floor_pitch_m
        )
        if not self._plausible_height(float(height_m), floor_count, float(pitch_m)):
            return (*fallback, None)

        source: HeightSource = (
            _SINGLE_SOURCE.get(fused.used[0], "fused") if len(fused.used) == 1 else "fused"
        )
        confidence = float(np.clip(0.95 - 2.0 * fused.sigma_rel, 0.1, 0.95))
        return round(float(height_m), 1), round(confidence, 4), source, fused

    def _scale_report(
        self,
        model: VerticalModel,
        anchors: list[ScaleAnchor],
        fused: FusedScale | None,
        extent: tuple[float, float],
    ) -> ScaleReport | None:
        """Assemble the human-inspectable record of how metres were derived."""
        if fused is None:
            return None
        base_row = extent[1]
        pitch_m = (
            round(fused.metres_per_t * model.pitch_t, 2)
            if model.pitch_t is not None and model.pitch_t > 0
            else None
        )
        return ScaleReport(
            metres_per_pixel_at_base=round(
                model.metres_per_pixel(fused.metres_per_t, base_row), 6
            ),
            sigma_rel=round(fused.sigma_rel, 4),
            perspective_corrected=model.perspective_corrected,
            vertical_vanishing_point_y=(
                round(model.y_vp, 1) if model.y_vp is not None else None
            ),
            storey_pitch_m=pitch_m,
            storey_pitch_residual=(
                round(model.residual, 4) if np.isfinite(model.residual) else None
            ),
            facade_span_px=round(extent[1] - extent[0], 1),
            rows_detected=model.row_count,
            anchors=[
                ScaleAnchorInfo(
                    source=a.source,
                    reference_m=a.detail.get("reference_m", 0.0),
                    metres_per_pixel_at_base=round(
                        model.metres_per_pixel(a.metres_per_t, base_row), 6
                    ),
                    sigma_rel=round(a.sigma_rel, 4),
                    used=a.source in fused.used,
                    detail={k: v for k, v in a.detail.items() if k != "reference_m"},
                )
                for a in anchors
            ],
        )

    # ----- public API --------------------------------------------------------
    def extract(
        self,
        windows: list[Detection],
        doors: list[Detection],
        house_bbox: Bbox | None = None,
        floor_lines: list[Detection] | None = None,
        base_bracket: tuple[float, float] | None = None,
    ) -> BuildingFeaturesResult:
        """Run the estimation and return a structured result.

        Args:
            windows: Window detections from Module 1.
            doors: Door detections from Module 1.
            house_bbox: Optional house/facade mask xyxy bbox; strongly
                recommended — it is what makes the shopfront storey and the
                parapet count toward the height.
            floor_lines: Optional detections from Module 1's storey-line
                classes (`sill`, `balcony`, `cornice`, `molding`), used as a
                backup source of rows when windows are too few.
            base_bracket: Optional `(visible, limit)` rows from the scene gate
                bounding where the building's base really is. Without it an
                occluded base measures the building short by the height of
                whatever stands in front of it.

        Returns:
            `BuildingFeaturesResult` with floor count, height (m),
            `height_source`, and a `scale` report naming the cues used.
        """
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        with Image.open(self.image_path) as im:
            width_px, height_px = im.size

        self.last_detection_counts = {"window": len(windows), "door": len(doors)}

        extent = self._facade_extent(windows, doors, house_bbox)
        rows, from_windows = self._rows_for_model(windows, floor_lines or [])

        def fit(ext: tuple[float, float]) -> VerticalModel:
            return fit_vertical_model(
                rows,
                extent=ext,
                max_scale_ratio=self.cfg.max_vertical_scale_ratio,
                max_extra_top=self.cfg.max_storeys_above_top_row,
                max_extra_bottom=self.cfg.max_storeys_below_bottom_row,
                min_rows=self.cfg.min_rows_for_perspective,
                improvement=self.cfg.perspective_improvement,
            )

        # Fit once on the certainly-building extent to obtain a storey pitch,
        # then re-fit if that pitch lets the base move down through an
        # occluder. The first fit only supplies the pitch, so the extension
        # cannot bootstrap itself off an extent it has already stretched.
        model = fit(extent)
        base = self._base_row(model, rows, extent[1], base_bracket)
        if base > extent[1]:
            extent = (extent[0], base)
            model = fit(extent)
        floor_count, floor_conf = self._estimate_floor_count(
            windows, model, from_windows
        )
        self.last_door_snap_px = 0.0
        anchors = self._collect_anchors(model, windows, doors, extent)
        height_m, height_conf, height_source, fused = self._estimate_height(
            model, anchors, extent, floor_count
        )

        return BuildingFeaturesResult(
            image=ImageInfo(id=self.image_path.stem, width=width_px, height=height_px),
            view_type=self.view_type,
            predictions=Predictions(
                building_height_m=FloatPrediction(value=height_m, confidence=height_conf),
                floor_count=IntPrediction(value=floor_count, confidence=floor_conf),
            ),
            height_source=height_source,
            scale=self._scale_report(model, anchors, fused, extent),
            metadata=Metadata(
                model_version=self.cfg.model_version,
                timestamp=datetime.now(timezone.utc),
            ),
        )
