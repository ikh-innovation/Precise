"""Module 2 — building feature extraction (floor count + height)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from config import BuildingFeaturesConfig
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


Detection = tuple[np.ndarray, float]
Bbox = tuple[float, float, float, float]


class BuildingFeatures:
    """Estimate floor count and building height from window/door bboxes."""

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

    def _cluster_rows(self, boxes: list[np.ndarray]) -> list[list[int]]:
        """Greedy 1-D clustering of box Y-centers into horizontal rows.

        Args:
            boxes: List of xyxy boxes.

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
        return rows

    def _estimate_floor_count(
        self, windows: list[Detection]
    ) -> tuple[int, float]:
        """Compute floor count and a row-regularity confidence.

        Args:
            windows: List of window detections.

        Returns:
            `(floor_count, confidence)`.
        """
        if not windows:
            return 0, 0.0
        boxes = [w[0] for w in windows]
        rows = self._cluster_rows(boxes)
        n_floors = len(rows)
        if n_floors == 0:
            return 0, 0.0
        counts = np.array([len(r) for r in rows], dtype=float)
        mean_count = float(counts.mean())
        if mean_count <= 0:
            return n_floors, 0.3
        std_count = float(counts.std())
        regularity = max(0.0, 1.0 - (std_count / (mean_count + 1e-6)))
        mean_score = float(np.mean([w[1] for w in windows]))
        confidence = float(np.clip(0.5 * regularity + 0.5 * mean_score, 0.0, 1.0))
        return n_floors, round(confidence, 4)

    def _pick_door_anchor(
        self, doors: list[Detection], building_pixel_h: float
    ) -> Detection | None:
        """Pick the front door used as the metric scale anchor.

        Front-door heuristic: among detected doors, prefer those whose
        bottom edge sits near the bottom of the building (within
        `bottom_door_tol_ratio` of the building's pixel height), and from
        that subset pick the tallest one. Falls back to the overall
        tallest door if no door sits near the bottom.

        Args:
            doors: List of door detections.
            building_pixel_h: Pixel height of the building region.

        Returns:
            The chosen `(bbox, score)`, or `None` if `doors` is empty.
        """
        if not doors:
            return None
        bottom_y = max(float(b[3]) for b, _ in doors)
        tol = max(building_pixel_h * self.cfg.bottom_door_tol_ratio, 5.0)
        bottom_doors = [d for d in doors if bottom_y - float(d[0][3]) <= tol]
        candidates = bottom_doors or doors
        return max(candidates, key=lambda d: float(d[0][3] - d[0][1]))

    def _building_pixel_height(
        self,
        windows: list[Detection],
        doors: list[Detection],
        house_bbox: Bbox | None,
    ) -> float:
        """Vertical pixel extent of the building.

        Prefers the `house` mask bbox from Module 1 when available; otherwise
        falls back to the union of window+door detections. The former is the
        correct anchor because a single misdetected window high or low in the
        scene can otherwise inflate the extent.

        Args:
            windows: Window detections.
            doors: Door detections.
            house_bbox: xyxy bbox of the house mask, or `None`.

        Returns:
            Pixel height, at least 1.0.
        """
        if house_bbox is not None:
            return max(float(house_bbox[3] - house_bbox[1]), 1.0)
        all_boxes = [b for b, _ in windows] + [b for b, _ in doors]
        if not all_boxes:
            return 1.0
        ys_top = np.array([b[1] for b in all_boxes])
        ys_bot = np.array([b[3] for b in all_boxes])
        return max(float(ys_bot.max() - ys_top.min()), 1.0)

    def _estimate_height(
        self,
        windows: list[Detection],
        doors: list[Detection],
        floor_count: int,
        house_bbox: Bbox | None,
    ) -> tuple[float, float, HeightSource]:
        """Estimate building height in meters with a sanity-check fallback.

        Door-scaled height is accepted only when `height_m / floor_count`
        sits inside the plausible per-floor range from config. Outside it,
        the result falls back to `floor_count * assumed_floor_height_m`
        and `height_source` is reported as `"fallback"`.

        Args:
            windows: Window detections.
            doors: Door detections.
            floor_count: Output of `_estimate_floor_count`.
            house_bbox: Optional house mask xyxy bbox.

        Returns:
            `(height_m, confidence, height_source)`.
        """
        building_pixel_h = self._building_pixel_height(windows, doors, house_bbox)
        anchor = self._pick_door_anchor(doors, building_pixel_h)

        if anchor is not None:
            door_box, door_score = anchor
            door_pixel_h = float(door_box[3] - door_box[1])
            if door_pixel_h > 0:
                scale_m_per_px = self.cfg.assumed_door_height_m / door_pixel_h
                height_m = building_pixel_h * scale_m_per_px
                lo, hi = self.cfg.plausible_floor_height_m
                if floor_count > 0:
                    per_floor = height_m / floor_count
                    if not (lo <= per_floor <= hi):
                        fb = floor_count * self.cfg.assumed_floor_height_m
                        return round(fb, 1), 0.45, "fallback"
                confidence = float(np.clip(0.55 + 0.35 * door_score, 0.0, 0.95))
                return round(height_m, 1), round(confidence, 4), "door_scale"

        if floor_count > 0:
            return round(floor_count * self.cfg.assumed_floor_height_m, 1), 0.45, "fallback"

        return 0.0, 0.0, "none"

    def extract(
        self,
        windows: list[Detection],
        doors: list[Detection],
        house_bbox: Bbox | None = None,
    ) -> BuildingFeaturesResult:
        """Run the heuristic estimation and return a structured result.

        Args:
            windows: Window detections from Module 1.
            doors: Door detections from Module 1.
            house_bbox: Optional house mask xyxy bbox; strongly recommended.

        Returns:
            `BuildingFeaturesResult` with floor count, height (m), and
            `height_source` flag.
        """
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        with Image.open(self.image_path) as im:
            width_px, height_px = im.size

        self.last_detection_counts = {"window": len(windows), "door": len(doors)}

        floor_count, floor_conf = self._estimate_floor_count(windows)
        height_m, height_conf, height_source = self._estimate_height(
            windows, doors, floor_count, house_bbox
        )

        return BuildingFeaturesResult(
            image=ImageInfo(id=self.image_path.stem, width=width_px, height=height_px),
            view_type=self.view_type,
            predictions=Predictions(
                building_height_m=FloatPrediction(value=height_m, confidence=height_conf),
                floor_count=IntPrediction(value=floor_count, confidence=floor_conf),
            ),
            height_source=height_source,
            metadata=Metadata(
                model_version=self.cfg.model_version,
                timestamp=datetime.now(timezone.utc),
            ),
        )
