"""Repair Module 1's facade mask using the parsed street scene.

Module 1 hands back a mask that reaches wherever the frame stops being sky, so
on these captures it routinely contains the oleander hedge, the parked cars,
the pavement and the neighbouring wing. Module 2's height is
`metres_per_t x the facade mask's t-span`, which makes the mask's top and
bottom edges load-bearing — so a blob costs height directly.

The repair is an intersection, not a re-segmentation: keep only what Cityscapes
also calls building. Two details matter more than the intersection itself.

**Occluders raise the lower edge, they do not reveal the base.** A hedge in
front of a plinth is building *behind* vegetation; deleting it leaves the mask
ending at the top of the hedge, above the true base. So the base is taken as a
high percentile of the per-column lower edges — which favours the columns that
see furthest down, i.e. the unoccluded ones — and the fraction of columns whose
base is still hidden is reported rather than papered over.

**A neighbour at the same depth cannot be separated here.** Both wings are
genuinely `building` to any segmenter; telling them apart is a depth question,
not a semantic one. Splitting the mask into connected components removes a
detached neighbour but not an abutting one, and the residual is left visible in
`components_kept` instead of being silently absorbed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .parser import SceneMasks

Bbox = tuple[float, float, float, float]


@dataclass(frozen=True)
class BuildingRegion:
    """Module 1's facade mask after the scene gate, with what it cost recorded.

    Attributes:
        mask: `[H, W]` uint8 refined building mask (1 = building surface).
        bbox: Refined xyxy extent, or `None` when nothing survived.
        base_occluded_ratio: Fraction of the region's columns whose lower edge
            is still hidden behind an occluder — the extent's bottom is a lower
            bound by roughly that much.
        kept_ratio: Fraction of Module 1's facade mask that survived.
        removed: Fraction of Module 1's facade mask each scene group took away.
        components_kept: Connected components retained; >1 means the region is
            split (a tree cutting across) or holds an abutting neighbour.
        refined: False when the gate was skipped because it would have deleted
            essentially everything (an unusable scene parse), leaving Module 1's
            mask untouched.
    """

    mask: np.ndarray
    bbox: Bbox | None
    base_visible_row: float | None = None
    base_limit_row: float | None = None
    base_occluded_ratio: float = 0.0
    kept_ratio: float = 1.0
    removed: dict[str, float] = field(default_factory=dict)
    components_kept: int = 0
    refined: bool = True

    def base_bracket(self) -> tuple[float, float] | None:
        """`(visible, limit)` rows bounding the building's true base.

        `visible` is the lowest wall actually seen; `limit` is where the ground
        in front of it starts. They coincide when nothing occludes the base.
        Consumers pick within the bracket — Module 2 does so with the storey
        pitch — rather than treating either end as the answer.
        """
        if self.base_visible_row is None or self.base_limit_row is None:
            return None
        return self.base_visible_row, max(self.base_limit_row, self.base_visible_row)


def _largest_components(mask: np.ndarray, min_ratio: float) -> tuple[np.ndarray, int]:
    """Keep components at least `min_ratio` of the largest one.

    Not just the single largest: a street tree crossing a facade splits it into
    two legitimate halves, and dropping the smaller one would shorten the
    building. A detached blob elsewhere in the frame falls below the ratio.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask, 0
    keep_ids = [i + 1 for i, a in enumerate(areas) if a >= areas.max() * min_ratio]
    return np.isin(labels, keep_ids).astype(np.uint8), len(keep_ids)


def _column_edges(mask: np.ndarray, min_pixels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column first and last set row, for columns with enough pixels.

    Returns:
        `(columns, top_rows, bottom_rows)`, all empty when nothing qualifies.
    """
    counts = mask.sum(axis=0)
    cols = np.flatnonzero(counts >= max(min_pixels, 1))
    if cols.size == 0:
        return cols, cols, cols
    rows = np.arange(mask.shape[0])[:, None]
    sub = mask[:, cols].astype(bool)
    tops = np.where(sub, rows, mask.shape[0]).min(axis=0)
    bottoms = np.where(sub, rows, -1).max(axis=0)
    return cols, tops.astype(float), bottoms.astype(float)


def _base_bracket(
    scene: SceneMasks,
    cols: np.ndarray,
    bottoms: np.ndarray,
    probe_px: int,
) -> tuple[np.ndarray, float]:
    """Bracket each column's true base between what is visible and the ground.

    A column's last building pixel is only the base if nothing is standing in
    front of the building there. Where a hedge or a car intervenes, the base is
    hidden *between* two things that can be seen:

    * the last visible building row — the base is at or below it, since the
      wall continues behind the occluder;
    * the first ground row below that — the base is at or above it, since the
      pavement in front of the building images lower than the building's own
      footing.

    Returning the bracket rather than one edge keeps the choice within it
    honest: a decision made here would be a guess, whereas Module 2 has the
    storey rhythm to place the base inside these bounds.

    Returns:
        `(limits, occluded_ratio)` — `limits[i]` is the lowest row column `i`'s
        base could occupy (its own bottom when nothing occludes it).
    """
    height = scene.height
    occluder = scene.group("occluder")
    ground = scene.group("ground")
    limits = bottoms.astype(float).copy()
    hidden = 0
    for i, (col, bottom) in enumerate(zip(cols, bottoms)):
        x = int(col)
        start = int(bottom) + 1
        if start >= height:
            continue
        stop = min(height, start + max(probe_px, 1))
        if not occluder[start:stop, x].any():
            continue  # pavement or road right below: this column found the base.
        hidden += 1
        below_ground = np.flatnonzero(ground[start:, x])
        if below_ground.size:
            limits[i] = float(start + below_ground[0])
        else:
            limits[i] = float(height - 1)
    return limits, (hidden / float(cols.size) if cols.size else 0.0)


def refine_building_region(
    facade_mask: np.ndarray,
    scene: SceneMasks,
    cfg,
) -> BuildingRegion:
    """Intersect Module 1's facade mask with the scene's building surface.

    Args:
        facade_mask: `[H, W]` uint8 mask from Module 1 (`facade` / `house`).
        scene: Parsed Cityscapes groups for the same image.
        cfg: `SceneParsingConfig`.

    Returns:
        A `BuildingRegion`. When the gate would delete all but
        `cfg.min_kept_ratio` of the input the original mask is returned with
        `refined=False`, so a bad scene parse degrades to today's behaviour
        rather than emptying the pipeline.
    """
    original = (np.asarray(facade_mask) > 0).astype(np.uint8)
    total = float(original.sum())
    if total <= 0:
        return BuildingRegion(mask=original, bbox=None, kept_ratio=0.0, refined=False)

    removed = {
        name: float((original & scene.group(name)).sum()) / total
        for name in ("occluder", "ground", "sky")
    }

    kept = (original & scene.group("building")).astype(np.uint8)
    if kept.sum() / total < cfg.min_kept_ratio:
        # Cityscapes disagrees with Module 1 about almost the whole mask; more
        # likely a bad scene parse than a building that is not a building.
        return BuildingRegion(
            mask=original,
            bbox=_bbox_of(original),
            kept_ratio=float(kept.sum()) / total,
            removed=removed,
            refined=False,
        )

    if cfg.morph_kernel_px > 0:
        k = np.ones((cfg.morph_kernel_px, cfg.morph_kernel_px), np.uint8)
        kept = cv2.morphologyEx(kept, cv2.MORPH_OPEN, k)
        kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, k)

    kept, components = _largest_components(kept, cfg.min_component_ratio)
    if kept.sum() <= 0:
        return BuildingRegion(
            mask=original, bbox=_bbox_of(original), kept_ratio=0.0,
            removed=removed, refined=False,
        )

    min_pixels = int(round(cfg.min_column_pixels_ratio * kept.shape[0]))
    cols, tops, bottoms = _column_edges(kept, min_pixels)
    if cols.size == 0:
        return BuildingRegion(
            mask=kept, bbox=_bbox_of(kept),
            kept_ratio=round(float(kept.sum()) / total, 4),
            removed={k: round(v, 4) for k, v in removed.items()},
            components_kept=components, refined=True,
        )

    # Percentiles rather than min/max: the deepest columns are the ones that
    # actually reached the base, and a lone stray row should not set the
    # extent in either direction.
    edge = cfg.edge_percentile
    y_top = float(np.percentile(tops, edge))
    y_visible = float(np.percentile(bottoms, 100.0 - edge))
    limits, occluded = _base_bracket(
        scene, cols, bottoms, int(round(cfg.base_probe_ratio * kept.shape[0]))
    )
    y_limit = float(np.percentile(limits, 100.0 - edge))

    return BuildingRegion(
        mask=kept,
        # The bbox bottom stays at the visible wall: it is the one end of the
        # bracket that is certainly building. Module 2 moves it down within
        # `base_bracket()` once it knows the storey pitch.
        bbox=(float(cols.min()), y_top, float(cols.max()), max(y_visible, y_top + 1.0)),
        base_visible_row=y_visible,
        base_limit_row=max(y_limit, y_visible),
        base_occluded_ratio=round(occluded, 4),
        kept_ratio=round(float(kept.sum()) / total, 4),
        removed={k: round(v, 4) for k, v in removed.items()},
        components_kept=components,
        refined=True,
    )


def _bbox_of(mask: np.ndarray) -> Bbox | None:
    """Plain xyxy extent of a binary mask."""
    ys, xs = np.asarray(mask).nonzero()
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
