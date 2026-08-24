"""Region-restricted patch sampling shared by the Module 3 material backends.

Contract — *classify ONLY the red building region*:
  The building material is estimated strictly from Module 1's building region
  (``house``/``facade`` polygons minus the dilated ``window``/``door``
  openings). Non-region pixels are neutralised (filled with the region's mean
  colour) BEFORE any patch reaches the encoder, so sky, ground, neighbouring
  buildings and window glass can never influence the material decision.

  There is deliberately NO fallback to the whole image. When Module 1 finds no
  building, this returns ``([], "none")`` and the caller MUST fail closed
  (report "no building region") rather than classifying the full frame — the
  behaviour the previous implementation silently did via three separate
  whole-image fallbacks.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

# How the classified pixels were obtained, recorded on the result so callers
# can tell a real wall read from a degraded one (never a whole-image read).
RegionSource = Literal["patches", "masked_bbox", "none"]


def neutralise_outside(rgb: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Copy `rgb` with every pixel outside boolean `inside` set to the region mean.

    Filling with the region's own mean colour (rather than black) avoids hard
    edges that would otherwise read as a spurious dark material, while still
    guaranteeing no real non-region content reaches the encoder.
    """
    out = rgb.copy()
    if not inside.any():
        return out
    fill = rgb[inside].reshape(-1, rgb.shape[-1]).mean(axis=0)
    out[~inside] = fill.astype(out.dtype)
    return out


def sample_region_patches(
    rgb: np.ndarray,
    region_mask: np.ndarray | None,
    patch_size: int,
    max_patches: int,
    min_wall_ratio: float,
) -> tuple[list[np.ndarray], RegionSource]:
    """Sample square wall patches strictly inside `region_mask`.

    Args:
        rgb: ``[H, W, 3]`` image.
        region_mask: ``[H, W]`` mask (>0 = building wall) from Module 1, or None.
        patch_size: side length of square patches, in pixels.
        max_patches: cap on the number of patches returned.
        min_wall_ratio: minimum fraction of wall pixels a patch must contain.

    Returns:
        ``(patches, source)`` where source is:
          * ``"patches"``     — up to `max_patches` `patch_size`-side crops whose
            wall coverage >= `min_wall_ratio` (purest first), from the
            region-masked image.
          * ``"masked_bbox"`` — a single region-bbox crop (non-region pixels
            neutralised) when the region is smaller than one patch or too
            fragmented for any patch to clear the ratio.
          * ``"none"``        — `region_mask` is None/empty. The caller MUST NOT
            classify the whole image; it should fail closed.
    """
    if region_mask is None:
        return [], "none"
    m = region_mask > 0
    if not m.any():
        return [], "none"

    # Neutralise everything outside the wall ONCE; all crops below are taken
    # from this masked image, so the encoder only ever sees building pixels.
    filled = neutralise_outside(rgb, m)
    ys, xs = m.nonzero()
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    ps = int(patch_size)

    if (x1 - x0) >= ps and (y1 - y0) >= ps:
        step = max(ps // 2, 1)
        # Configured purity first, then a relaxed pass, before the masked-bbox
        # fallback — every option stays strictly inside the building region.
        for thr in (float(min_wall_ratio), float(min_wall_ratio) * 0.6):
            cand: list[tuple[float, int, int]] = []
            for yy in range(y0, y1 - ps + 1, step):
                for xx in range(x0, x1 - ps + 1, step):
                    ratio = float(m[yy:yy + ps, xx:xx + ps].mean())
                    if ratio >= thr:
                        cand.append((ratio, xx, yy))
            if cand:
                cand.sort(reverse=True)  # purest-wall patches first
                return (
                    [filled[yy:yy + ps, xx:xx + ps]
                     for _, xx, yy in cand[: int(max_patches)]],
                    "patches",
                )

    crop = filled[y0:y1, x0:x1]
    return ([crop], "masked_bbox") if crop.size else ([], "none")
