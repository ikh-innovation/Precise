"""Warp a street-level facade to fronto-parallel, to ask the model again.

The CMP checkpoint was trained on rectified, head-on facade crops. These
images are Google Street View screenshots taken from across a road, so the
facade is both tilted back (the camera looks up) and turned away (the camera
stands off to one side). How much that costs is measurable rather than a
guess: warping the held-out CMP split by a known homography and re-scoring
the same checkpoint gives

    warp                     mIoU     door IoU
    none (as trained)        0.584    0.523
    look-up      k_v = 0.05  0.574    0.501
    look-up      k_v = 0.20  0.499    0.436
    oblique      k_h = 0.10  0.540    0.362      <- door -31%
    look-up + oblique        0.519    0.387

so it is *obliquity*, not looking up, that the door class cannot survive —
which is the one class Module 2 takes its metric scale from.

**Estimating the warp.** Line segments (LSD) are restricted to Module 1's own
facade mask, because on a street capture the strongest line families belong
to the road, the kerb and the neighbouring building receding at a different
angle, and a vanishing point fitted to those rectifies the street instead of
the building. Two vanishing points are fitted by RANSAC, one per family. They
are the images of two orthogonal world directions, so with a principal point
at the image centre and square pixels they give the focal length,

    f^2 = -(v_h - p) . (v_v - p)

and from there a rotation whose columns are those directions, and the metric
rectification `H = K R^T K^-1`. Affine rectification — sending the vanishing
line to infinity — was tried first and is not usable: it is numerically
violent and turned package 19 into a wedge even when its vanishing points
were sound.

**Why the result is offered rather than adopted.** Where the estimate is good
the gain is decisive: package 20's gated entrance, which the plain pass calls
shopfront at p(door) = 0.02 and no threshold, prior correction or resolution
change could reach, comes back at p(door) = 0.80 and wins its argmax outright.
But across the ten packages that pass the guards the plain and rectified
passes disagree in both directions — packages 8, 14 and 20 gain, packages 5,
10, 18 and 19 lose most of their door — and mean confidence drops 2.5%. The
checkpoint is simply unstable under input changes (swapping the resize filter
alone moves one p(door) by 9x).

So the caller merges *additively*: door evidence is taken from the rectified
pass only where it is confident and the plain pass had nothing better to say,
and nothing is ever removed on the rectified pass's word. Under that policy
every one of the regressions above is a no-op and only the gains survive. The
durable fix is to train with perspective augmentation instead — the warps at
the top of this docstring are the recipe — at which point this module becomes
unnecessary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class Rectification:
    """The estimated warp, or the reason there isn't one.

    Attributes:
        homography: `3x3` image-to-rectified warp, or `None`.
        focal_px: Focal length recovered from the two vanishing points.
        rotation_deg: How far the warp tips the vertical and horizontal axes;
            a sane rectification barely rotates the frame.
        anisotropy: Ratio of the x and y scales the warp implies.
        reason: Why no homography was produced, when there is none.
        detail: Segment and inlier counts, for diagnosis.
    """

    homography: np.ndarray | None = None
    focal_px: float | None = None
    rotation_deg: float | None = None
    anisotropy: float | None = None
    reason: str = ""
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when a usable homography was found."""
        return self.homography is not None


def _segments(gray: np.ndarray, mask: np.ndarray | None, min_len: float) -> np.ndarray:
    """LSD segments whose midpoint lies on `mask`, longer than `min_len`."""
    lines = cv2.createLineSegmentDetector().detect(gray)[0]
    if lines is None:
        return np.empty((0, 4), np.float32)
    seg = lines.reshape(-1, 4)
    keep = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) >= min_len
    seg = seg[keep]
    if mask is None or not len(seg):
        return seg
    h, w = mask.shape
    mx = np.clip(((seg[:, 0] + seg[:, 2]) / 2).astype(int), 0, w - 1)
    my = np.clip(((seg[:, 1] + seg[:, 3]) / 2).astype(int), 0, h - 1)
    return seg[mask[my, mx] > 0]


def _split(seg: np.ndarray, tol_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Partition segments into near-vertical and near-horizontal families."""
    ang = np.degrees(
        np.arctan2(np.abs(seg[:, 3] - seg[:, 1]), np.abs(seg[:, 2] - seg[:, 0]))
    )
    return seg[ang >= 90.0 - tol_deg], seg[ang <= tol_deg]


def _vanishing_point(
    seg: np.ndarray, iters: int, tol: float, seed: int
) -> tuple[np.ndarray, int] | None:
    """RANSAC vanishing point of one line family, refined on its inliers.

    Scored by inlier *length* rather than count, so a facade's few long storey
    lines outweigh a scatter of short texture edges.
    """
    if len(seg) < 4:
        return None
    rng = np.random.default_rng(seed)
    p1 = np.c_[seg[:, 0], seg[:, 1], np.ones(len(seg))]
    p2 = np.c_[seg[:, 2], seg[:, 3], np.ones(len(seg))]
    lines = np.cross(p1, p2)
    lines /= np.maximum(np.linalg.norm(lines[:, :2], axis=1, keepdims=True), 1e-9)
    mid = np.c_[(seg[:, 0] + seg[:, 2]) / 2, (seg[:, 1] + seg[:, 3]) / 2, np.ones(len(seg))]
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])

    def errors(v: np.ndarray) -> np.ndarray:
        through = np.cross(mid, v[None, :])
        through /= np.maximum(np.linalg.norm(through[:, :2], axis=1, keepdims=True), 1e-9)
        return 1.0 - np.abs(np.sum(through[:, :2] * lines[:, :2], axis=1))

    best, best_score = None, -1.0
    for _ in range(iters):
        i, j = rng.choice(len(seg), 2, replace=False)
        v = np.cross(lines[i], lines[j])
        if abs(v[2]) < 1e-12:
            continue
        score = float(length[errors(v) < tol].sum())
        if score > best_score:
            best, best_score = v, score
    if best is None:
        return None
    inliers = errors(best) < tol
    if int(inliers.sum()) < 4:
        return None
    _, _, vt = np.linalg.svd(lines[inliers] * length[inliers, None])
    return vt[-1], int(inliers.sum())


def estimate(bgr: np.ndarray, facade_mask: np.ndarray | None, cfg) -> Rectification:
    """Estimate the warp taking this facade to fronto-parallel.

    Args:
        bgr: The image, BGR uint8.
        facade_mask: Module 1's facade mask. Lines are taken only from here;
            without it the road and the neighbouring building dominate.
        cfg: `FacadeParsingSegformerConfig`.

    Returns:
        A `Rectification`, whose `homography` is `None` whenever the estimate
        fails a guard. Every guard is a case that was observed to produce a
        warp worse than no warp at all.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    seg = _segments(gray, facade_mask, max(20.0, cfg.rect_min_segment_ratio * max(h, w)))
    vert, horiz = _split(seg, cfg.rect_angle_tol_deg)
    detail = {"segments": float(len(seg)), "vertical": float(len(vert)),
              "horizontal": float(len(horiz))}

    rv = _vanishing_point(vert, cfg.rect_ransac_iters, cfg.rect_inlier_tol, cfg.rect_seed)
    rh = _vanishing_point(horiz, cfg.rect_ransac_iters, cfg.rect_inlier_tol, cfg.rect_seed)
    if rv is None or rh is None:
        return Rectification(reason="no vanishing point", detail=detail)
    vv, nv = rv
    vh, nh = rh
    detail.update(inliers_vertical=float(nv), inliers_horizontal=float(nh))
    if abs(vv[2]) < 1e-12 or abs(vh[2]) < 1e-12:
        return Rectification(reason="vanishing point at infinity", detail=detail)

    v_h = np.array([vh[0] / vh[2], vh[1] / vh[2]])
    v_v = np.array([vv[0] / vv[2], vv[1] / vv[2]])
    centre = np.array([w / 2.0, h / 2.0])
    # The two directions are orthogonal in the world, which fixes the focal
    # length; a negative f^2 means they are not, i.e. one of them is wrong.
    f_sq = -float(np.dot(v_h - centre, v_v - centre))
    if f_sq <= 0.0:
        return Rectification(reason="vanishing points not orthogonal", detail=detail)
    focal = float(np.sqrt(f_sq))
    if not (cfg.rect_min_focal_ratio * max(h, w) < focal < cfg.rect_max_focal_ratio * max(h, w)):
        return Rectification(reason=f"implausible focal ({focal:.0f} px)",
                             focal_px=focal, detail=detail)

    k = np.array([[focal, 0, centre[0]], [0, focal, centre[1]], [0, 0, 1.0]])
    k_inv = np.linalg.inv(k)
    r1 = k_inv @ np.r_[v_h, 1.0]
    r1 /= np.linalg.norm(r1)
    r2 = k_inv @ np.r_[v_v, 1.0]
    r2 /= np.linalg.norm(r2)
    # Orient the facade's own axes: x to the right, y down. Without this the
    # rotation can come out as a reflection or a quarter turn, which passes
    # every numeric check and renders the building mirrored or on its side.
    if r1[0] < 0:
        r1 = -r1
    if r2[1] < 0:
        r2 = -r2
    r2 -= np.dot(r1, r2) * r1
    r2 /= np.linalg.norm(r2)
    rot = np.stack([r1, r2, np.cross(r1, r2)], axis=1)
    if np.linalg.det(rot) < 0:
        rot[:, 2] *= -1
    homography = k @ rot.T @ k_inv

    probe = np.float32([[centre[0], centre[1]], [centre[0], centre[1] + 50],
                        [centre[0] + 50, centre[1]]]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(probe, homography).reshape(-1, 2)
    if not np.isfinite(moved).all():
        return Rectification(reason="non-finite warp", focal_px=focal, detail=detail)
    d_v, d_h = moved[1] - moved[0], moved[2] - moved[0]
    rotation = max(abs(float(np.degrees(np.arctan2(d_v[0], d_v[1])))),
                   abs(float(np.degrees(np.arctan2(d_h[1], d_h[0])))))
    if rotation > cfg.rect_max_rotation_deg:
        return Rectification(reason=f"rotates {rotation:.0f} deg", focal_px=focal,
                             rotation_deg=rotation, detail=detail)

    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    box = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    if not np.isfinite(box).all():
        return Rectification(reason="non-finite corners", focal_px=focal, detail=detail)
    x0, y0 = box.min(0)
    x1, y1 = box.max(0)
    sx, sy = w / max(x1 - x0, 1e-9), h / max(y1 - y0, 1e-9)
    anisotropy = float(max(sx, sy) / max(min(sx, sy), 1e-9))
    if anisotropy > cfg.rect_max_anisotropy:
        return Rectification(reason=f"anisotropy {anisotropy:.2f}", focal_px=focal,
                             rotation_deg=rotation, anisotropy=anisotropy, detail=detail)

    scale = min(sx, sy)
    fit = np.array([[scale, 0.0, -x0 * scale + (w - (x1 - x0) * scale) / 2.0],
                    [0.0, scale, -y0 * scale + (h - (y1 - y0) * scale) / 2.0],
                    [0.0, 0.0, 1.0]])
    return Rectification(homography=fit @ homography, focal_px=focal,
                         rotation_deg=rotation, anisotropy=anisotropy, detail=detail)


def warp_for_inference(
    bgr: np.ndarray, homography: np.ndarray
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Warp, then crop away the pixels the warp invented.

    Outside the warped frame there is no image, and whatever fills it is
    something the model has never seen. `BORDER_REPLICATE` smears the edge
    pixels into long streaks, which measurably poisons the prediction —
    package 8 finds no door at all with replicated borders and 8757 px of
    door once the invented region is cropped away instead.

    Returns:
        `(image, (x0, y0, x1, y1))` — the cropped warp and where it sits in
        the warped frame, which the caller needs to map results back.
    """
    h, w = bgr.shape[:2]
    warped = cv2.warpPerspective(bgr, homography, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.warpPerspective(np.full((h, w), 255, np.uint8), homography, (w, h),
                                flags=cv2.INTER_NEAREST) > 0
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return warped, (0, 0, w, h)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    # Shrink the box from whichever edge is emptiest until it is essentially
    # all real pixels; a few iterations is enough at these sizes.
    for _ in range(200):
        sub = valid[y0:y1, x0:x1]
        if sub.size == 0 or sub.mean() > 0.995 or (y1 - y0) < 40 or (x1 - x0) < 40:
            break
        rows, cols = sub.mean(axis=1), sub.mean(axis=0)
        edges = (rows[0], rows[-1], cols[0], cols[-1])
        worst = int(np.argmin(edges))
        step_y = max(1, (y1 - y0) // 50)
        step_x = max(1, (x1 - x0) // 50)
        if worst == 0:
            y0 += step_y
        elif worst == 1:
            y1 -= step_y
        elif worst == 2:
            x0 += step_x
        else:
            x1 -= step_x
    return warped[y0:y1, x0:x1], (x0, y0, x1, y1)


def unwarp_channel(
    channel: np.ndarray,
    homography: np.ndarray,
    crop: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    """Map one probability channel from the cropped warp back to the image."""
    h, w = shape
    x0, y0, x1, y1 = crop
    full = np.zeros((h, w), np.float32)
    resized = cv2.resize(channel.astype(np.float32), (x1 - x0, y1 - y0),
                         interpolation=cv2.INTER_LINEAR)
    full[y0:y1, x0:x1] = resized
    return cv2.warpPerspective(full, np.linalg.inv(homography), (w, h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
