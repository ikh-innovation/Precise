"""Fetch street-level images around a coordinate, for facade analysis.

Given a (lon, lat) that sits somewhere in a street, get N (default 6) views of
the facades around it. Two modes:

  * `rotate` (default) — stand at the capture nearest the point and turn the
    camera. A 360° panorama gives N views at evenly spaced compass headings; a
    Mapillary fisheye/perspective frame is undistorted into N pinhole views
    spread across its own field of view (a GoPro fisheye covers ~123°, so this
    turns left-to-right but cannot see behind). Falls back to `surround` when
    nothing is within `--rotate-max-dist`.
  * `orbit` — a full 360° around the point from a single-direction sequence.
    N target spots are placed on a circle of `--orbit-radius` metres around
    the point (about the facade distance). Each spot is looked at from the
    frame that sees it closest while keeping it inside the lens, and that
    frame is undistorted into a view aimed at it. Spots behind the point come
    from frames further back looking forward. Needs Mapillary camera params.
  * `surround` — keep the N nearby frames that best surround the point and
    look at it (see below).

Providers:
  * `google`    — Street View Static API; needs a key in `GOOGLE_MAPS_API_KEY`
                  (or `--google-key`). Stands on the panorama nearest the point
                  and turns the camera: N views at evenly spaced headings.
  * `mapillary` — Graph API v4; needs a token in `MAPILLARY_TOKEN` (or
                  `--token`, or a `MAPILLARY_TOKEN=...` line in the repo's
                  gitignored `.env`).
  * `kartaview` — open API, no token. Dashcam-style frames looking along the
                  street.
  * `auto`      — Google if a key is available, else Mapillary if a token is,
                  else KartaView (default).

How `surround` chooses frames:
  1. Candidates within `--radius` metres of the point.
  2. Frames whose camera heading puts the point inside the field of view rank
     first; a frame taken just *past* the point looks away from it.
  3. First pass: the nearest candidate in each of N angular sectors around the
     point, so the picks surround it. Second pass: fill any sector left empty
     with the nearest remaining frames. Both passes skip frames closer than
     `--min-spacing` to an existing pick, so a parked car's burst of near-
     identical frames does not fill the set.

Outputs go to `<out>/<lat>_<lon>/`:
  * `NN_<provider>_<id>.jpg` — the images, nearest first.
  * `meta.json`              — per image: position, distance, heading, the
                               bearing from camera to point, and the point's
                               angle off the camera axis (negative = left).
  * `contact_sheet.jpg`      — all picks in one grid, for a quick look.

Usage (Windows conda env, see README):
    PY=C:/Users/achitos/AppData/Local/miniconda3/envs/precise/python.exe
    $PY src/street_imagery/fetch_street_images.py --lon -0.36146865338242 --lat 39.457402431416
    $PY src/street_imagery/fetch_street_images.py --lon ... --lat ... --mode orbit -n 12
    $PY src/street_imagery/fetch_street_images.py --coords-json assets_coords.json --index 0
    $PY src/street_imagery/fetch_street_images.py --lon ... --lat ... --provider mapillary --token MLY|...
    $PY src/street_imagery/fetch_street_images.py --lon ... --lat ... --provider google --google-key AIza...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import requests

PRECISE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PRECISE_ROOT / "data" / "street_images"
EARTH_R = 6_371_000.0
TIMEOUT = 30


@dataclass
class Capture:
    """One street-level image, normalised across providers."""

    provider: str
    id: str
    lon: float
    lat: float
    heading: float | None  # compass degrees, 0 = north, clockwise
    captured_at: str | None
    is_pano: bool
    url: str
    sequence: str | None = None
    extra: dict = field(default_factory=dict)
    # Filled in relative to the query point.
    distance_m: float = 0.0
    bearing_to_point: float = 0.0  # from the camera towards the point
    point_off_axis_deg: float | None = None  # point angle vs heading, -180..180


# ---------------------------------------------------------------- geometry
def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Initial compass bearing from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(x, y)) % 360


def wrap180(a: float) -> float:
    return (a + 180) % 360 - 180


def bbox_around(lon: float, lat: float, radius_m: float) -> tuple[float, float, float, float]:
    dlat = math.degrees(radius_m / EARTH_R)
    dlon = math.degrees(radius_m / (EARTH_R * math.cos(math.radians(lat))))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


# ---------------------------------------------------------------- providers
def query_mapillary(lon: float, lat: float, radius_m: float, token: str) -> list[Capture]:
    fields = ",".join([
        "id", "geometry", "computed_geometry", "compass_angle", "computed_compass_angle",
        "camera_type", "camera_parameters", "width", "height",
        "captured_at", "sequence", "thumb_2048_url", "thumb_original_url",
    ])
    r = requests.get(
        "https://graph.mapillary.com/images",
        params={
            "access_token": token,
            "bbox": ",".join(f"{v:.7f}" for v in bbox_around(lon, lat, radius_m)),
            "fields": fields,
            "limit": 2000,
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    out = []
    for d in r.json().get("data", []):
        geom = d.get("computed_geometry") or d.get("geometry")
        url = d.get("thumb_original_url") or d.get("thumb_2048_url")
        if not geom or not url:
            continue
        heading = d.get("computed_compass_angle", d.get("compass_angle"))
        out.append(Capture(
            provider="mapillary",
            id=str(d["id"]),
            lon=geom["coordinates"][0],
            lat=geom["coordinates"][1],
            heading=None if heading is None else float(heading) % 360,
            captured_at=str(d.get("captured_at")) if d.get("captured_at") else None,
            is_pano=d.get("camera_type") in ("spherical", "equirectangular"),
            url=url,
            sequence=d.get("sequence"),
            extra={"camera_type": d.get("camera_type"),
                   "camera_parameters": d.get("camera_parameters"),
                   "width": d.get("width"), "height": d.get("height")},
        ))
    return out


def _kartaview_url(name: str) -> str:
    # v1 returns "storage13/files/photo/...": host prefix, then the path.
    host, _, path = name.partition("/")
    return f"https://{host}.openstreetcam.org/{path}"


def query_kartaview(lon: float, lat: float, radius_m: float) -> list[Capture]:
    r = requests.post(
        "https://api.openstreetcam.org/1.0/list/nearby-photos/",
        data={"lat": f"{lat:.7f}", "lng": f"{lon:.7f}", "radius": int(radius_m)},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    out = []
    for d in r.json().get("currentPageItems", []):
        heading = d.get("heading")
        out.append(Capture(
            provider="kartaview",
            id=str(d["id"]),
            lon=float(d["lng"]),
            lat=float(d["lat"]),
            heading=None if heading in (None, "") else float(heading) % 360,
            captured_at=d.get("shot_date"),
            is_pano=str(d.get("projection", "")).upper() in ("SPHERE", "EQUIRECTANGULAR"),
            url=_kartaview_url(d["name"]),
            sequence=d.get("sequence_id"),
            extra={"sequence_index": d.get("sequence_index")},
        ))
    return out


# ---------------------------------------------------------------- selection
def annotate(caps: list[Capture], lon: float, lat: float) -> None:
    for c in caps:
        c.distance_m = haversine_m(c.lon, c.lat, lon, lat)
        c.bearing_to_point = bearing_deg(c.lon, c.lat, lon, lat)
        if c.heading is not None:
            c.point_off_axis_deg = wrap180(c.bearing_to_point - c.heading)


def sees_point(c: Capture, hfov: float) -> bool:
    """True if the point falls inside a perspective frame's horizontal FOV.

    Unknown heading counts as seeing it; panoramas always do. Within ~3 m of the
    camera the bearing is dominated by GPS noise, so those frames also pass.
    """
    if c.is_pano or c.point_off_axis_deg is None or c.distance_m < 3:
        return True
    return abs(c.point_off_axis_deg) <= hfov / 2


def select(caps: list[Capture], n: int, hfov: float, min_spacing_m: float) -> list[Capture]:
    """Pick up to n frames that surround the point and look at it.

    Frames that see the point are exhausted before any that look away.
    """
    by_dist = sorted(caps, key=lambda c: c.distance_m)
    seeing = [c for c in by_dist if sees_point(c, hfov)]
    away = [c for c in by_dist if not sees_point(c, hfov)]
    chosen: list[Capture] = []
    sector_w = 360 / n

    def far_enough(c: Capture) -> bool:
        return all(haversine_m(c.lon, c.lat, o.lon, o.lat) >= min_spacing_m for o in chosen)

    for tier in (seeing, away):
        # Pass 1: nearest per sector (sector = bearing from the point to the camera).
        taken = set()
        for c in tier:
            s = int(((c.bearing_to_point + 180) % 360) // sector_w)
            if len(chosen) < n and s not in taken and far_enough(c):
                chosen.append(c)
                taken.add(s)
        # Pass 2: fill up with the nearest of the rest, still respecting spacing.
        for c in tier:
            if len(chosen) < n and c not in chosen and far_enough(c):
                chosen.append(c)
    # Pass 3: if the street has too few distinct spots, relax the spacing.
    for c in seeing + away:
        if len(chosen) < n and c not in chosen:
            chosen.append(c)
    return sorted(chosen, key=lambda c: c.distance_m)


# ---------------------------------------------------------------- images
def download(url: str) -> np.ndarray:
    r = requests.get(url, timeout=TIMEOUT * 2)
    r.raise_for_status()
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"not an image: {url}")
    return img


def view_rays(yaw_deg: float, pitch_deg: float, hfov_deg: float, out_w: int, out_h: int) -> np.ndarray:
    """Unit rays (x right, y down, z forward) of a pinhole view turned by yaw/pitch."""
    f = (out_w / 2) / math.tan(math.radians(hfov_deg) / 2)
    xs, ys = np.meshgrid(np.arange(out_w) - out_w / 2, np.arange(out_h) - out_h / 2)
    # Camera rays: x right, y down, z forward.
    rays = np.stack([xs, ys, np.full_like(xs, f)], axis=-1).astype(np.float64)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    p, y = -math.radians(pitch_deg), math.radians(yaw_deg)  # y axis points down: + pitch looks up
    rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    return rays @ (ry @ rx).T


def equirect_to_perspective(
    pano: np.ndarray, yaw_deg: float, pitch_deg: float, hfov_deg: float, out_w: int, out_h: int
) -> np.ndarray:
    """Render a pinhole view from an equirectangular panorama.

    yaw is relative to the pano's centre column (0 = centre, + = right).
    """
    ph, pw = pano.shape[:2]
    rays = view_rays(yaw_deg, pitch_deg, hfov_deg, out_w, out_h)
    lon = np.arctan2(rays[..., 0], rays[..., 2])
    lat = np.arcsin(np.clip(rays[..., 1], -1, 1))
    map_x = ((lon / (2 * math.pi) + 0.5) * pw).astype(np.float32)
    map_y = ((lat / math.pi + 0.5) * ph).astype(np.float32)
    return cv2.remap(pano, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _distort_scale(camera_type: str, k1: float, k2: float, t: np.ndarray) -> np.ndarray:
    """OpenSfM radial model: fisheye distorts the angle, perspective the tangent."""
    return t * (1 + k1 * t**2 + k2 * t**4)


def camera_half_fov_deg(camera_type: str, params: list[float], w: int, h: int) -> tuple[float, float]:
    """Horizontal and vertical half field of view, from the edge midpoints."""
    f, k1, k2 = params
    t = np.linspace(0, math.pi / 2 - 1e-3 if camera_type == "fisheye" else 1.5, 20000)
    arg = t if camera_type == "fisheye" else np.tan(t)
    r = f * _distort_scale(camera_type, k1, k2, arg)
    s = max(w, h)
    return tuple(float(np.degrees(t[np.searchsorted(r, (d / 2) / s)])) for d in (w, h))


def camera_to_perspective(
    img: np.ndarray, camera_type: str, params: list[float],
    yaw_deg: float, pitch_deg: float, hfov_deg: float, out_w: int, out_h: int,
) -> np.ndarray:
    """Undistort a Mapillary fisheye/perspective frame into a turned pinhole view.

    Uses Mapillary's OpenSfM camera: params = [focal, k1, k2], focal and pixel
    coordinates normalised by max(width, height) around the image centre. yaw
    and pitch are relative to the camera axis. Rays outside the frame are black.
    """
    h, w = img.shape[:2]
    f, k1, k2 = params
    rays = view_rays(yaw_deg, pitch_deg, hfov_deg, out_w, out_h)
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    r = np.hypot(x, y)
    if camera_type == "fisheye":
        arg = np.arctan2(r, z)
    else:
        arg = np.where(z > 1e-6, r / np.maximum(z, 1e-6), np.inf)
    scale = f * _distort_scale(camera_type, k1, k2, arg) / np.maximum(r, 1e-12)
    s = max(w, h)
    map_x = (x * scale * s + (w - 1) / 2).astype(np.float32)
    map_y = (y * scale * s + (h - 1) / 2).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def camera_view_fit(
    img: np.ndarray, camera_type: str, params: list[float],
    yaw_deg: float, pitch_deg: float, hfov_deg: float, out_w: int, out_h: int,
    max_black: float = 0.002, step: float = 2.0,
) -> tuple[np.ndarray, float]:
    """camera_to_perspective, lowering the pitch until the view stays inside the frame.

    Tilting up can run past the top edge of the lens, leaving black corners;
    this backs off in `step` degree increments (down to 0) until at most
    `max_black` of the view is outside the frame. Returns (view, pitch used).
    """
    probe = np.full(img.shape[:2], 255, np.uint8)
    pitch = pitch_deg
    while True:
        mask = camera_to_perspective(probe, camera_type, params, yaw_deg, pitch, hfov_deg,
                                     out_w // 4, out_h // 4)
        if pitch <= 0 or (mask < 128).mean() <= max_black:
            break
        pitch = max(0.0, pitch - step)
    view = camera_to_perspective(img, camera_type, params, yaw_deg, pitch, hfov_deg, out_w, out_h)
    return view, pitch


def contact_sheet(images: list[np.ndarray], labels: list[str], cell_w: int = 640) -> np.ndarray:
    cols = 3 if len(images) > 4 else 2 if len(images) > 1 else 1
    rows = math.ceil(len(images) / cols)
    portrait = all(im.shape[0] > im.shape[1] for im in images)
    cell_w = cell_w // 2 if portrait else cell_w
    cols = min(len(images), 6) if portrait else cols
    if portrait and len(images) > 6:
        cols = math.ceil(len(images) / 2)
    rows = math.ceil(len(images) / cols)
    cell_h = int(cell_w * 4 / 3) if portrait else int(cell_w * 9 / 16)
    sheet = np.zeros((rows * cell_h, cols * cell_w, 3), np.uint8)
    for i, (img, label) in enumerate(zip(images, labels)):
        h, w = img.shape[:2]
        s = min(cell_w / w, cell_h / h)
        small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        y0 = r * cell_h + (cell_h - small.shape[0]) // 2
        x0 = c * cell_w + (cell_w - small.shape[1]) // 2
        sheet[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        cv2.putText(sheet, label, (c * cell_w + 8, r * cell_h + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(sheet, label, (c * cell_w + 8, r * cell_h + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return sheet


# ---------------------------------------------------------------- orbit
def enu(lon: float, lat: float, lon0: float, lat0: float) -> np.ndarray:
    """Local east/north metres around (lon0, lat0); fine at street scale."""
    return np.array([math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0)),
                     math.radians(lat - lat0) * EARTH_R])


def usable_half_fov(c: Capture, hfov: float) -> float | None:
    """Largest yaw a view of width hfov can turn inside this frame (180 for panos)."""
    if c.is_pano:
        return 180.0
    cam = c.extra
    if (c.heading is None or cam.get("camera_type") not in ("fisheye", "perspective")
            or not cam.get("camera_parameters") or len(cam["camera_parameters"]) != 3
            or not cam.get("width") or not cam.get("height")):
        return None
    half_h, _ = camera_half_fov_deg(cam["camera_type"], cam["camera_parameters"],
                                    cam["width"], cam["height"])
    return max(0.0, half_h - hfov / 2)


def plan_orbit(caps: list[Capture], lon: float, lat: float, n: int, radius_m: float,
               hfov: float, min_dist_m: float, start_heading: float) -> list[dict]:
    """For each of n spots around the point, the frame + yaw that sees it best."""
    usable = [(c, m) for c in caps if (m := usable_half_fov(c, hfov)) is not None]
    plan = []
    for i in range(n):
        theta = (start_heading + i * 360 / n) % 360
        tgt = radius_m * np.array([math.sin(math.radians(theta)), math.cos(math.radians(theta))])
        best = None
        for c, max_yaw in usable:
            v = tgt - enu(c.lon, c.lat, lon, lat)
            d = float(np.hypot(*v))
            bearing = math.degrees(math.atan2(v[0], v[1])) % 360
            yaw = wrap180(bearing - (c.heading or 0.0))
            if d < min_dist_m or abs(yaw) > max_yaw:
                continue
            # Closest wins; strong yaw (image edge, most stretched) costs a bit.
            score = d * (1 + 0.3 * abs(yaw) / max(max_yaw, 1))
            if best is None or score < best["score"]:
                best = {"cap": c, "yaw": yaw, "dist": d, "bearing": bearing, "score": score}
        plan.append({"theta": theta, "target_enu": tgt} | (best or {}))
    return plan


def orbit_map(plan: list[dict], caps: list[Capture], lon: float, lat: float,
              hfov: float, size: int = 900) -> np.ndarray:
    """Top-down sketch: point, target circle, all frames, and each chosen sight line."""
    used = [p["cap"] for p in plan if "cap" in p]
    pts = [enu(c.lon, c.lat, lon, lat) for c in used] + [p["target_enu"] for p in plan]
    extent = max(10.0, max(float(np.abs(q).max()) for q in pts)) * 1.15
    k = (size / 2) / extent

    def px(e):  # east right, north up
        return int(size / 2 + e[0] * k), int(size / 2 - e[1] * k)

    img = np.full((size, size, 3), 250, np.uint8)
    for c in caps:
        cv2.circle(img, px(enu(c.lon, c.lat, lon, lat)), 3, (170, 170, 170), -1)
    for i, p in enumerate(plan):
        t = px(p["target_enu"])
        if "cap" in p:
            c = p["cap"]
            o = enu(c.lon, c.lat, lon, lat)
            for side in (-1, 1):  # view wedge edges
                a = math.radians(p["bearing"] + side * hfov / 2)
                cv2.line(img, px(o), px(o + p["dist"] * np.array([math.sin(a), math.cos(a)])),
                         (215, 220, 240), 1, cv2.LINE_AA)
            cv2.line(img, px(o), t, (40, 120, 230), 1, cv2.LINE_AA)
            cv2.circle(img, px(o), 5, (40, 120, 230), -1)
        cv2.circle(img, t, 6, (60, 160, 60) if "cap" in p else (60, 60, 220), -1)
        cv2.putText(img, str(i), (t[0] + 7, t[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (30, 30, 30), 1, cv2.LINE_AA)
    cv2.drawMarker(img, px((0, 0)), (0, 0, 200), cv2.MARKER_CROSS, 18, 2)
    cv2.putText(img, f"N up, {2 * extent:.0f} m across. + point, green spot covered, red none, "
                "orange = frame used", (10, size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (30, 30, 30), 1, cv2.LINE_AA)
    return img


def fetch_orbit(caps: list[Capture], lon: float, lat: float, args: argparse.Namespace,
                out_dir: Path) -> tuple[list[dict], list[np.ndarray], list[str]]:
    plan = plan_orbit(caps, lon, lat, args.n, args.orbit_radius, args.rotate_hfov,
                      args.orbit_min_dist, args.start_heading)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, np.ndarray] = {}
    records, images, labels = [], [], []
    for i, p in enumerate(plan):
        if "cap" not in p:
            print(f"  #{i} around {p['theta']:.0f}: no frame sees this spot")
            continue
        c = p["cap"]
        if c.id not in cache:
            cache[c.id] = download(c.url)
            cv2.imwrite(str(frames_dir / f"{c.provider}_{c.id}.jpg"), cache[c.id])
        img = cache[c.id]
        if c.is_pano:
            pitch = args.pitch
            view = equirect_to_perspective(img, wrap180(p["bearing"] - (c.heading or 0.0)),
                                           pitch, args.rotate_hfov, 960, 1280)
        else:
            view, pitch = camera_view_fit(img, c.extra["camera_type"], c.extra["camera_parameters"],
                                          p["yaw"], args.pitch, args.rotate_hfov, 960, 1280)
        name = f"{i:02d}_around{p['theta']:03.0f}_{c.provider}_{c.id}.jpg"
        cv2.imwrite(str(out_dir / name), view)
        records.append(asdict(c) | {
            "file": name, "around_heading": p["theta"], "orbit_radius_m": args.orbit_radius,
            "view_heading": p["bearing"], "view_yaw": p["yaw"], "view_pitch": pitch,
            "view_hfov": args.rotate_hfov, "distance_to_spot_m": p["dist"],
            "source_frame": f"frames/{c.provider}_{c.id}.jpg"})
        images.append(view)
        labels.append(f"#{i} around {p['theta']:.0f}  {p['dist']:.0f} m  tilt {pitch:.0f}")
    cv2.imwrite(str(out_dir / "orbit_map.png"), orbit_map(plan, caps, lon, lat, args.rotate_hfov))
    print(f"orbit: {len(records)}/{len(plan)} spots covered from {len(cache)} frames")
    return records, images, labels


# ---------------------------------------------------------------- google
GSV = "https://maps.googleapis.com/maps/api/streetview"


def fetch_google(lon: float, lat: float, key: str, args: argparse.Namespace, out_dir: Path):
    """Nearest outdoor Street View pano, rotated through n headings.

    The metadata call is free; each image counts against the Static API quota.
    """
    r = requests.get(f"{GSV}/metadata", timeout=TIMEOUT, params={
        "location": f"{lat:.7f},{lon:.7f}", "radius": int(args.radius),
        "source": "outdoor", "key": key})
    r.raise_for_status()
    m = r.json()
    if m.get("status") != "OK":
        sys.exit(f"Street View metadata: {m.get('status')} {m.get('error_message', '')}".strip())
    pano_id = m["pano_id"]
    plon, plat = m["location"]["lng"], m["location"]["lat"]
    dist = haversine_m(plon, plat, lon, lat)
    print(f"google: pano {pano_id} ({m.get('date', '?')}) {dist:.1f} m from the point")

    records, images, labels = [], [], []
    for i in range(args.n):
        heading = (args.start_heading + i * 360 / args.n) % 360
        r = requests.get(GSV, timeout=TIMEOUT * 2, params={
            "pano": pano_id, "size": args.gsv_size, "heading": f"{heading:.1f}",
            "pitch": args.pitch, "fov": args.hfov, "return_error_code": "true", "key": key})
        r.raise_for_status()
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        name = f"{i:02d}_google_{pano_id}_h{heading:03.0f}.jpg"
        cv2.imwrite(str(out_dir / name), img)
        records.append({
            "provider": "google", "id": pano_id, "lon": plon, "lat": plat,
            "captured_at": m.get("date"), "copyright": m.get("copyright"),
            "distance_m": dist, "heading": heading, "view_pitch": args.pitch,
            "view_hfov": args.hfov, "file": name})
        images.append(img)
        labels.append(f"#{i} hdg {heading:.0f}")
    return records, images, labels


# ---------------------------------------------------------------- main
def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines into os.environ without overriding what is set."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, val = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))


def fetch(lon: float, lat: float, args: argparse.Namespace) -> Path:
    load_dotenv(PRECISE_ROOT / ".env")
    token = args.token or os.environ.get("MAPILLARY_TOKEN") or os.environ.get("MAPILLARY_ACCESS_TOKEN")
    gkey = args.google_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    provider = args.provider
    if provider == "auto":
        provider = "google" if gkey else "mapillary" if token else "kartaview"
    if provider == "mapillary" and not token:
        sys.exit("Mapillary needs a token: set MAPILLARY_TOKEN or pass --token.")
    if provider == "google" and not gkey:
        sys.exit("Street View needs a key: set GOOGLE_MAPS_API_KEY or pass --google-key.")

    out_dir = Path(args.out) / f"{lat:.6f}_{lon:.6f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if provider == "google":
        records, images, labels = fetch_google(lon, lat, gkey, args, out_dir)
        return write_outputs(out_dir, {"lon": lon, "lat": lat, "provider": provider,
                                       "n": args.n, "hfov": args.hfov}, records, images, labels)

    if args.mode == "orbit":
        if provider != "mapillary":
            sys.exit("orbit needs Mapillary camera parameters: set a Mapillary token.")
        radius = args.orbit_search
        caps = query_mapillary(lon, lat, radius, token)
        annotate(caps, lon, lat)
        print(f"mapillary: {len(caps)} captures within {radius:.0f} m")
        orbit_dir = out_dir / "orbit"
        orbit_dir.mkdir(exist_ok=True)
        records, images, labels = fetch_orbit(caps, lon, lat, args, orbit_dir)
        return write_outputs(orbit_dir, {"lon": lon, "lat": lat, "radius_m": radius,
                                         "provider": provider, "mode": "orbit", "n": args.n,
                                         "orbit_radius_m": args.orbit_radius,
                                         "hfov": args.rotate_hfov}, records, images, labels)

    caps: list[Capture] = []
    radius = args.radius
    # Widen the search until enough frames face the point.
    for _ in range(3):
        caps = query_mapillary(lon, lat, radius, token) if provider == "mapillary" \
            else query_kartaview(lon, lat, radius)
        annotate(caps, lon, lat)
        n_seeing = sum(sees_point(c, args.hfov) for c in caps)
        if n_seeing >= args.n or any(c.is_pano for c in caps):
            break
        radius *= 1.6
    if not caps:
        sys.exit(f"No {provider} imagery within {radius:.0f} m of ({lat}, {lon}).")
    print(f"{provider}: {len(caps)} captures within {radius:.0f} m, {n_seeing} facing the point")

    records, images, labels = [], [], []
    panos = sorted((c for c in caps if c.is_pano), key=lambda c: c.distance_m)
    nearest = min(caps, key=lambda c: c.distance_m)
    cam = nearest.extra
    can_undistort = (
        provider == "mapillary" and not nearest.is_pano and nearest.heading is not None
        and cam.get("camera_type") in ("fisheye", "perspective")
        and cam.get("camera_parameters") and len(cam["camera_parameters"]) == 3)

    if args.mode == "rotate" and panos and panos[0].distance_m <= args.rotate_max_dist:
        # A 360 pano covers every direction on its own: render n views from it.
        pano_cap = panos[0]
        pano = download(pano_cap.url)
        cv2.imwrite(str(out_dir / f"pano_{pano_cap.provider}_{pano_cap.id}.jpg"), pano)
        base = pano_cap.heading or 0.0  # compass heading of the pano's centre column
        for i in range(args.n):
            view_heading = (args.start_heading + i * 360 / args.n) % 360
            view = equirect_to_perspective(
                pano, wrap180(view_heading - base), args.pitch, args.hfov, 1280, 960)
            name = f"{i:02d}_{pano_cap.provider}_{pano_cap.id}_h{view_heading:03.0f}.jpg"
            cv2.imwrite(str(out_dir / name), view)
            rec = asdict(pano_cap) | {"file": name, "view_heading": view_heading,
                                      "view_pitch": args.pitch, "view_hfov": args.hfov}
            records.append(rec)
            images.append(view)
            labels.append(f"#{i} pano h={view_heading:.0f}")
    elif args.mode == "rotate" and can_undistort and nearest.distance_m <= args.rotate_max_dist:
        # One frame at the point: turn a virtual pinhole camera inside its FOV.
        img = download(nearest.url)
        params = cam["camera_parameters"]
        half_h, half_v = camera_half_fov_deg(cam["camera_type"], params, img.shape[1], img.shape[0])
        max_yaw = max(0.0, half_h - args.rotate_hfov / 2)
        yaws = np.linspace(-max_yaw, max_yaw, args.n) if args.n > 1 else np.zeros(1)
        out_w, out_h = 960, 1280  # portrait: facades are taller than wide
        print(f"rotate: {cam['camera_type']} frame {nearest.id} {nearest.distance_m:.1f} m from the "
              f"point, FOV +/-{half_h:.0f} x +/-{half_v:.0f} deg, yaws +/-{max_yaw:.0f} deg")
        cv2.imwrite(str(out_dir / f"src_{nearest.provider}_{nearest.id}.jpg"), img)
        for i, yaw in enumerate(yaws):
            view, pitch = camera_view_fit(img, cam["camera_type"], params, float(yaw),
                                          args.pitch, args.rotate_hfov, out_w, out_h)
            view_heading = (nearest.heading + yaw) % 360
            name = f"{i:02d}_{nearest.provider}_{nearest.id}_h{view_heading:03.0f}.jpg"
            cv2.imwrite(str(out_dir / name), view)
            records.append(asdict(nearest) | {
                "file": name, "view_heading": view_heading, "view_yaw": float(yaw),
                "view_pitch": pitch, "view_hfov": args.rotate_hfov})
            images.append(view)
            labels.append(f"#{i} hdg {view_heading:.0f} (yaw {yaw:+.0f})")
    else:
        if args.mode == "rotate":
            print(f"rotate: nothing to turn within {args.rotate_max_dist:.0f} m, using surround")
        for i, c in enumerate(select([c for c in caps if not c.is_pano] or caps,
                                     args.n, args.hfov, args.min_spacing)):
            try:
                img = download(c.url)
            except (requests.RequestException, ValueError) as e:
                print(f"  skip {c.id}: {e}")
                continue
            name = f"{i:02d}_{c.provider}_{c.id}.jpg"
            cv2.imwrite(str(out_dir / name), img)
            records.append(asdict(c) | {"file": name, "sees_point": sees_point(c, args.hfov)})
            images.append(img)
            off = "?" if c.point_off_axis_deg is None else f"{c.point_off_axis_deg:+.0f}"
            labels.append(f"#{i} {c.distance_m:.0f} m  hdg {c.heading or 0:.0f}  pt {off} deg")

    return write_outputs(out_dir, {"lon": lon, "lat": lat, "radius_m": radius, "provider": provider,
                                   "n": args.n, "hfov": args.hfov}, records, images, labels)


def write_outputs(out_dir: Path, query: dict, records: list[dict],
                  images: list[np.ndarray], labels: list[str]) -> Path:
    meta = {"query": query, "images": records}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if images:
        cv2.imwrite(str(out_dir / "contact_sheet.jpg"), contact_sheet(images, labels))
    for r in records:
        if "around_heading" in r:
            print(f"  {r['file']}: frame {r['id']} ({r['distance_m']:.0f} m from the point) "
                  f"looking {r['view_heading']:.0f} (yaw {r['view_yaw']:+.0f}), "
                  f"spot {r['distance_to_spot_m']:.0f} m away")
            continue
        if "view_heading" in r:
            print(f"  {r['file']}: view heading {r['view_heading']:.0f}, pitch {r['view_pitch']:+.0f}, "
                  f"from {r['distance_m']:.1f} m")
            continue
        off = r.get("point_off_axis_deg")
        print(f"  {r['file']}: {r['distance_m']:.1f} m, heading {r['heading']}, "
              f"point {'n/a' if off is None else f'{off:+.0f} deg'} off axis")
    print(f"-> {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lon", type=float)
    ap.add_argument("--lat", type=float)
    ap.add_argument("--coords-json", help="JSON list of {coordinates: [lon, lat]} items")
    ap.add_argument("--index", type=int, nargs="*", help="items of --coords-json to run (default all)")
    ap.add_argument("--provider", choices=["auto", "google", "mapillary", "kartaview"], default="auto")
    ap.add_argument("--token", help="Mapillary access token (else MAPILLARY_TOKEN env var)")
    ap.add_argument("--google-key", help="Street View Static API key (else GOOGLE_MAPS_API_KEY env var)")
    ap.add_argument("--gsv-size", default="640x640", help="Street View image size (max 640x640)")
    ap.add_argument("--start-heading", type=float, default=0, help="rotating views: first heading, deg")
    ap.add_argument("-n", type=int, default=6, help="images per point")
    ap.add_argument("--radius", type=float, default=50, help="initial search radius, m")
    ap.add_argument("--min-spacing", type=float, default=6, help="min distance between picks, m")
    ap.add_argument("--hfov", type=float, default=90, help="assumed / rendered horizontal FOV, deg")
    ap.add_argument("--pitch", type=float, default=20,
                    help="rotating views: upward tilt, deg (lowered per view if it leaves the lens)")
    ap.add_argument("--mode", choices=["rotate", "orbit", "surround"], default="rotate")
    ap.add_argument("--orbit-radius", type=float, default=15,
                    help="orbit: distance from the point to the spots looked at, m")
    ap.add_argument("--orbit-search", type=float, default=120,
                    help="orbit: search radius for frames, m")
    ap.add_argument("--orbit-min-dist", type=float, default=6,
                    help="orbit: ignore frames closer than this to a spot, m")
    ap.add_argument("--rotate-max-dist", type=float, default=15,
                    help="rotate: use a pano/frame only this close to the point, m")
    ap.add_argument("--rotate-hfov", type=float, default=40,
                    help="rotate from a fisheye/perspective frame: view width, deg")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if args.coords_json:
        items = json.loads(Path(args.coords_json).read_text(encoding="utf-8"))
        idx = args.index if args.index else range(len(items))
        points = [tuple(items[i]["coordinates"][:2]) for i in idx]
    elif args.lon is not None and args.lat is not None:
        points = [(args.lon, args.lat)]
    else:
        ap.error("give --lon/--lat or --coords-json")
    for lon, lat in points:
        fetch(lon, lat, args)


if __name__ == "__main__":
    main()
