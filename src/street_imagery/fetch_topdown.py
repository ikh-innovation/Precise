"""Fetch a top-down orthophoto tile centred on each coordinate.

For each (lon, lat), a square of `2 * --half` metres around the point from the
open national orthophoto service of the country the point is in (see
ORTHO_SOURCES). The country comes from OSM Nominatim reverse geocoding, or
`--country`. Works Europe-wide wherever a source is configured; Esri World
Imagery is an opt-in global fallback (`--allow-esri`).

Outputs, per point in `<out>/<lat>_<lon>/topdown/` (next to the facade views
written by `fetch_street_images.py`):
  * `ortho.jpg`  — the tile, north up, centred on the point.
  * `ortho.json` — its bbox (WGS84), pixel size, source, licence, attribution.

Usage (Windows conda env, see README):
    PY=C:/Users/achitos/AppData/Local/miniconda3/envs/precise/python.exe
    $PY src/street_imagery/fetch_topdown.py --coords-json data/assets_coords.json
    $PY src/street_imagery/fetch_topdown.py --lon 2.3470 --lat 48.8590
    $PY src/street_imagery/fetch_topdown.py --lon 23.7275 --lat 37.9838 --allow-esri
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests

PRECISE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PRECISE_ROOT / "data" / "street_images"
EARTH_R = 6_371_000.0
TIMEOUT = 90
# Nominatim's usage policy asks for an identifying User-Agent.
HEADERS = {"User-Agent": "Precise-building-analysis/0.1 (street + top-down imagery research)"}


@dataclass(frozen=True)
class OrthoSource:
    key: str
    countries: tuple[str, ...]  # ISO 3166-1 alpha-2, lower case
    url: str
    layer: str
    gsd_m: float  # native ground sample distance
    attribution: str
    license: str
    version: str = "1.1.1"
    region: tuple[float, float, float, float] | None = None  # lon0, lat0, lon1, lat1


# Open national / regional orthophoto WMS, all verified to return imagery.
# More specific (regional) sources first. Add a country by adding a row.
ORTHO_SOURCES = [
    OrthoSource("de-berlin", ("de",), "https://gdi.berlin.de/services/wms/truedop_2024",
                "truedop_2024", 0.2, "Geoportal Berlin / TrueDOP 2024", "dl-de/zero-2-0",
                region=(13.08, 52.33, 13.77, 52.68)),
    OrthoSource("de-nrw", ("de",), "https://www.wms.nrw.de/geobasis/wms_nw_dop",
                "nw_dop_rgb", 0.1, "Geobasis NRW", "dl-de/zero-2-0",
                region=(5.86, 50.32, 9.46, 52.53)),
    OrthoSource("es-pnoa", ("es",), "https://www.ign.es/wms-inspire/pnoa-ma",
                "OI.OrthoimageCoverage", 0.25, "PNOA cedido por © Instituto Geográfico Nacional",
                "CC-BY 4.0"),
    OrthoSource("fr-ign", ("fr",), "https://data.geopf.fr/wms-r/wms",
                "HR.ORTHOIMAGERY.ORTHOPHOTOS", 0.2, "IGN-F / Géoportail", "Etalab Licence Ouverte 2.0",
                version="1.3.0"),
    OrthoSource("nl-pdok", ("nl",), "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0",
                "Actueel_orthoHR", 0.08, "Beeldmateriaal.nl / PDOK", "CC-BY 4.0", version="1.3.0"),
    OrthoSource("pl-geoportal", ("pl",),
                "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolution",
                "Raster", 0.25, "GUGiK / geoportal.gov.pl", "open (GUGiK terms)"),
]
# Global fallback, opt-in only: Esri's terms tie World Imagery to Esri/ArcGIS use.
ESRI = OrthoSource("esri-world", (), "https://services.arcgisonline.com/ArcGIS/rest/services/"
                   "World_Imagery/MapServer/export", "", 0.3,
                   "Esri, Maxar, Earthstar Geographics, and the GIS User Community",
                   "Esri terms of use (check before redistributing or training on it)")


def pick_source(lon: float, lat: float, country: str | None, forced: str | None,
                allow_esri: bool) -> OrthoSource:
    if forced:
        for s in ORTHO_SOURCES + [ESRI]:
            if s.key == forced:
                return s
        sys.exit(f"unknown --ortho-source {forced}; have: "
                 + ", ".join(s.key for s in ORTHO_SOURCES + [ESRI]))
    for s in ORTHO_SOURCES:
        if country in s.countries and (s.region is None or (
                s.region[0] <= lon <= s.region[2] and s.region[1] <= lat <= s.region[3])):
            return s
    if allow_esri:
        return ESRI
    sys.exit(f"no open orthophoto source configured for country '{country}' at ({lat}, {lon}). "
             "Add one to ORTHO_SOURCES, or pass --allow-esri.")


def _get(url: str, retries: int = 4, **kw) -> requests.Response:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, **kw)
            if r.status_code < 500 and r.status_code != 429:
                r.raise_for_status()
                return r
        except requests.ConnectionError:
            pass
        time.sleep(2 * (attempt + 1))
    r.raise_for_status()
    return r


def get_ortho(src: OrthoSource, bbox: tuple[float, float, float, float], w: int, h: int) -> np.ndarray:
    lon0, lat0, lon1, lat1 = bbox
    if src is ESRI:
        params = {"bbox": f"{lon0},{lat0},{lon1},{lat1}", "bboxSR": 4326, "imageSR": 4326,
                  "size": f"{w},{h}", "format": "jpg", "f": "image"}
    else:
        params = {"SERVICE": "WMS", "VERSION": src.version, "REQUEST": "GetMap", "LAYERS": src.layer,
                  "STYLES": "", "WIDTH": w, "HEIGHT": h, "FORMAT": "image/jpeg"}
        if src.version == "1.3.0":  # EPSG:4326 axis order is lat, lon in WMS 1.3.0
            params |= {"CRS": "EPSG:4326", "BBOX": f"{lat0},{lon0},{lat1},{lon1}"}
        else:
            params |= {"SRS": "EPSG:4326", "BBOX": f"{lon0},{lat0},{lon1},{lat1}"}
    r = _get(src.url, params=params)
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"{src.key} returned no image: {r.text[:200]}")
    return img


_country_cache: dict[tuple[float, float], str | None] = {}


def country_of(lon: float, lat: float) -> str | None:
    """ISO country code from Nominatim, cached per ~10 km cell (1 request/s policy)."""
    cell = (round(lon, 1), round(lat, 1))
    if cell not in _country_cache:
        r = _get("https://nominatim.openstreetmap.org/reverse",
                 params={"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 3})
        _country_cache[cell] = (r.json().get("address") or {}).get("country_code")
        time.sleep(1)
    return _country_cache[cell]


def bbox_around(lon: float, lat: float, half_m: float) -> tuple[float, float, float, float]:
    dlat = math.degrees(half_m / EARTH_R)
    dlon = math.degrees(half_m / (EARTH_R * math.cos(math.radians(lat))))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def fetch_ortho(lon: float, lat: float, args: argparse.Namespace) -> Path:
    out_dir = Path(args.out) / f"{lat:.6f}_{lon:.6f}" / "topdown"
    out_dir.mkdir(parents=True, exist_ok=True)
    country = args.country or country_of(lon, lat)
    src = pick_source(lon, lat, country, args.ortho_source, args.allow_esri)
    gsd = args.gsd or src.gsd_m
    size = min(int(round(2 * args.half / gsd)), 4000)
    bbox = bbox_around(lon, lat, args.half)
    ortho = get_ortho(src, bbox, size, size)
    cv2.imwrite(str(out_dir / "ortho.jpg"), ortho, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out_dir / "ortho.json").write_text(json.dumps({
        "bbox_lonlat": bbox, "center": [lon, lat], "half_size_m": args.half,
        "pixel_size_m": 2 * args.half / size, "width": size, "height": size,
        "source": src.key, "layer": src.layer, "url": src.url, "native_gsd_m": src.gsd_m,
        "attribution": src.attribution, "license": src.license, "country": country,
    }, indent=1), encoding="utf-8")
    print(f"({lat:.6f}, {lon:.6f}) {country}/{src.key}: {size}x{size} px, "
          f"{2 * args.half / size:.2f} m/px -> {out_dir / 'ortho.jpg'}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lon", type=float)
    ap.add_argument("--lat", type=float)
    ap.add_argument("--coords-json", help="JSON list of {coordinates: [lon, lat]} items")
    ap.add_argument("--index", type=int, nargs="*", help="items of --coords-json to run (default all)")
    ap.add_argument("--half", type=float, default=80, help="half side of the square, m")
    ap.add_argument("--gsd", type=float, help="pixel size, m (default: the source's native)")
    ap.add_argument("--country", help="ISO country code; skips Nominatim")
    ap.add_argument("--ortho-source", help="force a source key, e.g. es-pnoa, fr-ign, esri-world")
    ap.add_argument("--allow-esri", action="store_true",
                    help="fall back to Esri World Imagery where no open source is configured")
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
        try:
            fetch_ortho(lon, lat, args)
        except (RuntimeError, requests.RequestException) as e:
            print(f"({lat}, {lon}) failed: {e}")


if __name__ == "__main__":
    main()
