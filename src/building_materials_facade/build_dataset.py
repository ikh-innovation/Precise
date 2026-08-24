"""Build the unified Facade-8 training manifest from all available sources.

Pools four facade-domain sources into one CSV manifest with a stratified
train/val/test split:

  * MINC-2500    (HF `mcimpoi/minc-2500_split_1`) — material patches, kind=patch
  * OpenFACADES  (HF `seshing/openfacades-dataset`) — building crops, kind=facade
  * URC          (cloned repo)                      — facade crops,   kind=facade
  * London/Scot. (figshare zip)                     — facade crops,   kind=facade

Each source is mapped to Facade-8 via labels.py; sources that are not present
on disk yet are skipped with a warning, so this can be re-run as downloads
finish. Output: data/material_datasets/facade8_manifest.csv

    python src/building_materials_facade/build_dataset.py
    python src/building_materials_facade/build_dataset.py --minc-cap 3000 --limit 50  # smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from building_materials_facade.labels import (  # noqa: E402
    LONDONSCOT_TO_TARGET,
    MINC_TO_TARGET,
    OPENFACADES_TO_TARGET,
    TARGET_CLASSES,
    TARGET_TO_IDX,
    URC_TO_TARGET,
)

ROOT = _SRC.parent
DATA = ROOT / "data" / "material_datasets"
MANIFEST = DATA / "facade8_manifest.csv"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def log(msg: str) -> None:
    print(f"[facade8] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Source indexers — each returns list of dict(path, target, source, kind)
# --------------------------------------------------------------------------- #
def index_minc(cap_per_class: int, limit: int | None) -> list[dict]:
    """Materialise the mapped MINC-2500 facade subset to disk as patch files."""
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover
        log(f"SKIP minc: datasets not importable ({e!r})")
        return []
    log("loading MINC-2500 (mcimpoi/minc-2500_split_1)...")
    try:
        ds = load_dataset("mcimpoi/minc-2500_split_1")
    except Exception as e:
        log(f"SKIP minc: load_dataset failed ({e!r})")
        return []

    out_root = DATA / "minc_facade"
    rows: list[dict] = []
    counts: Counter = Counter()
    for split in [s for s in ("train", "validation") if s in ds]:
        d = ds[split]
        names = d.features["label"].names  # MINC-23 order
        n = len(d) if limit is None else min(limit, len(d))
        for i in range(n):
            label = names[int(d[i]["label"])]
            tgt = MINC_TO_TARGET.get(label)
            if tgt is None or counts[tgt] >= cap_per_class:
                continue
            outdir = out_root / tgt
            outdir.mkdir(parents=True, exist_ok=True)
            p = outdir / f"{split}_{i}.jpg"
            if not p.exists():
                d[i]["image"].convert("RGB").save(p, quality=92)
            counts[tgt] += 1
            rows.append({"path": str(p), "target": tgt, "source": "minc", "kind": "patch"})
        log(f"  minc {split}: kept {sum(counts.values())} so far")
    return rows


def _ensure_unzipped(zip_path: Path, dest: Path, marker_glob: str = "*") -> bool:
    """Extract `zip_path` into `dest` once; return True if `dest` has content."""
    if dest.exists() and any(dest.rglob(marker_glob)):
        return True
    if not zip_path.exists():
        return False
    dest.mkdir(parents=True, exist_ok=True)
    log(f"unzipping {zip_path.name} -> {dest} ...")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
    except zipfile.BadZipFile:
        # Partial/corrupt archive (e.g. download still in progress) — skip for
        # now; a later re-run picks it up once complete.
        log(f"  {zip_path.name} is not a complete zip yet — skipping this source")
        return False
    return any(dest.rglob("*"))


def index_openfacades(cap_per_class: int, limit: int | None) -> list[dict]:
    """Index OpenFACADES building crops by their `surface_material` answer."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:  # pragma: no cover
        log(f"SKIP openfacades: huggingface_hub missing ({e!r})")
        return []
    try:
        jsonl = Path(hf_hub_download("seshing/openfacades-dataset", "jsonl/train.jsonl",
                                     repo_type="dataset"))
        zip_path = Path(hf_hub_download("seshing/openfacades-dataset", "img/train.zip",
                                        repo_type="dataset"))
    except Exception as e:
        log(f"SKIP openfacades: download not ready ({e!r})")
        return []

    img_dest = DATA / "openfacades" / "img"
    if not _ensure_unzipped(zip_path, img_dest, marker_glob="*.png"):
        log("SKIP openfacades: image zip not extractable yet")
        return []
    # Build filename -> absolute path (images may sit in a subfolder of the zip).
    by_name = {p.name: p for p in img_dest.rglob("*") if p.suffix.lower() in IMG_EXT}

    def extract_material(ans: str) -> str | None:
        a = ans.strip()
        if a.startswith("```"):
            a = re.sub(r"^```json|^```|```$", "", a, flags=re.M).strip()
        try:
            j = json.loads(a)
            if isinstance(j, dict) and "surface_material" in j:
                return str(j["surface_material"]).lower()
        except Exception:
            pass
        return a.lower() if (len(a) < 25 and a.isalpha()) else None

    rows: list[dict] = []
    counts: Counter = Counter()
    n = 0
    for line in jsonl.open():
        if limit is not None and n >= limit:
            break
        rec = json.loads(line)
        conv = rec.get("conversations", [])
        for i, turn in enumerate(conv):
            if turn.get("from") == "human" and "material" in turn["value"].lower() and i + 1 < len(conv):
                raw = extract_material(conv[i + 1]["value"])
                tgt = OPENFACADES_TO_TARGET.get(raw) if raw else None
                path = by_name.get(rec.get("image", ""))
                if tgt and path and counts[tgt] < cap_per_class:
                    counts[tgt] += 1
                    n += 1
                    rows.append({"path": str(path), "target": tgt,
                                 "source": "openfacades", "kind": "facade"})
                break
    log(f"  openfacades: kept {len(rows)} ({dict(counts)})")
    return rows


def index_urc(limit: int | None) -> list[dict]:
    """Index URC single-label facade crops (combined `facadematerials-all`)."""
    base = DATA / "urban-resource-cadastre-repository" / "data" / "facadematerials-all" / "train"
    csv_path = base / "_classes.csv"
    if not csv_path.exists():
        log("SKIP urc: not cloned")
        return []
    rows: list[dict] = []
    with csv_path.open() as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader)]
        cols = header[1:]
        for r in reader:
            fname = r[0].strip()
            vals = [int(x) for x in r[1:]]
            pos = [cols[i] for i, v in enumerate(vals) if v == 1]
            if len(pos) != 1:                      # single-label only
                continue
            tgt = URC_TO_TARGET.get(pos[0].lower())
            p = base / fname
            if tgt and p.exists():
                rows.append({"path": str(p), "target": tgt, "source": "urc", "kind": "facade"})
            if limit is not None and len(rows) >= limit:
                break
    log(f"  urc: kept {len(rows)}")
    return rows


def index_londonscot(cap_per_class: int, limit: int | None) -> list[dict]:
    """Index the London/Scotland cladding subset, raw (non-augmented) images only.

    Layout in the zip: ``Exterior Cladding Material/{City}/Before Augmentation/
    Data/{train,val}/{Class}/*.jpg``. We selectively extract ONLY those raw
    images (skipping the multi-GB augmented copies and the unrelated
    stories/SSVI subsets) so augmented near-duplicates can't leak across splits.
    """
    zip_path = DATA / "building_characteristics.zip"
    if not zip_path.exists():
        log("SKIP londonscot: zip not present yet")
        return []
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        log("SKIP londonscot: zip not complete yet")
        return []

    dest = DATA / "london_scotland"
    rows: list[dict] = []
    counts: Counter = Counter()
    for n in zf.namelist():
        if not n.lower().endswith(IMG_EXT):
            continue
        low = n.lower()
        if "exterior cladding material" not in low or "before augmentation" not in low:
            continue
        parts = n.strip("/").split("/")
        if "Data" not in parts:
            continue
        di = parts.index("Data")
        if len(parts) <= di + 2:
            continue
        split_name, cls_folder = parts[di + 1], parts[di + 2]
        tgt = LONDONSCOT_TO_TARGET.get(cls_folder.strip().lower())
        if tgt is None or counts[tgt] >= cap_per_class:
            continue
        outdir = dest / tgt
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / f"{split_name}_{Path(n).name}"
        if not out.exists():
            with zf.open(n) as src, open(out, "wb") as fh:
                fh.write(src.read())
        counts[tgt] += 1
        rows.append({"path": str(out), "target": tgt, "source": "londonscot", "kind": "facade"})
        if limit is not None and len(rows) >= limit:
            break
    log(f"  londonscot: kept {len(rows)} ({dict(counts)})")
    return rows


# --------------------------------------------------------------------------- #
def stratified_split(rows: list[dict], seed: int, val_frac=0.1, test_frac=0.1) -> None:
    """Assign each row a `split` in-place, stratified by target class."""
    rng = random.Random(seed)
    by_t: dict[str, list[dict]] = {}
    for r in rows:
        by_t.setdefault(r["target"], []).append(r)
    for t, items in by_t.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        for r in items[:n_test]:
            r["split"] = "test"
        for r in items[n_test:n_test + n_val]:
            r["split"] = "val"
        for r in items[n_test + n_val:]:
            r["split"] = "train"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minc-cap", type=int, default=3000, help="Max patches per target class from MINC.")
    ap.add_argument("--of-cap", type=int, default=4000, help="Max images per class from OpenFACADES.")
    ap.add_argument("--ls-cap", type=int, default=4000, help="Max images per class from London/Scotland.")
    ap.add_argument("--limit", type=int, default=None, help="Per-source row cap (smoke test).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    rows += index_minc(args.minc_cap, args.limit)
    rows += index_openfacades(args.of_cap, args.limit)
    rows += index_urc(args.limit)
    rows += index_londonscot(args.ls_cap, args.limit)

    if not rows:
        log("ERROR: no samples indexed from any source — check downloads.")
        sys.exit(1)

    for r in rows:
        r["target_idx"] = TARGET_TO_IDX[r["target"]]
    stratified_split(rows, args.seed)

    with MANIFEST.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "target", "target_idx", "source", "kind", "split"])
        w.writeheader()
        w.writerows(rows)

    # Summary table: class x split, and per-source totals.
    log(f"wrote {len(rows)} rows -> {MANIFEST}")
    log("class distribution (train / val / test):")
    for c in TARGET_CLASSES:
        d = {s: sum(1 for r in rows if r["target"] == c and r["split"] == s)
             for s in ("train", "val", "test")}
        log(f"  {c:9s} {d['train']:6d} / {d['val']:5d} / {d['test']:5d}")
    log("per-source totals: " + str(dict(Counter(r["source"] for r in rows))))


if __name__ == "__main__":
    main()
