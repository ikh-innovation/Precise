"""Benchmark the DINOv3 Facade-8 model against the MINC backend.

Evaluates both checkpoints on the SAME Facade-8 test split (from the manifest),
classifying each test crop with a single centre-crop forward pass. MINC's
8-class predictions are mapped into Facade-8 where a correspondence exists;
MINC has no `concrete` and no `render` class, so those test rows are counted as
misses for MINC — which is precisely the coverage gap this work closes.

    python src/building_materials_facade/benchmark.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from building_materials_facade.data import ManifestDataset, build_transform  # noqa: E402
from building_materials_facade.labels import TARGET_CLASSES, TARGET_TO_IDX  # noqa: E402
from building_materials_facade.model import build_classifier as build_facade  # noqa: E402

MANIFEST = _SRC.parent / "data" / "material_datasets" / "facade8_manifest.csv"

# MINC label -> Facade-8 label (None = no Facade-8 equivalent, always a miss).
MINC_PRED_TO_FACADE8 = {
    "brick": "brick", "glass": "glass", "metal": "metal", "stone": "stone",
    "tile": "tile", "wood": "wood", "painted": "render", "plastic": None,
}


def load_test_rows() -> list[dict]:
    with MANIFEST.open() as fh:
        return [r for r in csv.DictReader(fh) if r["split"] == "test"]


@torch.no_grad()
def predict(model, rows, mean, std, input_size, device, label_map=None) -> np.ndarray:
    """Return Facade-8 predicted indices for each test row (label_map for MINC)."""
    ds = ManifestDataset(
        [{**r, "kind": "patch"} for r in rows],  # eval = deterministic centre crop
        mean, std, input_size, train=False,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)
    out = []
    for x, _ in tqdm(loader, desc="predict", leave=False):
        logits = model(x.to(device))
        out.append(logits.argmax(1).cpu().numpy())
    raw = np.concatenate(out)
    if label_map is None:
        return raw  # already Facade-8 indices
    # Map source-class indices -> Facade-8 indices (-1 = no equivalent).
    src_names, mapped = label_map
    return np.array([
        TARGET_TO_IDX.get(MINC_PRED_TO_FACADE8.get(src_names[i]) or "", -1) for i in raw
    ])


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    overall = float((y_true == y_pred).mean())
    macro = float(f1_score(y_true, y_pred, average="macro",
                           labels=list(range(len(TARGET_CLASSES))), zero_division=0))
    print(f"\n=== {name}:  acc={overall:.4f}  macroF1={macro:.4f} ===")
    for i, c in enumerate(TARGET_CLASSES):
        m = y_true == i
        n = int(m.sum())
        acc = float((y_pred[m] == i).mean()) if n else float("nan")
        print(f"  {c:9s} n={n:4d}  acc={acc:.3f}")


def main() -> None:
    if not MANIFEST.exists():
        sys.exit("manifest missing — run build_dataset.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_test_rows()
    y_true = np.array([int(r["target_idx"]) for r in rows])
    print(f"[benchmark] {len(rows)} Facade-8 test crops")

    # --- DINOv3 Facade-8 model ---
    fck = torch.load("src/building_materials_facade/runs/best.pt", map_location="cpu")
    fmodel = build_facade(len(fck["class_names"]), arch=fck["arch"], pretrained=False,
                          unfreeze_blocks=fck.get("unfreeze_blocks", 0),
                          hidden=fck.get("hidden", 512), dropout=fck.get("dropout", 0.3))
    fmodel.load_state_dict(fck["model_state"])
    fmodel.to(device).eval()
    dc = fck["data_config"]
    y_facade = predict(fmodel, rows, dc["mean"], dc["std"], dc["input_size"][-1], device)
    report("DINOv3 Facade-8 (new)", y_true, y_facade)

    # --- MINC ConvNeXt model (mapped into Facade-8) ---
    minc_path = Path("src/building_materials_minc/runs/best.pt")
    if minc_path.exists():
        from building_materials_minc.model import build_classifier as build_minc
        mck = torch.load(minc_path, map_location="cpu")
        mmodel = build_minc(len(mck["class_names"]), arch=mck.get("arch", "convnext_tiny"), pretrained=False)
        mmodel.load_state_dict(mck["model_state"])
        mmodel.to(device).eval()
        mdc = mck["data_config"]
        y_minc = predict(mmodel, rows, mdc["mean"], mdc["std"], mdc["input_size"][-1], device,
                         label_map=(mck["class_names"], True))
        report("MINC ConvNeXt (old, mapped to Facade-8)", y_true, y_minc)
        # Spell out the structural gap.
        for gap in ("concrete", "render"):
            gi = TARGET_TO_IDX[gap]
            n = int((y_true == gi).sum())
            hit = int(((y_true == gi) & (y_minc == gi)).sum())
            print(f"  [gap] MINC on '{gap}': {hit}/{n} correct (no '{gap}' class in MINC)")
    else:
        print("[benchmark] MINC checkpoint not found — skipping comparison")


if __name__ == "__main__":
    main()
