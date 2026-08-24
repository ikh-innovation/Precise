# Module 3 — DINOv3 Facade-8 material backend

A facade-domain replacement for the MINC material backend. It predicts the
dominant **building wall** material from the red building region that Module 1
produces, using a self-supervised **DINOv3 ViT-B/16** backbone (frozen probe by
default) with a small MLP head, trained on a pooled, facade-specific corpus.

## Why this exists

The MINC backend (`building_materials_minc`) is trained on MINC-2500, a generic
in-the-wild material dataset. Two problems for facades:

1. **No `concrete` and no `render`/plaster class** — the two most common
   European facade surfaces. MINC's `painted` is a *finish*, not a material.
2. **Domain gap** — MINC patches are object-centric close-ups, not facades.

This backend fixes both: a facade taxonomy that includes `concrete` and
`render`, and training data drawn from real facades.

## Taxonomy — Facade-8

```
brick · concrete · render · stone · glass · metal · wood · tile
```

* `render` = plaster / stucco / EIFS coatings.
* `metal` folds steel + aluminium (visually near-identical on a facade).

## Data sources (mapped in `labels.py`, pooled by `build_dataset.py`)

| Source | Role | kind | Key classes |
|---|---|---|---|
| MINC-2500 (`mcimpoi/minc-2500_split_1`) | material patches | `patch` | brick, glass, metal, stone, wood, **tile** |
| OpenFACADES (`seshing/openfacades-dataset`) | building crops | `facade` | **concrete, render**, + all others |
| Urban Resource Cadastre (cloned repo) | facade crops | `facade` | brick, render(stucco), wood, stone, metal |
| London/Scotland (figshare 25931941) | facade crops | `facade` | brick, concrete, stone, glass(curtain-wall) |

Single-label images only from the facade sources; whole-building crops become
weakly-labeled material patches via wide `RandomResizedCrop` (see `data.py`).
The pooled manifest is `data/material_datasets/facade8_manifest.csv` with a
stratified train/val/test split.

## Model

`model.py` — DINOv3 backbone (`vit_base_patch16_dinov3.lvd1689m`, 768-d, 256px)
+ `LayerNorm → Linear → GELU → Dropout → Linear` head.

* **Frozen probe (default):** backbone frozen, only the head trains (~0.4M
  params). Robust to the weak labels, trains in minutes on an 8 GB GPU.
* **Fine-tune:** `--unfreeze-blocks N` re-enables the last N transformer blocks
  at a low LR for extra accuracy.

## Usage

```bash
# 1. Acquire + unify datasets -> manifest (idempotent; re-run as downloads land)
python src/building_materials_facade/build_dataset.py

# 2. Train (frozen probe). Self-describing checkpoint -> runs/best.pt
python src/building_materials_facade/train.py --epochs 15 --batch-size 64

# 3. Benchmark vs the MINC backend on the same Facade-8 test split
python src/building_materials_facade/benchmark.py
```

Enable it in `config.yaml`:

```yaml
building_materials:
  material_backend: facade   # siglip2 | minc | facade
  facade_checkpoint: src/building_materials_facade/runs/best.pt
  facade_arch: vit_base_patch16_dinov3.lvd1689m
```

## Inference contract

`classifier.FacadeMaterials` mirrors `BuildingMaterials` / `MincMaterials`:
`classify(region_mask) -> MaterialClassificationResult` and
`classify_instances(detections) -> PerObjectMaterialResult`. It samples wall
patches **strictly inside** Module 1's building region via the shared
`building_materials.region.sample_region_patches`, so the "classify only the
red region, never the whole image" rule is enforced here too — an empty region
yields `classified_region="none"`.

All runs use the `precise-seem-gpu` conda env (torch cu128, GPU).
