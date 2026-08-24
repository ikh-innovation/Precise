# Precise

Computer-vision pipeline for building analysis from a single facade image.
Three modules and one orchestrator, each module with a selectable backend:

| # | Module               | Backends                                              | Output |
|---|----------------------|-------------------------------------------------------|--------|
| 1 | facade parsing       | **SegFormer** (trained, default) · SEEM (zero-shot)   | semantic masks (12 CMP classes for SegFormer; house/window/door for SEEM) + bboxes/polygons |
| 2 | `building_features`  | heuristics                                            | floor count + building height (m) |
| 3 | materials            | **DINOv3 Facade-8** (trained, default) · MINC · SigLIP2 (zero-shot) | dominant wall material (read **only** from the building region) + fixed mechanical properties |
| – | `client.Pipeline`    | —                                                     | runs all three, writes JSON + an annotated PNG |

```
Precise/
├── config.yaml                       # single source of truth for all knobs
├── src/
│   ├── client.py                     # CLI entry point — Pipeline + main()
│   ├── config.py                     # pydantic config loader
│   ├── facade_parsing/               # Module 1 backend A: SEEM (zero-shot)
│   ├── facade_parsing_segm/
│   │   └── segformer_cmp/            # Module 1 backend B: SegFormer trained on CMP (12 classes)
│   ├── building_features/            # Module 2: geometry heuristics
│   ├── building_materials/           # Module 3 backend A: SigLIP2 (zero-shot) + shared region sampler
│   ├── building_materials_minc/      # Module 3 backend B: MINC-trained classifier
│   └── building_materials_facade/    # Module 3 backend C: DINOv3 Facade-8 (default)
├── base/                             # CMP Facade Database (Module 1 train data)   [gitignored]
├── data/material_datasets/           # pooled material datasets + manifest         [gitignored]
└── src/**/runs/                      # trained checkpoints / training outputs       [gitignored]
```

The default pipeline is **SegFormer → geometry → DINOv3 Facade-8 classifier**,
the trained/supervised path. The other backends (SEEM, MINC, SigLIP2) remain
selectable for comparison.

> **Weights and data are not in git.** `base/`, `data/`, and every `runs/`
> checkpoint are gitignored (too large / not source). Reproduce them with the
> training commands below; the datasets download automatically.

## Environments

Two conda envs, by hardware:

| Env | torch | Use |
|-----|-------|-----|
| `precise-seem`     | 2.1.0 **CPU** | the SEEM backend (its detectron2 fork is pinned to torch 2.1) |
| `precise-seem-gpu` | 2.10 **cu128** (+ transformers 4.45, timm, datasets, open_clip) | the **default GPU path** — SegFormer + DINOv3/MINC/SigLIP2 on the RTX 5070 |

Run the pipeline on GPU:

```bash
~/miniconda3/envs/precise-seem-gpu/bin/python src/client.py                    # all base/*.jpg
~/miniconda3/envs/precise-seem-gpu/bin/python src/client.py path/to/facade.jpg  # one image
~/miniconda3/envs/precise-seem-gpu/bin/python src/client.py path/to/dir         # a directory
```

The default `python` is CPU-only torch and cannot use the GPU. SEEM is **not**
available in `precise-seem-gpu` (its detectron2 build needs torch 2.1); use
`manual_seg=True` (the default) there, or run SEEM in `precise-seem` on CPU.

## Usage

`main()` in [src/client.py](src/client.py) takes three switches:

```python
def main(img_path: str, materials_all=True, manual_seg=True):
    ...
```

- `manual_seg` — **Module 1 backend**: `True` → trained SegFormer, `False` → SEEM.
- `materials_all` — whole-wall material (True) vs. per-object (False).
- **Module 3 backend** is chosen in `config.yaml` (`building_materials.material_backend`: `facade` | `minc` | `siglip2`).

Each run writes `m1.json`, `m2.json`, `m3.json` and `<stem>_pipeline.png` next
to the input image, plus a terminal panel with floors / height / dominant
material and a **mechanical-properties table** for the detected material.

## Configuration

Everything tunable lives in `config.yaml`:

| Section | Notable keys |
|---------|--------------|
| `pipeline` | `view_type` (`facade`/`topdown`), `materials_all` |
| `facade_parsing` | SEEM `threshold`, `backbone`, `prompts`, `label_colors_bgr` |
| `facade_parsing_segformer` | SegFormer `checkpoint`, `prob_threshold`, `overlay_alpha` |
| `building_features` | `assumed_door_height_m`, `assumed_floor_height_m`, `window_cluster_tol_ratio`, `plausible_floor_height_m` |
| `building_materials` | `material_backend`, `materials`, `material_properties`, patch/wall knobs (`facade_patch_size`, `facade_min_wall_ratio`, `wall_opening_dilation_ratio`), `facade_checkpoint`, `minc_checkpoint`, SigLIP2 `model_name` |

## Algorithms

### Module 1 — Facade parsing (SegFormer, default; SEEM optional)

Output (either backend): one `ClassMask` per class with confidence, pixel area,
bbox, and contour polygons; plus an in-memory color overlay reused as the base
of the final PNG. Module 2 treats `facade` (SegFormer) or `house` (SEEM) as the
building extent.

**SegFormer backend** (`facade_parsing_segm/segformer_cmp`, `manual_seg=True`):
SegFormer-B4 fine-tuned on the **CMP Facade Database** (`base/`, **12 classes**;
train with `segformer_cmp/train.py`, backbone `nvidia/mit-b0`…`mit-b5`). The
overlay draws **every facade-element class in its own color with a legend** —
facade (red), window (blue), door (green), cornice, sill, balcony, blind, deco,
molding, pillar, shop; only the `background` catch-all is left as the raw photo.
The adapter (`FacadeSegformerParser`) emits the same `FacadeParsingResult`
schema as SEEM, so it's a drop-in. Runs on GPU.

**SEEM backend** (`facade_parsing`, `manual_seg=False`): zero-shot,
text-prompted semantic segmentation (house/window/door). Runs on CPU in
`precise-seem`.

### Module 2 — Building features (heuristics)

Input: window/door bboxes (one `RETR_EXTERNAL` polygon = one instance) and the
optional building bbox. Output: floor count, height (m), `height_source`.

- **Floors** — cluster window Y-centers into rows: start a new row when the gap
  exceeds `window_cluster_tol_ratio × median_window_height` (default 0.6). Row
  count = floors.
- **Height, Path 1 `door_scale`** (preferred) — pick the ground-level front
  door, set `meters_per_pixel = assumed_door_height_m / door_px_height`
  (door ≈ 2.05 m), multiply by the building's pixel height. Accepted only if the
  implied per-floor height ∈ `plausible_floor_height_m` (default [2.4, 4.5] m).
- **Height, Path 2 `fallback`** — `floor_count × assumed_floor_height_m` (3.2 m).
  `height_source: "none"` if neither doors nor floors exist.

### Module 3 — Material classification (DINOv3 Facade-8, default; MINC / SigLIP2 optional)

Goal: the **building-wall** material — not glass/doors, and **never the whole
image**. The material is read **strictly from the red building region** (a
shared contract in `building_materials/region.py`):

1. **Wall mask** (`client._wall_region_mask`): rasterize the facade/house region
   minus the window/door polygons, dilated by `wall_opening_dilation_ratio` so
   patches stay off (reflective) window/door frames.
2. **Region-only patch sampling** (`sample_region_patches`): non-wall pixels are
   neutralised (filled with the region mean) **before** the encoder sees them,
   then square patches are sampled inside the mask (purest-wall first). There is
   **no whole-image fallback** — an empty region yields `classified_region:
   "none"` rather than classifying sky/ground/neighbours.
3. **Vote**: classify all patches in one batch and average the per-material
   probabilities. Fixed mechanical properties for the dominant material come
   from `config.material_properties` and are printed as a table.

**DINOv3 Facade-8 backend** (`building_materials_facade`, `material_backend:
facade`): a self-supervised **DINOv3 ViT-B/16** backbone (frozen probe) + MLP
head, trained on a pooled facade-domain corpus (MINC patches + OpenFACADES + URC
+ London/Scotland). Outputs the **Facade-8** taxonomy — **brick, concrete,
render, stone, glass, metal, wood, tile** — which, unlike MINC, includes
`concrete` and `render`/plaster. Test macro-F1 ≈ 0.81 (vs ≈ 0.57 for MINC mapped
to the same classes). See [`building_materials_facade/README.md`](src/building_materials_facade/README.md).

**MINC backend** (`building_materials_minc`, `material_backend: minc`): timm
**ConvNeXt-Tiny fine-tuned on MINC-2500** — 8 classes (brick, glass, metal,
painted, plastic, stone, tile, wood); no `concrete`/`render`.

**SigLIP2 backend** (`building_materials`, `material_backend: siglip2`):
zero-shot `open_clip` SigLIP2 (`ViT-L-16-SigLIP2-384`), sigmoid scoring over a
candidate-material list with view-aware text prompts.

### Pipeline orchestration (`client.Pipeline`)

1. Module 1 (SegFormer or SEEM) → masks/bboxes + overlay.
2. Window/door bboxes + building bbox → Module 2.
3. Wall mask (facade − dilated openings) → Module 3, whole-wall
   (`materials_all=True`) or per-object.
4. Write `m1/m2/m3.json` + `<stem>_pipeline.png`.

## Training

```bash
GPU=~/miniconda3/envs/precise-seem-gpu/bin/python

# Module 1 — SegFormer on CMP (base/), 12 classes
$GPU src/facade_parsing_segm/segformer_cmp/train.py --backbone nvidia/mit-b4

# Module 3 — DINOv3 Facade-8 (downloads + unifies the facade datasets, then trains)
$GPU src/building_materials_facade/build_dataset.py     # -> data/material_datasets/facade8_manifest.csv
$GPU src/building_materials_facade/train.py --epochs 15
$GPU src/building_materials_facade/benchmark.py          # DINOv3 vs MINC on the same test split

# Module 3 (alt) — MINC material classifier (downloads MINC-2500 from HF)
$GPU src/building_materials_minc/train.py
```

Each saves a self-describing best checkpoint under `runs/` (gitignored) that the
inference code and `config.yaml` already point to.

## Library use

```python
from config import load_config
from facade_parsing_segm.segformer_cmp.pipeline_adapter import FacadeSegformerParser
from building_materials_facade import FacadeMaterials

cfg = load_config()
parse = FacadeSegformerParser("base/cmp_b0001.jpg", cfg.facade_parsing_segformer).parse()
mats  = FacadeMaterials("base/cmp_b0001.jpg", cfg.building_materials).classify()
# mats.dominant_material.label / .score ; mats.dominant_material_properties ; mats.classified_region
```
