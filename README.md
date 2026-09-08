# Precise

**Building analysis from a single facade photo.** One image goes in; Precise
returns *what the building looks like* (semantic segmentation), *how big it is*
(floor count + height), and *what it's made of* (wall material + mechanical
properties) — as annotated imagery and structured JSON.

It is a three-module pipeline with one orchestrator. Each module has a
selectable backend, so you can trade zero-shot flexibility for trained accuracy.

```
                       ┌──────────────────────────────────────────────┐
   facade.jpg  ─────▶  │  Module 1 · Facade parsing (SegFormer / SEEM)  │
                       └──────────────────────────────────────────────┘
                                        │ masks · bboxes · polygons · overlay
                          ┌─────────────┴─────────────┐
                          ▼                            ▼
        ┌────────────────────────────┐   ┌───────────────────────────────────┐
        │ Module 2 · Building features │   │ Module 3 · Material classification │
        │  floors + height (m)         │   │  wall material + properties        │
        └────────────────────────────┘   └───────────────────────────────────┘
                          │                            │
                          └────────────┬───────────────┘
                                       ▼
                     m1.json · m2.json · m3.json  +  <stem>_pipeline.png
```

## Contents
- [Quickstart](#quickstart)
- [What each module does](#what-each-module-does)
- [Outputs & JSON schemas](#outputs--json-schemas)
- [Configuration](#configuration)
- [Backends](#backends)
- [Training](#training)
- [Environments](#environments)
- [Paired top-down / facade packages](#paired-top-down--facade-packages)
- [Library use](#library-use)
- [Repository layout](#repository-layout)

---

## Quickstart

Activate the GPU conda env and run the client:

```console
$ conda activate precise-seem-gpu
(precise-seem-gpu) kgyftodimos@IKH-LP207-L:~/Desktop/dev/IKH/Precise$ python src/client.py
```

`src/client.py` accepts an optional path:

```console
(precise-seem-gpu) …/Precise$ python src/client.py                      # browse the packages
(precise-seem-gpu) …/Precise$ python src/client.py --auto                # batch every package
(precise-seem-gpu) …/Precise$ python src/client.py base/cmp_b0003.jpg    # one image
(precise-seem-gpu) …/Precise$ python src/client.py --images              # every base/*.jpg
(precise-seem-gpu) …/Precise$ python src/client.py --images path/to/dir  # every .jpg in a folder
```

**Package mode is the default**: bare `client.py` iterates the paired
top-down/facade dataset in `data/Precise-Data` — see
[Paired top-down / facade packages](#paired-top-down--facade-packages). Plain
facade images need `--images`, except that a positional *file* selects image
mode on its own (a file can never be a folder of pairs), so
`client.py photo.jpg` still works unchanged.

> A positional **directory** is now read as a package folder. The old
> "every `.jpg` in this directory" pass is `--images path/to/dir`.

> **Use the `precise-seem-gpu` env.** The default `python` is a CPU-only torch
> build and cannot drive the RTX 5070. `precise-seem-gpu` has torch 2.10 (cu128)
> plus transformers, timm, datasets, and open_clip.

For **each** input image the pipeline:
1. runs Modules 1 → 2 → 3,
2. writes `m1.json`, `m2.json`, `m3.json` next to the image,
3. writes `<stem>_pipeline.png` — the segmentation overlay (with a color legend)
   plus a text box of floors / height / material,
4. prints a summary panel and a **mechanical-properties table** to the terminal.

The default pipeline is **SegFormer → geometry → DINOv3 Facade-8** (the trained
path). Backends are switchable — see [Backends](#backends).

---

## What each module does

### Module 1 — Facade parsing
**In:** the RGB image. **Out:** one labelled mask per class (+ bbox, contour
polygons, confidence) and a color overlay.

- The **SegFormer** backend (default) segments the photo into the **12 CMP
  classes** — `background, facade, window, door, cornice, sill, balcony, blind,
  deco, molding, pillar, shop` — and draws every class in its own color with an
  on-image legend (`background` is left as the raw photo).
- `facade` (SegFormer) / `house` (SEEM) is taken as the **building extent**;
  `window` and `door` feed Modules 2 and 3.
- The **SEEM** backend is a zero-shot alternative that emits `house/window/door`.

### Module 2 — Building features (geometry heuristics)
**In:** window/door bboxes + the building bbox from Module 1. **Out:** floor
count and building height, each with a confidence and a `height_source`.

- **Floors** — window Y-centers are clustered into rows (a new row starts when
  the vertical gap exceeds `window_cluster_tol_ratio × median window height`).
  Row count = floors.
- **Height** — two paths:
  - `door_scale` *(preferred)*: calibrate meters-per-pixel from the ground-floor
    door (assumed ≈ 2.05 m tall), scale up by the building's pixel height.
    Accepted only if the implied per-floor height is plausible (2.4–4.5 m).
  - `fallback`: `floor_count × assumed_floor_height_m` (3.2 m).
  - `none`: neither doors nor floors were found.

### Module 3 — Material classification
**In:** the image + Module 1's building region. **Out:** ranked wall materials,
the dominant pick, and its reference mechanical properties.

- The wall region is `facade` **minus dilated `window`/`door` openings**.
  Classification is read **strictly from that region** — non-wall pixels are
  neutralised before the model sees them and **there is no whole-image
  fallback**. If Module 1 finds no building, the result is
  `classified_region: "none"`.
- Texture patches are sampled inside the region and their per-material
  probabilities averaged (patch-voting).
- The **DINOv3 Facade-8** backend (default) predicts `brick, concrete, render,
  stone, glass, metal, wood, tile`. Fixed mechanical properties for the winning
  material are attached from `config.yaml`.

---

## Outputs & JSON schemas

Three JSON files are written next to each input image. All coordinates are pixel
`xyxy` for bboxes and `[x, y]` for polygon points; `normalized` variants are the
same values divided by width/height (0–1).

### `m1.json` — facade parsing (`FacadeParsingResult`)

| Field | Type | Meaning |
|---|---|---|
| `image` | `{id, width, height}` | source image id (filename stem) and pixel size |
| `view_type` | `"facade" \| "topdown"` | pipeline view mode |
| `classes[]` | list of `ClassMask` | one entry per class (see below) |
| `metadata` | `{model_version, backbone, prompts[], threshold, timestamp}` | run info; `prompts` = class names |

**`ClassMask`**

| Field | Type | Meaning |
|---|---|---|
| `label` | `str` | class name (`facade`, `window`, `door`, `cornice`, `sill`, `balcony`, `blind`, `deco`, `molding`, `pillar`, `shop`) |
| `confidence` | `float` 0–1 | mean class probability over the class's pixels |
| `pixel_area` | `int` | number of pixels assigned to the class |
| `bbox` | `{pixel[4], normalized[4]}` \| `null` | axis-aligned bounds, `null` if the class is absent |
| `polygons[]` | `[{pixel[[x,y]…], normalized[[x,y]…]}]` | simplified contour(s), one per connected region |
| `mask_path` | `str \| null` | optional path to a saved mask (unused by default) |

```jsonc
{
  "image": { "id": "cmp_b0004", "width": 1024, "height": 691 },
  "view_type": "facade",
  "classes": [
    {
      "label": "facade", "confidence": 0.9011, "pixel_area": 206801,
      "bbox": { "pixel": [0.0, 12.0, 1023.0, 690.0], "normalized": [0.0, 0.017, 0.999, 0.999] },
      "polygons": [ { "pixel": [[1023.0, 0.0], /* … */], "normalized": [[0.999, 0.0], /* … */] } ],
      "mask_path": null
    }
    // … one entry per detected class
  ],
  "metadata": {
    "model_version": "v1.0", "backbone": "segformer-mit-b0",
    "prompts": ["facade", "window", "door", "…"], "threshold": 0.5,
    "timestamp": "2026-08-24T09:35:54.808135Z"
  }
}
```

### `m2.json` — building features (`BuildingFeaturesResult`)

| Field | Type | Meaning |
|---|---|---|
| `image` | `{id, width, height}` | source image |
| `view_type` | `"facade" \| "topdown"` | view mode |
| `predictions.building_height_m` | `{value: float, confidence: 0–1}` | estimated height in metres |
| `predictions.floor_count` | `{value: int, confidence: 0–1}` | estimated number of floors |
| `height_source` | `"door_scale" \| "fallback" \| "none"` | which height method was used |
| `metadata` | `{model_version, timestamp}` | run info |

```jsonc
{
  "image": { "id": "cmp_b0004", "width": 1024, "height": 691 },
  "view_type": "facade",
  "predictions": {
    "building_height_m": { "value": 17.5, "confidence": 0.8789 },
    "floor_count":       { "value": 4,    "confidence": 0.8629 }
  },
  "height_source": "door_scale",
  "metadata": { "model_version": "v1.0", "timestamp": "2026-08-24T09:35:54.811932Z" }
}
```

### `m3.json` — material classification (`MaterialClassificationResult`)

| Field | Type | Meaning |
|---|---|---|
| `image` | `{id, width, height}` | source image |
| `view_type` | `"facade" \| "topdown"` | view mode |
| `materials[]` | `[{label, score}]` | all candidate materials, ranked; `score` is a relative share summing to ~1 |
| `dominant_material` | `{label, score}` | top pick (`label: "none"` if no building region) |
| `dominant_material_properties` | `MaterialProperties \| null` | fixed reference properties of the winner (see below), `null` if unlisted |
| `classified_region` | `"patches" \| "masked_bbox" \| "none"` | how the pixels were obtained: tiled wall patches, a masked region-bbox crop, or nothing (fail-closed) |
| `metadata` | `{candidate_materials[], model}` | class list + backend id |

**`MaterialProperties`** — `category`, `density_kg_m3`, `youngs_modulus_gpa`,
`compressive_strength_mpa`, `tensile_strength_mpa`, `poisson_ratio`, `note`.
These are **static reference values** looked up by label from `config.yaml`
(never computed at runtime).

```jsonc
{
  "image": { "id": "cmp_b0004", "width": 1024, "height": 691 },
  "view_type": "facade",
  "materials": [ { "label": "stone", "score": 0.7129 }, /* … 7 more */ ],
  "dominant_material": { "label": "stone", "score": 0.7129 },
  "dominant_material_properties": {
    "category": "natural-stone", "density_kg_m3": 2650.0, "youngs_modulus_gpa": 50.0,
    "compressive_strength_mpa": 130.0, "tensile_strength_mpa": 10.0,
    "poisson_ratio": 0.25, "note": "granite (representative)"
  },
  "classified_region": "patches",
  "metadata": {
    "candidate_materials": ["brick", "concrete", "render", "stone", "glass", "metal", "wood", "tile"],
    "model": "facade-vit_base_patch16_dinov3.lvd1689m"
  }
}
```

> Per-object mode (`materials_all=False`) instead writes a
> `PerObjectMaterialResult`: `objects` keyed by class label, each an array of
> `{bbox, dominant_material, properties}`.

---

## Configuration

Everything tunable lives in [`config.yaml`](config.yaml):

| Section | Notable keys |
|---|---|
| `pipeline` | `view_type` (`facade`/`topdown`), `materials_all` |
| `facade_parsing` | SEEM `threshold`, `backbone`, `prompts`, `label_colors_bgr` |
| `facade_parsing_segformer` | SegFormer `checkpoint`, `prob_threshold`, `overlay_alpha` |
| `building_features` | `assumed_door_height_m`, `assumed_floor_height_m`, `window_cluster_tol_ratio`, `plausible_floor_height_m` |
| `building_materials` | `material_backend`, `materials`, `material_properties`, patch/wall knobs (`facade_patch_size`, `facade_min_wall_ratio`, `wall_opening_dilation_ratio`), `facade_checkpoint`, `minc_checkpoint`, SigLIP2 `model_name` |

`main()` in [`src/client.py`](src/client.py) exposes three switches:
`manual_seg` (Module 1: `True`=SegFormer, `False`=SEEM), `materials_all`
(whole-wall vs. per-object material), and the Module 3 backend is chosen by
`building_materials.material_backend`.

---

## Backends

| Module | Default | Alternatives |
|---|---|---|
| 1 · parsing | **SegFormer** (trained on CMP, 12 classes) | SEEM (zero-shot, `house/window/door`) |
| 3 · materials | **DINOv3 Facade-8** (trained; adds `concrete`+`render`) | MINC (ConvNeXt on MINC-2500) · SigLIP2 (zero-shot open_clip) |

The DINOv3 Facade-8 model reaches **macro-F1 ≈ 0.81** on a held-out facade test
set, vs ≈ 0.57 for MINC mapped to the same classes (MINC has no
`concrete`/`render`). See
[`src/building_materials_facade/README.md`](src/building_materials_facade/README.md).

---

## Training

Weights and datasets are **not** in git; reproduce them (datasets auto-download):

```bash
GPU=~/miniconda3/envs/precise-seem-gpu/bin/python

# Module 1 — SegFormer on the CMP Facade Database (base/), 12 classes
$GPU src/facade_parsing_segm/segformer_cmp/train.py --backbone nvidia/mit-b4

# Module 3 — DINOv3 Facade-8 (build the pooled manifest, then train + benchmark)
$GPU src/building_materials_facade/build_dataset.py
$GPU src/building_materials_facade/train.py --epochs 15
$GPU src/building_materials_facade/benchmark.py

# Module 3 (alt) — MINC classifier
$GPU src/building_materials_minc/train.py
```

Each writes a self-describing best checkpoint under `runs/` (gitignored) that
`config.yaml` already points to.

---

## Environments

| Env | torch | Use |
|---|---|---|
| `precise-seem-gpu` | 2.10 **cu128** | **default GPU path** — SegFormer + DINOv3/MINC/SigLIP2 |
| `precise-seem` | 2.1.0 **CPU** | the SEEM backend only (its detectron2 fork pins torch 2.1) |

SEEM is not available in `precise-seem-gpu`; run the SegFormer backend there
(the default), or SEEM in `precise-seem` on CPU.

---

## Paired top-down / facade packages

`data/Precise-Data` holds one building per numeric id as **two** images:
`<id>.png` (top-down / aerial) and `Fac<id>.png` (street-level facade). That
pair is a **package**. `src/building_packages/` iterates them, shows both views
in one window, and feeds the facade half to Modules 1–3. This is what
`src/client.py` does by default:

```console
(precise) …/Precise$ python src/client.py                  # browse interactively
(precise) …/Precise$ python src/client.py --auto            # batch all packages
(precise) …/Precise$ python src/client.py --show            # display only, no models
(precise) …/Precise$ python src/client.py --show --auto     # slideshow of every pair
(precise) …/Precise$ python src/client.py --ids 3 7 12      # only these ids
(precise) …/Precise$ python src/client.py --list            # index and exit
(precise) …/Precise$ python src/client.py path/to/pairs     # a different package folder
```

`--packages [DIR]` still selects this mode explicitly and takes precedence over
the positional target; it is redundant now that it is the default. See
[Quickstart](#quickstart) for `--images`, which processes plain facade images.

**The window shows only the two images** — no titles, captions, filenames,
counters or key legend. Which package is on screen, its pipeline result, and the
key reference are printed on the terminal instead.

`--show` (alias `--no-run`) is the display-only path: it iterates and shows the
pairs without importing torch, loading a checkpoint, or running the pipeline.
Modules 1–3 are imported lazily inside the methods that use them, so `--show`
starts instantly and works in an environment that cannot run inference at all.
Combine it with `--auto` for an unattended slideshow — handy for eyeballing a
folder of pairs before committing to inference.

**One pair in memory at a time.** Discovery matches filenames only and never
decodes pixels, so indexing is cheap. The iteration then decodes a package on
entry and releases it before the next id is decoded — peak memory is a single
pair regardless of folder size:

```python
from building_packages import discover_packages, iter_loaded, compose_package_view

index = discover_packages()                 # 20 packages, all unloaded
for package in iter_loaded(index.packages):  # exactly one resident
    canvas = compose_package_view(package)   # top-down beside facade, BGR
    # package.topdown / package.facade are [H, W, 3] uint8; released on the next step
```

**Interactive keys** (printed on the terminal at startup, not drawn on the canvas):

| key | action |
| --- | --- |
| `→` / `d` | next package |
| `←` / `a` | previous package (wraps at both ends) |
| `r` | run Modules 1–3 on this package's facade |
| `q` / `esc` | quit |

Keys are read with `cv2.waitKeyEx`, not `cv2.waitKey`: the latter is defined as
`waitKeyEx(delay) & 0xff`, which masks every arrow key to `0` and makes arrow
navigation impossible. An unrecognized press reports its raw code on the
terminal instead of being silently ignored, so a backend with different arrow
codes is diagnosable on the spot.

**Only the facade is analysed.** Module 1 (CMP SegFormer / SEEM) and Module 2
(floors from window/door rows) are street-level models, so the top-down image is
carried for display and context rather than pushed through them. Outputs go to
one directory per package — `data/package_runs/package-07/` — because all pairs
share a single input folder and would otherwise overwrite each other's
`m1/m2/m3.json`. `Pipeline.run()` takes the matching `out_dir` argument:

```python
Pipeline(cfg, use_segformer=True).run(image="…/Fac7.png", out_dir="data/package_runs/package-07")
```

Inference failures are captured per package (`PackageRun.error`) instead of
raised, so one bad image cannot end a batch pass. Missing checkpoints or Python
packages are reported once at startup, and browsing still works without them.

---

## Library use

```python
from config import load_config
from facade_parsing_segm.segformer_cmp.pipeline_adapter import FacadeSegformerParser
from building_materials_facade import FacadeMaterials

cfg = load_config()
parse = FacadeSegformerParser("base/cmp_b0001.jpg", cfg.facade_parsing_segformer).parse()
mats  = FacadeMaterials("base/cmp_b0001.jpg", cfg.building_materials).classify()
# parse.classes[i].label / .bbox / .polygons
# mats.dominant_material.label / .score ; mats.dominant_material_properties ; mats.classified_region
```

---

## Repository layout

```
Precise/
├── config.yaml                       # single source of truth for all knobs
├── src/
│   ├── client.py                     # CLI entry point — Pipeline, images + --packages
│   ├── config.py                     # pydantic config loader
│   ├── building_packages/            # top-down/facade pairs: iterate, display, run
│   ├── facade_parsing/               # Module 1 backend A: SEEM (zero-shot)
│   ├── facade_parsing_segm/
│   │   └── segformer_cmp/            # Module 1 backend B: SegFormer on CMP (12 classes)
│   ├── building_features/            # Module 2: geometry heuristics
│   ├── building_materials/           # Module 3 backend A: SigLIP2 + shared region sampler
│   ├── building_materials_minc/      # Module 3 backend B: MINC classifier
│   └── building_materials_facade/    # Module 3 backend C: DINOv3 Facade-8 (default)
├── base/                             # CMP Facade Database (Module 1 data)        [gitignored]
├── data/Precise-Data/                # paired <id>.png / Fac<id>.png packages      [gitignored]
├── data/package_runs/                # per-package pipeline outputs                [gitignored]
├── data/material_datasets/           # pooled material datasets + manifest        [gitignored]
└── src/**/runs/                      # trained checkpoints / training outputs      [gitignored]
```
