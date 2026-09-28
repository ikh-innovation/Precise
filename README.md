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
- **Door recovery** (`segformer_cmp/door_refine.py`, SegFormer backend only).
  `door` is the rarest class in CMP (1.26% of training pixels vs 2.98% for
  `shop`), so a glass entrance set into a shopfront loses the argmax to
  `shop` and only its dark lintel survives. Three steps fix that, all from
  the same forward pass: the prior is divided out of `door`'s score against
  `shop` (logit adjustment), each door is walked *down* through the glazing
  to its base, and the mask is widened sideways into the opening. The first
  two feed Module 2's scale anchor; the third is **presentation only** —
  widening cannot change a door's height but it does change its aspect, which
  is what Module 2 uses to tell a door from a drainpipe, so the geometry
  Module 2 reads stays un-widened. See that module's docstring for what was
  measured and what was rejected.
- **Rectified second opinion** (`segformer_cmp/rectify.py`). CMP is rectified
  head-on facades; these are Street View captures from across a road. Warping
  the held-out CMP split by a known homography measures the cost directly:
  a 0.10 oblique warp drops **door IoU 31%** (0.523 → 0.362) where even
  severe look-up drops it 17% — obliquity is what the door class cannot
  survive, and door is what Module 2 measures scale from. So the facade is
  warped fronto-parallel — vanishing points from its own line segments
  (restricted to the facade mask, or the road wins), focal length from their
  orthogonality, `H = K·Rᵀ·K⁻¹` — and the model is asked again. The merge is
  **add-only** and restricted to the same classes a door may outbid: the
  checkpoint loses doors on a rectified frame about as often as it finds
  them, so only gains are trusted. Package 20's gated entrance, at p(door)
  = 0.02 in the straight-on pass and out of reach of every threshold, prior
  and resolution change tried, comes back at 0.80. Fires on 9/20; costs one
  extra forward pass.

### Module 1b — Scene gate (Cityscapes)
**In:** the RGB image + Module 1's facade mask. **Out:** a refined building
mask, a tightened extent, and a bracket on where the building's base really is.

Module 1's CMP checkpoint is trained on tightly-cropped, head-on facade
photographs in which essentially the whole frame *is* facade — so it has never
had to learn what "not a facade" looks like. On street-level captures it labels
hedges, parked cars, pavement and sky as `facade` and returns a blob. Measured
on the 20-building sample, **24–54% of the mask was street rather than
building** in the worst cases.

A **Cityscapes-finetuned SegFormer** *is* in domain for street photography, so
it is used as a gate: keep the parts of Module 1's mask that the street model
also calls `building`/`wall`. Module 1 keeps the job CMP actually taught it —
`window`, `door`, `sill`, `balcony`.

Two details carry most of the value:

- **Occluders raise the lower edge; they do not reveal the base.** A hedge in
  front of a plinth is building *behind* vegetation, so simply deleting it
  leaves the mask ending above the true base and the building measures short.
  The gate instead reports a **bracket** — the lowest wall actually seen, and
  the row where the ground in front of it begins — and Module 2 places the base
  inside it using the storey pitch. Skipping this biased heights short by
  ~12% at the median across the sample (28.8 m vs 32.6 m).
- **An abutting neighbour cannot be separated here.** Both wings are genuinely
  `building` to any segmenter; telling them apart is a depth question, not a
  semantic one. The residual is reported as `components_kept` rather than
  hidden.

The refined mask also gates Module 3, so material patches are not sampled off
a street tree standing in front of the wall, and Module 1's overlay is
re-rendered from it so the PNG shows the region the pipeline actually used.
Window and door detections outside the refined *extent* are dropped — a "door"
found in a hedge would otherwise become a 2.05 m scale anchor. That test is
against the extent, not the mask: a tree crossing the facade holes the mask,
and a window seen between the leaves is still a window on this building.

Disable with `scene_parsing.enabled: false`; weights download from the Hub on
first run (~250 MB, cached), and the gate adds roughly 1–2 s per image.

### Module 2 — Building features (geometry heuristics)
**In:** window/door bboxes + the building bbox from Module 1. **Out:** floor
count and building height, each with a confidence, plus a `height_source` and a
`scale` block recording how metres were recovered.

- **Floors** — window Y-centers are clustered into rows (a new row starts when
  the vertical gap exceeds `window_cluster_tol_ratio × median window height`).
  Rows too sparse to be a storey are dropped, and a gap spanning two storeys
  counts as two, so a floor whose windows were all missed is still counted.
- **Height** — *measured*, in two steps
  ([`scale.py`](src/building_features/scale.py)):

  1. **Vertical model.** A street photo of a tall block looks *up*, so
     metres-per-pixel shrinks toward the roof. On a vertical facade plane
     `t(y) = 1/(y − y_vp)` is affine in real height, and equally spaced storeys
     land on an arithmetic progression in `t` — so the vanishing point `y_vp`
     is solved from the facade's own storey rhythm, with no camera metadata
     (these are screenshots; there is no EXIF) and no calibration target. The
     fit is bounded by how far it may extrapolate past the outermost window
     row, which is where an unconstrained vanishing point does its damage.
  2. **Scale anchors, fused.** Each cue yields the same quantity — metres per
     unit of `t` — so they combine rather than compete: the **door** (≈ 2.05 m,
     when one is visible), the **storey pitch** (≈ 3.0 m, available whenever
     the facade shows enough window rows), and the **median window** (≈ 1.45 m,
     a weak third opinion). They are merged by inverse-variance weighting in
     log space after anchors that disagree with the median are discarded —
     which is what catches a "door" mask that actually caught a shopfront
     shutter.
  3. **The door class is not allowed to lose on a technicality.** `door` is
     the rarest class in CMP — 1.26% of training pixels against 2.98% for
     `shop` — so the model carries a prior that argues against it, and `shop`
     (the shopfront glazing) is exactly what a glass entrance looks like. In
     17 of 20 packages the shop region is 4–30× the door region and covers the
     same rows, so what reaches Module 2 is the dark lintel above the leaves.
     Module 1 therefore rescores `door` with the training prior divided out
     (logit adjustment) against `shop` only, then walks the door down through
     the glazing to its base. Both steps only ever grow a door *downward*,
     which is the one direction the physics supports — a door reaches the
     ground, and so does the glazing it sits in. Letting it grow any other way
     was measured and was worse; letting it outbid `facade` is worse still,
     because `facade` is the building extent the height is measured from.
     Because growing doors changes which one is tallest, the anchor is now
     also picked with a shape gate (`door_max_aspect`): a 12 px strip beside
     an ATM is not a door, however tall it grows.
  4. **A truncated door is put back on the ground.** A door reaches the
     ground, so one whose mask stops short of the resolved base row is
     missing part of itself — hidden behind a parked car, or lost to the
     shopfront glazing it is set into, which CMP labels `shop` rather than
     `door`. A short span makes the building *tall*, and by the most heavily
     weighted anchor there is. The foot is therefore moved down to the base
     row, but only where the storey pitch says the door is too short to be a
     whole one (2.05 m in a 3 m storey is ≈ 0.68 of a pitch), and only as far
     as still leaves a plausible door. Anything already door-sized is left
     exactly as detected. The pitch decides only *whether* to trust the foot;
     what replaces it is the base row, so the anchor keeps its independence
     from the pitch anchor it is fused with. `snapped_px` on the door anchor
     records the correction. On the 20-building sample it fires once — the
     Santander entrance in package 19 — and leaves the other nineteen
     untouched.

  Together, steps 3 and 4 move the door anchor's median error against 2.05 m
  from 0.266 to 0.245 in log terms (0.209 → 0.170 over the anchors actually
  fused): package 9 goes 1.26 m → 2.01 m, package 19 0.72 m → 2.07 m, and
  package 16's 9 px-wide sliver is dropped rather than believed. Nothing
  regresses and nothing falls back to `floor_count × 3.2`. Several more
  packages get a visibly fuller door in the overlay without their anchor
  moving at all.

  Height is then the **facade mask's full extent** through that scale, so the
  taller shopfront storey, the parapet and any setback penthouse all count.
  `height_source` names what carried it: `door_scale`, `floor_pitch`,
  `window_scale`, `fused`, or — only when no anchor at all could be formed, or
  the result failed its storey-budget sanity check — `fallback`
  (`floor_count × assumed_floor_height_m`), else `none`.

  On the 20-building Valencia sample this moves 17/20 from `fallback` to a
  measured height — **20/20 once the Module 1b scene gate cleans the mask** —
  with recovered storey pitches of 2.4–3.4 m. Against a synthetic pinhole
  camera with known ground truth (12–40 m blocks, look-up angles 4°–40°) the
  worst error is **2.8 %**.

  Accuracy on the real sample is *unverified*: there is no ground truth for
  these buildings yet, so the only error figure that means anything is the
  synthetic one.

### Module 3 — Material classification
**In:** the image + Module 1's building region. **Out:** ranked wall materials,
the dominant pick, and its reference mechanical properties.

- The wall region is `facade` **minus dilated `window`/`door` openings**,
  taken from Module 1's rasters when it keeps them rather than from the
  simplified polygons — the raster carries the opening's true outline,
  including the widened door, which is what keeps shopfront glazing out of
  the pixels the material is read from.
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
| `height_source` | `"door_scale" \| "floor_pitch" \| "window_scale" \| "fused" \| "fallback" \| "none"` | which cue carried the height |
| `scale` | `ScaleReport \| null` | how pixels became metres; `null` when the height fell back |
| `metadata` | `{model_version, timestamp}` | run info |

**`ScaleReport`** — `metres_per_pixel_at_base` (the *local* scale at the foot of
the facade; under perspective it shrinks toward the roof), `sigma_rel`,
`perspective_corrected`, `vertical_vanishing_point_y`, `storey_pitch_m`,
`storey_pitch_residual`, `facade_span_px`, `rows_detected`, and `anchors[]` —
every cue that was formed, each with its `reference_m`, its own implied scale,
its `sigma_rel`, and whether it was `used` or discarded as an outlier.

```jsonc
{
  "image": { "id": "Fac5", "width": 413, "height": 688 },
  "view_type": "facade",
  "predictions": {
    "building_height_m": { "value": 30.4, "confidence": 0.763 },
    "floor_count":       { "value": 8,    "confidence": 0.7611 }
  },
  "height_source": "fused",
  "scale": {
    "metres_per_pixel_at_base": 0.036033,
    "sigma_rel": 0.0935,
    "perspective_corrected": true,
    "vertical_vanishing_point_y": -1357.6,
    "storey_pitch_m": 2.98,
    "storey_pitch_residual": 0.0079,
    "facade_span_px": 590.0,
    "rows_detected": 7,
    "anchors": [
      // Rejected: this "door" was a shopfront element, not a 2.05 m doorway.
      // `snapped_px`: pixels the door's foot was moved down to reach the base row (0 = as detected).
      { "source": "door",        "reference_m": 2.05, "metres_per_pixel_at_base": 0.021735, "sigma_rel": 0.08,   "used": false, "detail": { "span_px": 90.0, "snapped_px": 0.0 } },
      { "source": "floor_pitch", "reference_m": 3.0,  "metres_per_pixel_at_base": 0.036247, "sigma_rel": 0.1008, "used": true,  "detail": { "rows": 7.0 } },
      { "source": "window",      "reference_m": 1.45, "metres_per_pixel_at_base": 0.034741, "sigma_rel": 0.25,   "used": true,  "detail": { "count": 38.0 } }
    ]
  },
  "metadata": { "model_version": "v2.0", "timestamp": "2026-09-16T08:54:18.894149Z" }
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
| `scene_parsing` | `enabled`, Cityscapes `model_id` (b0–b5), `input_long_side`, `min_kept_ratio`, component/edge cleanup (`morph_kernel_px`, `min_component_ratio`, `edge_percentile`), base bracket probe (`base_probe_ratio`) |
| `building_features` | reference lengths (`assumed_door_height_m`, `assumed_floor_pitch_m`, `assumed_window_height_m`) and their `sigma_rel_*` weights; row clustering (`window_cluster_tol_ratio`, `min_row_members_ratio`, `floor_line_labels`); the vertical model (`max_vertical_scale_ratio`, `max_storeys_above_top_row`, `max_storeys_below_bottom_row`, `min_rows_for_perspective`, `perspective_improvement`); the truncated-door correction (`door_short_storeys`, `door_max_storeys`, `door_base_snap_ratio`); sanity bounds (`plausible_floor_height_m`, `scale_agreement_log_tol`) and the `assumed_floor_height_m` last resort |
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
│   ├── scene_parsing/                # Module 1b: Cityscapes gate on the facade mask
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
