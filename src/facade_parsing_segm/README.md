# `facade_parsing_segm` — trained Module 1 (facade parsing)

The supervised alternative to SEEM for Module 1. It hosts
[`segformer_cmp/`](segformer_cmp/) — a **SegFormer** fine-tuned on the **CMP
Facade Database** (`base/`, 12 classes) — and the adapter that plugs it into the
pipeline.

- **Train**: `segformer_cmp/train.py` (backbone `nvidia/mit-b0`…`mit-b5`; saves
  `segformer_cmp/runs/best`). See [`segformer_cmp/README.md`](segformer_cmp/README.md).
- **Use in the pipeline**: `client.main(..., manual_seg=True)` (the default).
  `segformer_cmp.pipeline_adapter.FacadeSegformerParser` mirrors SEEM's
  `FacadeParser` (`parse() → FacadeParsingResult`, `last_visualization_image`),
  reducing the 12 CMP classes to what the pipeline shows: **facade (red),
  window (blue, blinds merged in), door (green)**. `facade` serves as the
  building extent for Module 2.

Runs on GPU in the `precise-seem-gpu` env.

> An earlier 4-class U-Net/DeepLab variant lived here; it was removed in favor of
> the SegFormer-on-CMP approach.
