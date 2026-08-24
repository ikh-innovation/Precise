# `segformer_cmp` — SegFormer reproduction of the published CMP result

A faithful reproduction of the **best documented segmentation method for the
CMP Facade Database**: fine-tuning **SegFormer (MiT-B0)** on the **full 12
classes**, following the official HuggingFace recipe
([blog](https://huggingface.co/blog/fine-tune-segformer), which uses the CMP
facade dataset as its worked example; published checkpoints exist e.g.
[Xpitfire/segformer-finetuned-segments-cmp-facade](https://huggingface.co/Xpitfire/segformer-finetuned-segments-cmp-facade)).

This is a **standalone benchmark reproduction** — separate from the 4-class
pipeline module (`facade_parsing_segm`), because it uses the native 12-class
CMP scheme, not the pipeline's background/facade/window/door.

## Recipe (matched exactly)

| | |
|---|---|
| Architecture | SegFormer, `nvidia/mit-b0` backbone + MLP decode head |
| Classes | 12 (full CMP) |
| Optimizer | AdamW (Trainer default) |
| Learning rate | `6e-5` |
| Epochs | `50` |
| Batch size | `2` |
| Image size | 512×512 (SegformerImageProcessor) |
| Selection metric | mean IoU (best model kept) |

Classes (0-indexed): `background, facade, window, door, cornice, sill, balcony,
blind, deco, molding, pillar, shop`. CMP raw mask ids `1..12` are remapped to
`0..11` (`raw-1`). Standard per-class colors are in
[`labels.py`](labels.py) (facade=red, window=blue, door=green reuse the
pipeline palette; the rest get distinct colors).

## Dependencies

`transformers>=4.30` and `accelerate>=0.20` (both already in the env). The
`nvidia/mit-b0` backbone is pulled from the HuggingFace Hub on first run —
needs internet that first time. `evaluate`/`datasets` are **not** required:
the dataset is a plain `torch` `Dataset` and mean-IoU is computed locally.

## Train

```bash
# Full reproduction (GPU strongly recommended)
python src/facade_parsing_segm/segformer_cmp/train.py --data-root base

# quick functional check
python src/facade_parsing_segm/segformer_cmp/train.py --limit 8 --epochs 1
```

Writes HuggingFace checkpoints under `runs/` and the best model (by mean IoU) to
`runs/best/` (model + image processor).

## Infer

```bash
python src/facade_parsing_segm/segformer_cmp/inference.py \
  --image base/cmp_b0001.jpg \
  --model-dir src/facade_parsing_segm/segformer_cmp/runs/best
```

Writes `<stem>_seg_mask.png` (12-color label map) and `<stem>_seg_overlay.png`
(colors over the photo, background untouched).

## Honest note on "best result"

There is **no single authoritative mIoU leaderboard** for CMP — reported
facade-parsing numbers vary by class set, split, and post-processing. SegFormer
is the modern, SOTA-class method with a public, dataset-specific recipe, which
is why it's the reproduction target here. The 12-class problem is hard on a
606-image dataset, so expect a modest mIoU (facade/window dominate; thin
classes like sill/molding are much harder).

## Files

| File | Purpose |
|------|---------|
| [`labels.py`](labels.py) | 12-class scheme, raw→0-indexed LUT, standard colors |
| [`data.py`](data.py) | pair discovery/split + `CmpSegformerDataset` (HF dict) |
| [`train.py`](train.py) | HuggingFace `Trainer` recipe (mit-b0, lr 6e-5, 50 ep, bs 2) |
| [`inference.py`](inference.py) | `SegformerCmpSegmenter` + CLI (mask + overlay) |
