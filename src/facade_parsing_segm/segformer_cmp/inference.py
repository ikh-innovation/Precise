"""Inference for the 12-class CMP SegFormer model.

Loads a directory produced by ``train.py`` (model + image processor), segments
an image into the 12 CMP classes, and writes a colored mask + overlay. This is
a standalone benchmark reproduction (12 classes); it is not wired into the
4-class pipeline.

    python src/facade_parsing_segm/segformer_cmp/inference.py \
        --image base/cmp_b0001.jpg \
        --model-dir src/facade_parsing_segm/segformer_cmp/runs/best
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from facade_parsing_segm.segformer_cmp.labels import CLASS_NAMES, colorize_mask, colors_bgr_array


class SegformerCmpSegmenter:
    """Run a trained 12-class CMP SegFormer over an image."""

    def __init__(self, model_dir: str | Path, device: str | None = None) -> None:
        model_dir = str(model_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = SegformerImageProcessor.from_pretrained(model_dir)
        self.model = (
            SegformerForSemanticSegmentation.from_pretrained(model_dir).to(self.device).eval()
        )

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(label_map [H,W] 0..11, image_bgr [H,W,3])`` at original size."""
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits  # [1, 12, H/4, W/4]
        upsampled = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        label_map = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return label_map, image_bgr

    def overlay(self, image_bgr: np.ndarray, label_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """Blend standard per-class colors over the image (background untouched)."""
        colors = colors_bgr_array()
        out = image_bgr.astype(np.float32)
        for cls_id in range(1, len(CLASS_NAMES)):  # skip background
            mask = label_map == cls_id
            if mask.any():
                out[mask] = out[mask] * (1 - alpha) + colors[cls_id].astype(np.float32) * alpha
        return out.clip(0, 255).astype(np.uint8)


def _iter_images(image: str | None, input_dir: str | None) -> list[Path]:
    if image:
        return [Path(image)]
    if input_dir:
        return [p for p in sorted(Path(input_dir).glob("*.jpg")) if not p.stem.endswith("_pipeline")]
    raise SystemExit("Pass --image or --input-dir.")


def main() -> None:
    p = argparse.ArgumentParser(description="Segment images with the 12-class CMP SegFormer.")
    p.add_argument("--image", default=None)
    p.add_argument("--input-dir", default=None)
    p.add_argument("--model-dir", default="src/facade_parsing_segm/segformer_cmp/runs/best")
    p.add_argument("--out-dir", default="out_segformer")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    seg = SegformerCmpSegmenter(args.model_dir, device=args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for img_path in _iter_images(args.image, args.input_dir):
        label_map, image_bgr = seg.predict(img_path)
        stem = img_path.stem
        cv2.imwrite(str(out / f"{stem}_seg_mask.png"), colorize_mask(label_map))
        cv2.imwrite(str(out / f"{stem}_seg_overlay.png"), seg.overlay(image_bgr, label_map))
        present = sorted({CLASS_NAMES[i] for i in np.unique(label_map)})
        print(f"{stem}: classes={present} -> {out}/{stem}_seg_overlay.png", flush=True)


if __name__ == "__main__":
    main()
