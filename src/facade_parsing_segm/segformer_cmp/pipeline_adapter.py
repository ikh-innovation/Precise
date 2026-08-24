"""Use the trained 12-class CMP SegFormer as Module 1 of the Precise pipeline.

`FacadeSegformerParser` mirrors SEEM's `FacadeParser` surface
(`parse() -> FacadeParsingResult`, `last_visualization_image`), so `client.py`
can swap it in. The model predicts all 12 CMP classes and the pipeline now
**shows every one of them** in its own color (facade=red, window=blue,
door=green, plus cornice/sill/balcony/blind/deco/molding/pillar/shop) — only
the catch-all `background` is left as the untouched photo. Module 2 still uses
`window`/`door` for floors and treats `facade` as the building extent (see
`client.py:_house_bbox`); Module 3 reads the wall from `facade` minus the
opening classes.

Runs on GPU if available, else CPU (a single image is ~0.3 s on CPU).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from facade_parsing.schemas import (
    BBox,
    ClassMask,
    FacadeParsingResult,
    ImageInfo,
    Metadata,
    Polygon,
    ViewType,
)

from .labels import CLASS_NAMES, COLORS_BGR

Bbox = tuple[float, float, float, float]

# Show every CMP facade-element class, each in its own color and as its own
# ClassMask — no merging (blind/shop/etc. are distinct). Only `background`
# (the catch-all for everything not annotated) is hidden, so its pixels stay as
# the raw photo. `pred = argmax` makes the per-pixel labels disjoint, so the
# emitted masks never overlap. facade/window/door keep their exact labels so
# Modules 2 and 3 are unaffected.
HIDDEN_CLASSES: set[str] = {"background"}
DISPLAY_CLASSES: list[str] = [c for c in CLASS_NAMES if c not in HIDDEN_CLASSES]
DISPLAY_MERGE: dict[str, str] = {c: c for c in DISPLAY_CLASSES}
DISPLAY_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    c: COLORS_BGR[c] for c in DISPLAY_CLASSES
}


class FacadeSegformerParser:
    """SEEM-compatible Module 1 backed by the trained CMP SegFormer (all 12 classes)."""

    def __init__(self, image_path, cfg, colors_bgr=None, view_type: ViewType = "facade") -> None:
        """Load the trained SegFormer checkpoint.

        Args:
            image_path: Image to segment on `parse()`.
            cfg: `FacadeParsingSegmFormerConfig` (checkpoint dir + overlay params).
            colors_bgr: Optional `label -> [B,G,R]` overrides on top of the
                standard 12-class palette (e.g. to match SEEM's window/door).
            view_type: Propagated into the result schema.
        """
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.image_path = Path(image_path)
        self.cfg = cfg
        self.view_type: ViewType = view_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = Path(cfg.checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"SegFormer checkpoint not found: {ckpt}. Train it first:\n"
                f"  python src/facade_parsing_segm/segformer_cmp/train.py --data-root base"
            )
        self.processor = SegformerImageProcessor.from_pretrained(str(ckpt))
        self.model = (
            SegformerForSemanticSegmentation.from_pretrained(str(ckpt)).to(self.device).eval()
        )
        # Map each displayed class to the CMP class ids that feed it
        # (e.g. window <- {window, blind}), from the checkpoint's own config.
        name2id = {n.lower(): int(i) for i, n in self.model.config.id2label.items()}
        self.display_to_ids: dict[str, list[int]] = {}
        for src, disp in DISPLAY_MERGE.items():
            if src in name2id:
                self.display_to_ids.setdefault(disp, []).append(name2id[src])

        # Display colors (red/blue/green), with optional caller overrides.
        self.colors = dict(DISPLAY_COLORS_BGR)
        if colors_bgr:
            self.colors.update({k: tuple(int(c) for c in v) for k, v in colors_bgr.items()})

        self.last_visualization_image: np.ndarray | None = None
        self.last_class_pixel_counts: dict[str, int] = {}

    @torch.no_grad()
    def _predict_probs(self, image_rgb: np.ndarray) -> np.ndarray:
        """Per-class probabilities `[12, H, W]` at the original resolution."""
        h, w = image_rgb.shape[:2]
        inputs = self.processor(images=Image.fromarray(image_rgb), return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        upsampled = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return upsampled.softmax(dim=1)[0].cpu().numpy()

    # ----- geometry helpers (mirror SEEM parser for output parity) -----------
    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> Bbox | None:
        ys, xs = mask.nonzero()
        if xs.size == 0:
            return None
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    @staticmethod
    def _norm_bbox(box: Bbox, w: int, h: int) -> list[float]:
        x1, y1, x2, y2 = box
        return [x1 / w, y1 / h, x2 / w, y2 / h]

    def _extract_polygons(self, mask: np.ndarray) -> list[list[list[float]]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys: list[list[list[float]]] = []
        for c in contours:
            if cv2.contourArea(c) < self.cfg.min_polygon_area_px:
                continue
            eps = self.cfg.poly_simplify_eps_ratio * cv2.arcLength(c, True)
            polys.append(cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float).tolist())
        return polys

    @staticmethod
    def _norm_poly(pts: list[list[float]], w: int, h: int) -> list[list[float]]:
        return [[x / w, y / h] for x, y in pts]

    def _draw_overlay(self, image_bgr: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
        """Composite every predicted class over the image (background untouched).

        Larger regions are painted first so smaller ones stay visible on top.
        """
        alpha = self.cfg.overlay_alpha
        composite = image_bgr.astype(np.float32)
        order = sorted(masks, key=lambda n: -int(masks[n].sum()))
        for name in order:
            mask = masks[name]
            if mask.sum() == 0:
                continue
            color = np.array(self.colors.get(name, (200, 200, 200)), dtype=np.float32)
            b = mask.astype(bool)
            composite[b] = composite[b] * (1 - alpha) + color * alpha
        for name in order:
            mask = masks[name]
            if mask.sum() == 0:
                continue
            color = tuple(int(c) for c in self.colors.get(name, (200, 200, 200)))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(composite, contours, -1, color, 2)
        out = composite.clip(0, 255).astype(np.uint8)
        present = [n for n in DISPLAY_CLASSES if n in masks and int(masks[n].sum()) > 0]
        self._draw_legend(out, present)
        return out

    def _draw_legend(self, img: np.ndarray, names: list[str]) -> None:
        """Draw a color-swatch legend (present classes) at the top-right, in place."""
        if not names:
            return
        h, w = img.shape[:2]
        fs = max(0.45, min(w, h) / 1600.0)      # font scale ~ image size
        th = 1 if fs < 0.9 else 2               # text thickness
        row = int(28 * fs) + 8                  # row height
        sw = int(20 * fs) + 6                   # swatch side
        pad = int(10 * fs) + 4
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_w = max(cv2.getTextSize(n, font, fs, th)[0][0] for n in names)
        box_w = pad + sw + 8 + text_w + pad
        box_h = pad + row * len(names)
        x0 = max(0, w - box_w - pad)
        y0 = pad
        cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        y = y0 + pad
        for name in names:
            color = tuple(int(c) for c in self.colors.get(name, (200, 200, 200)))
            cv2.rectangle(img, (x0 + pad, y), (x0 + pad + sw, y + sw), color, -1)
            cv2.rectangle(img, (x0 + pad, y), (x0 + pad + sw, y + sw), (255, 255, 255), 1)
            cv2.putText(img, name, (x0 + pad + sw + 8, y + sw - int(4 * fs)),
                        font, fs, (255, 255, 255), th, cv2.LINE_AA)
            y += row

    # ----- public API --------------------------------------------------------
    def parse(self) -> FacadeParsingResult:
        """Segment the image into all CMP classes; return a SEEM-compatible result."""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        image_rgb = np.asarray(Image.open(self.image_path).convert("RGB"))
        h, w = image_rgb.shape[:2]
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        probs = self._predict_probs(image_rgb)
        pred = probs.argmax(0).astype(np.uint8)

        masks: dict[str, np.ndarray] = {}
        classes: list[ClassMask] = []
        for name in DISPLAY_CLASSES:
            ids = self.display_to_ids.get(name, [])
            if not ids:
                continue
            mask = np.isin(pred, ids).astype(np.uint8)  # union (window = window+blind)
            masks[name] = mask
            area = int(mask.sum())
            self.last_class_pixel_counts[name] = area
            if area == 0:
                classes.append(ClassMask(label=name, confidence=0.0, pixel_area=0,
                                         bbox=None, polygons=[], mask_path=None))
                continue
            confidence = float(probs[ids].sum(axis=0)[mask.astype(bool)].mean())
            box = self._bbox_from_mask(mask)
            bbox = None
            if box is not None:
                bbox = BBox(
                    pixel=[round(v, 2) for v in box],
                    normalized=[round(v, 6) for v in self._norm_bbox(box, w, h)],
                )
            polygons = [
                Polygon(
                    pixel=[[round(x, 2), round(y, 2)] for x, y in pts],
                    normalized=[[round(x, 6), round(y, 6)] for x, y in self._norm_poly(pts, w, h)],
                )
                for pts in self._extract_polygons(mask)
            ]
            classes.append(ClassMask(
                label=name,
                confidence=round(min(max(confidence, 0.0), 1.0), 4),
                pixel_area=area, bbox=bbox, polygons=polygons, mask_path=None,
            ))

        self.last_visualization_image = self._draw_overlay(image_bgr, masks)
        return FacadeParsingResult(
            image=ImageInfo(id=self.image_path.stem, width=w, height=h),
            view_type=self.view_type,
            classes=classes,
            metadata=Metadata(
                model_version=getattr(self.cfg, "model_version", "v1.0"),
                backbone="segformer-mit-b0",
                prompts=list(DISPLAY_CLASSES),
                threshold=self.cfg.prob_threshold,
                timestamp=datetime.now(timezone.utc),
            ),
        )
