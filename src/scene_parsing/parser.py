"""Cityscapes scene parsing — what in the frame is *not* the building.

Wraps a Cityscapes-finetuned SegFormer from the Hub (weights download on first
run, then cache). The model is used only as a scene-level gate: the per-pixel
argmax is reduced to the four groups in `labels.py` and handed to `refine.py`.
Nothing here decides anything about facades — Module 1 still owns window, door,
sill and balcony, which is what it is actually trained for.

Runs on GPU when one is free, else CPU (~1-2 s for one image at b4).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .labels import CLASS_NAMES, GROUPS, ids_for


@dataclass(frozen=True)
class SceneMasks:
    """Per-group binary masks over one image, plus the raw label map.

    Attributes:
        labels: `[H, W]` uint8 Cityscapes train ids.
        groups: Group name -> `[H, W]` uint8 mask (1 = member of the group).
        height: Image height in pixels.
        width: Image width in pixels.
    """

    labels: np.ndarray
    groups: dict[str, np.ndarray]
    height: int
    width: int

    def group(self, name: str) -> np.ndarray:
        """Mask for one group, or an all-zero mask when the name is unknown."""
        if name in self.groups:
            return self.groups[name]
        return np.zeros((self.height, self.width), dtype=np.uint8)

    def coverage(self) -> dict[str, float]:
        """Fraction of the frame each group occupies, for logging."""
        total = float(self.height * self.width) or 1.0
        return {k: float(v.sum()) / total for k, v in self.groups.items()}


class SceneParser:
    """Cityscapes semantic segmentation, reduced to building/occluder/ground/sky."""

    def __init__(self, cfg) -> None:
        """Load the Cityscapes checkpoint named by `cfg.model_id`.

        Args:
            cfg: `SceneParsingConfig` (model id, input size, device preference).
        """
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.cfg = cfg
        use_cuda = torch.cuda.is_available() and cfg.device != "cpu"
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.processor = SegformerImageProcessor.from_pretrained(cfg.model_id)
        self.model = (
            SegformerForSemanticSegmentation.from_pretrained(cfg.model_id)
            .to(self.device)
            .eval()
        )
        # Trust the checkpoint's own label order rather than assuming it
        # matches `labels.CLASS_NAMES`; the group ids are derived from it.
        by_id = {int(i): str(n).lower() for i, n in self.model.config.id2label.items()}
        names = [by_id.get(i, CLASS_NAMES[i] if i < len(CLASS_NAMES) else "")
                 for i in range(self.model.config.num_labels)]
        self.class_names = names
        self.group_ids = {
            group: [i for i, n in enumerate(names) if n in members]
            for group, members in GROUPS.items()
        }

    @torch.no_grad()
    def parse(self, image: str | Path | np.ndarray) -> SceneMasks:
        """Segment one image and reduce it to group masks.

        Args:
            image: Path to an image, or an `[H, W, 3]` **BGR** uint8 array
                (the pipeline's in-memory convention).

        Returns:
            `SceneMasks` at the input image's original resolution.
        """
        rgb = self._as_rgb(image)
        h, w = rgb.shape[:2]

        # Cityscapes checkpoints are trained around 1024 px; a 413 px facade
        # crop segments noticeably better upsampled to that scale first.
        scaled = self._resize_long_side(rgb, self.cfg.input_long_side)
        inputs = self.processor(images=Image.fromarray(scaled), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        upsampled = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        pred = upsampled.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()

        groups = {
            name: np.isin(pred, ids).astype(np.uint8)
            for name, ids in self.group_ids.items()
        }
        return SceneMasks(labels=pred, groups=groups, height=h, width=w)

    @staticmethod
    def _as_rgb(image: str | Path | np.ndarray) -> np.ndarray:
        """Normalise the input to an `[H, W, 3]` RGB uint8 array."""
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected an [H, W, 3] BGR array, got {image.shape}")
            return np.ascontiguousarray(image[:, :, ::-1])
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return np.asarray(Image.open(path).convert("RGB"))

    @staticmethod
    def _resize_long_side(rgb: np.ndarray, long_side: int) -> np.ndarray:
        """Scale so the longer side is `long_side`, preserving aspect ratio."""
        h, w = rgb.shape[:2]
        longest = max(h, w)
        if long_side <= 0 or longest == long_side:
            return rgb
        scale = long_side / float(longest)
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        return np.asarray(Image.fromarray(rgb).resize(size, Image.BILINEAR))


def ids_for_group(group: str) -> list[int]:
    """Canonical Cityscapes train ids for one group name."""
    return ids_for(GROUPS.get(group, frozenset()))
