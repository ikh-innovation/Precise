"""Dataset for SegFormer fine-tuning on the 12-class CMP Facade Database.

Each item is the HuggingFace dict the model expects — ``pixel_values`` and
``labels`` — produced by a ``SegformerImageProcessor`` (resize to 512, ImageNet
normalization; masks resized nearest). Kept self-contained (its own pair
discovery) so this reproduction does not depend on the 4-class module.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from .labels import build_remap_lut

Pair = tuple[Path, Path]


def discover_pairs(root: str | Path) -> list[Pair]:
    """Find ``(image.jpg, mask.png)`` pairs under ``root`` (skips pipeline PNGs)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    pairs: list[Pair] = []
    for jpg in sorted(root.glob("*.jpg")):
        if jpg.stem.endswith("_pipeline"):
            continue
        mask = jpg.with_suffix(".png")
        if mask.exists():
            pairs.append((jpg, mask))
    return pairs


def split_pairs(
    pairs: list[Pair], val_split: float = 0.2, seed: int = 42
) -> tuple[list[Pair], list[Pair]]:
    """Deterministically shuffle and split into ``(train, val)``."""
    items = list(pairs)
    random.Random(seed).shuffle(items)
    n_val = int(round(len(items) * val_split))
    if len(items) > 1:
        n_val = min(max(1, n_val), len(items) - 1)
    else:
        n_val = 0
    return items[n_val:], items[:n_val]


class CmpSegformerDataset(Dataset):
    """Yields ``{"pixel_values": [3,H,W], "labels": [H,W]}`` for the HF Trainer.

    Args:
        pairs: ``(jpg, png)`` paths.
        processor: A ``transformers.SegformerImageProcessor`` (handles resize +
            normalization for both image and mask).
        lut: Optional raw->0-indexed LUT (built if omitted).
    """

    def __init__(
        self, pairs: list[Pair], processor, lut: np.ndarray | None = None, train: bool = False
    ) -> None:
        self.pairs = list(pairs)
        self.processor = processor
        self.lut = build_remap_lut() if lut is None else lut
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def _augment(self, image: Image.Image, label: np.ndarray) -> tuple[Image.Image, np.ndarray]:
        """Train-time aug: scale-jitter crop + h-flip + brightness/contrast jitter.

        The scale-jitter crop keeps a random sub-region; the processor then
        resizes it back to 512, so the model sees facades at varied zoom levels
        — a high-leverage augmentation on the small (606-image) CMP set. Image
        and label are cropped with identical coordinates.
        """
        W, H = image.size
        if random.random() < 0.7 and W > 64 and H > 64:
            scale = random.uniform(0.55, 1.0)        # fraction of area kept
            ratio = random.uniform(0.8, 1.25)        # aspect jitter
            cw = int(round((W * H * scale * ratio) ** 0.5))
            ch = int(round((W * H * scale / ratio) ** 0.5))
            cw = max(64, min(cw, W))
            ch = max(64, min(ch, H))
            x0 = random.randint(0, W - cw)
            y0 = random.randint(0, H - ch)
            image = image.crop((x0, y0, x0 + cw, y0 + ch))
            label = np.ascontiguousarray(label[y0:y0 + ch, x0:x0 + cw])
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = np.ascontiguousarray(label[:, ::-1])
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        return image, label

    def __getitem__(self, idx: int) -> dict:
        jpg, mask_path = self.pairs[idx]
        image = Image.open(jpg).convert("RGB")
        raw = np.array(Image.open(mask_path), dtype=np.uint8)
        if raw.ndim != 2:
            raise ValueError(f"Expected palettized mask at {mask_path}, got shape {raw.shape}")
        label = self.lut[raw]  # 0-indexed 0..11
        if self.train:
            image, label = self._augment(image, label)
        enc = self.processor(
            images=image, segmentation_maps=Image.fromarray(label), return_tensors="pt"
        )
        return {
            "pixel_values": enc["pixel_values"][0],
            "labels": enc["labels"][0].long(),
        }
