"""Dataset + transforms for the pooled Facade-8 training corpus.

Samples come from a unified manifest (see build_dataset.py) with a `kind`:
  * "patch"  — MINC material patches: already material-centric, so only a mild
               random-resized crop is applied.
  * "facade" — whole-building crops (OpenFACADES / URC / London-Scotland) whose
               image-level label is the dominant material. A wider
               random-resized crop turns each into a weakly-labeled material
               patch and is the main augmentation that makes these usable.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


def _normalize(mean, std) -> T.Normalize:
    return T.Normalize(mean=mean, std=std)


def build_transform(mean, std, input_size: int, train: bool, kind: str = "facade"):
    """Source-aware train/eval transform.

    Facade crops use an aggressive scale range to sample sub-regions of a
    building; MINC patches use a gentle range since they are already close-ups.
    """
    if not train:
        return T.Compose([
            T.Resize(input_size, antialias=True),
            T.CenterCrop(input_size),
            T.ToTensor(),
            _normalize(mean, std),
        ])
    scale = (0.5, 1.0) if kind == "patch" else (0.2, 0.8)
    return T.Compose([
        T.RandomResizedCrop(input_size, scale=scale, ratio=(0.75, 1.333), antialias=True),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        T.ToTensor(),
        _normalize(mean, std),
    ])


class ManifestDataset(Dataset):
    """Loads (image, target_idx) rows, applying a kind/split-aware transform."""

    def __init__(self, rows: list[dict], mean, std, input_size: int, train: bool) -> None:
        """Args:
            rows: dicts with keys `path`, `target_idx`, `kind`.
            mean, std: backbone normalization stats.
            input_size: square model input side (px).
            train: apply training augmentation when True.
        """
        self.rows = rows
        self.train = train
        # One transform per kind so MINC patches and facade crops are handled
        # differently without rebuilding the Compose every __getitem__.
        self._tf = {
            k: build_transform(mean, std, input_size, train, k)
            for k in ("patch", "facade")
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        with Image.open(row["path"]) as im:
            img = im.convert("RGB")
            x = self._tf.get(row["kind"], self._tf["facade"])(img)
        return x, int(row["target_idx"])
