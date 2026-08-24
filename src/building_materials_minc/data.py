"""MINC-2500 → facade-material training dataset.

Wraps a HuggingFace MINC-2500 split, keeps only the facade-relevant classes
(remapping MINC-23 indices to our 8 target classes), and applies a timm
transform. Dropped/merged per :mod:`.labels`.
"""
from __future__ import annotations

from torch.utils.data import Dataset

from .labels import minc_index_to_target_idx


class MincFacadeDataset(Dataset):
    """A HF MINC-2500 split filtered+remapped to facade target classes."""

    def __init__(self, hf_split, transform) -> None:
        self.ds = hf_split
        self.transform = transform
        # Precompute (row_index, target_idx) for the kept facade classes.
        labels = hf_split["label"]  # whole column (fast)
        self.items: list[tuple[int, int]] = []
        for row, minc_idx in enumerate(labels):
            target = minc_index_to_target_idx(int(minc_idx))
            if target is not None:
                self.items.append((row, target))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, k: int):
        row, target = self.items[k]
        image = self.ds[row]["image"].convert("RGB")
        return self.transform(image), target
