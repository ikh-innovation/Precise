"""Material patch classifier — a timm ImageNet-pretrained backbone.

A small, strong image classifier (ConvNeXt-Tiny by default) fine-tuned to
predict the facade-material class of an image patch. timm supplies both the
model and the matching train/eval transforms.
"""
from __future__ import annotations

import timm
import torch.nn as nn


def build_classifier(num_classes: int, arch: str = "convnext_tiny", pretrained: bool = True) -> nn.Module:
    """Create a timm classifier with `num_classes` outputs."""
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)


def build_transforms(model: nn.Module, train: bool):
    """Return the timm preprocessing transform matching `model`'s data config.

    Train mode adds RandAugment / random-resized-crop; eval is resize+center-crop.
    """
    cfg = timm.data.resolve_model_data_config(model)
    return timm.data.create_transform(**cfg, is_training=train)
