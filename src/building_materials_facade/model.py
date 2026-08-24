"""DINOv3 facade-material model: a self-supervised ViT backbone + MLP head.

Default mode is a *frozen probe* — the DINOv3 ViT-B/16 backbone (strong, clean
dense/texture features) is frozen and only a small MLP head is trained. This is
robust to the weak/noisy labels of the pooled facade datasets and trains in
minutes on an 8 GB GPU. Passing ``unfreeze_blocks > 0`` fine-tunes the last few
transformer blocks for extra accuracy when the probe plateaus.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

DEFAULT_ARCH = "vit_base_patch16_dinov3.lvd1689m"


class FacadeMaterialModel(nn.Module):
    """Frozen (or partially fine-tuned) backbone + 2-layer MLP classifier head."""

    def __init__(
        self,
        num_classes: int,
        arch: str = DEFAULT_ARCH,
        pretrained: bool = True,
        unfreeze_blocks: int = 0,
        hidden: int = 512,
        dropout: float = 0.3,
    ) -> None:
        """Build the model.

        Args:
            num_classes: number of Facade-8 classes.
            arch: timm backbone name (DINOv3 ViT-B/16 by default).
            pretrained: load pretrained backbone weights.
            unfreeze_blocks: number of final transformer blocks to leave
                trainable. 0 = fully frozen backbone (linear/MLP probe).
            hidden: hidden width of the MLP head.
            dropout: dropout before the final classification layer.
        """
        super().__init__()
        self.arch = arch
        self.unfreeze_blocks = unfreeze_blocks
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features

        # Freeze the backbone, then optionally re-enable the last N blocks.
        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_blocks > 0 and hasattr(self.backbone, "blocks"):
            for blk in self.backbone.blocks[-unfreeze_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
        self._backbone_frozen = unfreeze_blocks == 0

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._backbone_frozen:
            with torch.no_grad():
                feats = self.backbone(x)
        else:
            feats = self.backbone(x)
        return self.head(feats)

    def trainable_parameters(self):
        """Yield only parameters with grad enabled (head + any unfrozen blocks)."""
        return (p for p in self.parameters() if p.requires_grad)


def build_classifier(
    num_classes: int,
    arch: str = DEFAULT_ARCH,
    pretrained: bool = True,
    unfreeze_blocks: int = 0,
    hidden: int = 512,
    dropout: float = 0.3,
) -> FacadeMaterialModel:
    """Construct a `FacadeMaterialModel` (mirrors the MINC backend's factory)."""
    return FacadeMaterialModel(
        num_classes, arch=arch, pretrained=pretrained,
        unfreeze_blocks=unfreeze_blocks, hidden=hidden, dropout=dropout,
    )


def build_transforms(model: FacadeMaterialModel, train: bool):
    """timm train/eval transforms matching the backbone's expected preprocessing."""
    cfg = timm.data.resolve_model_data_config(model.backbone)
    return timm.data.create_transform(**cfg, is_training=train)
