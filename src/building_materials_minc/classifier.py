"""Inference: MINC-trained material classifier with patch-voting.

`MincMaterials` mirrors `building_materials.BuildingMaterials` — same
`classify(region_mask=...) -> MaterialClassificationResult` surface — so
`client.py` can swap backends. It samples native-resolution wall patches from
the facade region, classifies each with the fine-tuned timm model, and averages
the per-material probabilities. Reuses the project's material-result schema and
the fixed mechanical-properties lookup.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image

from config import BuildingMaterialsConfig, MaterialProperties
from building_materials.region import sample_region_patches
from building_materials.schemas import (
    ImageInfo,
    MaterialClassificationResult,
    MaterialScore,
    Metadata,
    ObjectMaterial,
    PerObjectMaterialResult,
    ViewType,
)
from .model import build_classifier

Bbox = tuple[float, float, float, float]


class MincMaterials:
    """Material classifier backed by the fine-tuned MINC timm model."""

    def __init__(
        self,
        image_path: str | Path,
        cfg: BuildingMaterialsConfig,
        view_type: ViewType = "facade",
    ) -> None:
        self.image_path = Path(image_path)
        self.cfg = cfg
        self.view_type: ViewType = view_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path = Path(cfg.minc_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"MINC material checkpoint not found: {ckpt_path}. Train it with "
                f"`python src/building_materials_minc/train.py`."
            )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.class_names: list[str] = ckpt["class_names"]
        self.arch: str = ckpt.get("arch", cfg.minc_arch)
        model = build_classifier(len(self.class_names), arch=self.arch, pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        self.model = model.to(self.device).eval()
        # Eval transform that matches how the model was trained.
        self.transform = timm.data.create_transform(**ckpt["data_config"], is_training=False)

    def _load_rgb(self) -> tuple[np.ndarray, int, int]:
        bgr = cv2.imread(str(self.image_path))
        if bgr is None:
            raise FileNotFoundError(f"Image not found or unreadable: {self.image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        return rgb, w, h

    @torch.no_grad()
    def _image_probs(self, patches: list[np.ndarray]) -> np.ndarray:
        batch = torch.stack([self.transform(Image.fromarray(p)) for p in patches]).to(self.device)
        probs = self.model(batch).softmax(dim=-1)
        return probs.cpu().numpy()

    def _rank(self, prob_vec: np.ndarray) -> list[MaterialScore]:
        total = float(prob_vec.sum())
        norm = prob_vec / total if total > 0 else prob_vec
        ranked = sorted(zip(self.class_names, norm.tolist()), key=lambda kv: kv[1], reverse=True)
        return [MaterialScore(label=l, score=round(float(s), 4)) for l, s in ranked]

    def _properties_for(self, label: str) -> MaterialProperties | None:
        return self.cfg.material_properties.get(label)

    def _no_region_result(self, width: int, height: int) -> MaterialClassificationResult:
        """Fail-closed result when Module 1 found no building region to read."""
        return MaterialClassificationResult(
            image=ImageInfo(id=self.image_path.stem, width=width, height=height),
            view_type=self.view_type,
            materials=[],
            dominant_material=MaterialScore(label="none", score=0.0),
            dominant_material_properties=None,
            classified_region="none",
            metadata=Metadata(
                candidate_materials=list(self.class_names),
                model=f"minc-{self.arch}",
            ),
        )

    def classify(
        self, region_mask: np.ndarray | None = None
    ) -> MaterialClassificationResult:
        """Classify the building material by voting over wall patches.

        Patches are sampled STRICTLY inside `region_mask` (facade/house minus
        dilated window/door openings) with non-region pixels neutralised; an
        empty region yields a fail-closed `classified_region="none"` result
        instead of classifying the whole image.
        """
        rgb, width, height = self._load_rgb()
        patches, source = sample_region_patches(
            rgb, region_mask,
            patch_size=self.cfg.minc_patch_size,
            max_patches=self.cfg.minc_max_patches,
            min_wall_ratio=self.cfg.minc_min_wall_ratio,
        )
        if not patches:
            return self._no_region_result(width, height)
        avg = self._image_probs(patches).mean(axis=0)
        scored = self._rank(avg)
        return MaterialClassificationResult(
            image=ImageInfo(id=self.image_path.stem, width=width, height=height),
            view_type=self.view_type,
            materials=scored,
            dominant_material=scored[0],
            dominant_material_properties=self._properties_for(scored[0].label),
            classified_region=source,
            metadata=Metadata(
                candidate_materials=list(self.class_names),
                model=f"minc-{self.arch}",
            ),
        )

    def classify_instances(
        self, detections: dict[str, list[Bbox]]
    ) -> PerObjectMaterialResult:
        """Classify each detection's crop independently (per-object mode)."""
        rgb, width, height = self._load_rgb()
        min_size = self.cfg.per_object_min_size_px
        objects: dict[str, list[ObjectMaterial]] = {}
        for label, bboxes in detections.items():
            entries: list[ObjectMaterial] = []
            for bbox in bboxes:
                x1, y1, x2, y2 = (int(round(v)) for v in bbox)
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height))
                if x2 - x1 < min_size or y2 - y1 < min_size:
                    continue
                scored = self._rank(self._image_probs([rgb[y1:y2, x1:x2]])[0])
                entries.append(ObjectMaterial(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    dominant_material=scored[0],
                    properties=self._properties_for(scored[0].label),
                ))
            objects[label] = entries
        return PerObjectMaterialResult(
            image=ImageInfo(id=self.image_path.stem, width=width, height=height),
            view_type=self.view_type,
            objects=objects,
            metadata=Metadata(
                candidate_materials=list(self.class_names),
                model=f"minc-{self.arch}",
            ),
        )
