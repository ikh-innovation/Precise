"""Module 3 — CLIP-based building material classifier."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

from config import BuildingMaterialsConfig, MaterialProperties
from .region import sample_region_patches
from .schemas import (
    ImageInfo,
    MaterialClassificationResult,
    MaterialScore,
    Metadata,
    ObjectMaterial,
    PerObjectMaterialResult,
    ViewType,
)


Bbox = tuple[float, float, float, float]


class BuildingMaterials:
    """Zero-shot building material classifier driven by CLIP image/text similarity.

    Operates in two modes:
      * `classify(region_mask=...)` — single dominant material read STRICTLY
        from Module 1's building region (facade/house minus window/door
        openings). Non-region pixels are neutralised before encoding and there
        is no whole-image fallback: an empty region yields a `"none"` result.
      * `classify_instances(detections)` — one dominant material per bbox.

    There is no GrabCut: Module 1's semantic masks localize the building far
    better than a blind center-crop ever did.
    """

    def __init__(
        self,
        image_path: str | Path,
        cfg: BuildingMaterialsConfig,
        view_type: ViewType = "facade",
    ) -> None:
        """Load CLIP, precompute per-label text embeddings.

        Args:
            image_path: Path to the input image.
            cfg: Module 3 configuration section.
            view_type: Pipeline view type; selects the prompt template set.
        """
        self.image_path = Path(image_path)
        self.cfg = cfg
        self.view_type: ViewType = view_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            cfg.model_name, pretrained=cfg.pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(cfg.model_name)
        self.model.to(self.device).eval()
        # SigLIP scores with sigmoid(scale·sim + bias) instead of softmax;
        # detect it by the presence of a logit_bias parameter.
        self.is_siglip = getattr(self.model, "logit_bias", None) is not None
        self._text_features = self._build_text_features()

    def _build_text_features(self) -> torch.Tensor:
        """Mean-pool per-label CLIP text embeddings across the view's prompts.

        Returns:
            `[N_materials, D]` L2-normalized tensor on `self.device`.
        """
        templates = self.cfg.prompt_templates[self.view_type]
        per_label: list[torch.Tensor] = []
        with torch.no_grad():
            for label in self.cfg.materials:
                prompts = [t.format(label=label) for t in templates]
                tokens = self.tokenizer(prompts).to(self.device)
                emb = self.model.encode_text(tokens)
                emb /= emb.norm(dim=-1, keepdim=True)
                per_label.append(emb.mean(dim=0, keepdim=True))
        features = torch.cat(per_label, dim=0)
        features /= features.norm(dim=-1, keepdim=True)
        return features

    def _properties_for(self, label: str) -> MaterialProperties | None:
        """Look up the fixed mechanical properties configured for `label`.

        Returns `None` when the label has no `material_properties` entry, so
        classification still works for labels added to `materials` without a
        properties block.
        """
        return self.cfg.material_properties.get(label)

    def _load_rgb(self) -> tuple[np.ndarray, int, int]:
        """Read the image as RGB and return `(rgb, width, height)`."""
        bgr = cv2.imread(str(self.image_path))
        if bgr is None:
            raise FileNotFoundError(f"Image not found or unreadable: {self.image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        return rgb, w, h

    def _image_probs(self, images: list[np.ndarray]) -> np.ndarray:
        """Per-material probabilities for each image — `[N, n_materials]`.

        One batched forward pass; sigmoid scoring for SigLIP, softmax for CLIP.
        """
        batch = torch.stack(
            [self.preprocess(Image.fromarray(im)) for im in images]
        ).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            sim = feats @ self._text_features.T
            scale = self.model.logit_scale.exp()
            if self.is_siglip:
                probs = (sim * scale + self.model.logit_bias).sigmoid()
            else:
                probs = (sim * scale).softmax(dim=-1)
        return probs.cpu().numpy()

    def _rank(self, prob_vec: np.ndarray) -> list[MaterialScore]:
        """Sum-normalize a per-material probability vector into ranked scores."""
        total = float(prob_vec.sum())
        norm = prob_vec / total if total > 0 else prob_vec
        ranked = sorted(
            zip(self.cfg.materials, norm.tolist()), key=lambda kv: kv[1], reverse=True
        )
        return [MaterialScore(label=l, score=round(float(s), 4)) for l, s in ranked]

    def _classify_one(self, rgb: np.ndarray) -> list[MaterialScore]:
        """Rank materials for a single image crop (per-object mode)."""
        return self._rank(self._image_probs([rgb])[0])

    def _no_region_result(self, width: int, height: int) -> MaterialClassificationResult:
        """Fail-closed result when Module 1 found no building region to read."""
        return MaterialClassificationResult(
            image=ImageInfo(id=self.image_path.stem, width=width, height=height),
            view_type=self.view_type,
            materials=[],
            dominant_material=MaterialScore(label="none", score=0.0),
            dominant_material_properties=None,
            classified_region="none",
            metadata=Metadata(candidate_materials=self.cfg.materials, model=self.cfg.model_id),
        )

    def classify(
        self, region_mask: np.ndarray | None = None
    ) -> MaterialClassificationResult:
        """Classify the building material from Module 1's wall region only.

        Texture patches are sampled STRICTLY inside `region_mask` (facade/house
        minus dilated window/door openings); non-region pixels are neutralised
        before encoding and their material probabilities are averaged. When the
        region is empty (Module 1 found no building) the result is a fail-closed
        `classified_region="none"` — the classifier never sees the whole image.

        Args:
            region_mask: `[H, W]` wall mask from Module 1. None/empty -> "none".
        """
        rgb, width, height = self._load_rgb()
        patches, source = sample_region_patches(
            rgb, region_mask,
            patch_size=self.cfg.patch_size,
            max_patches=self.cfg.max_patches,
            min_wall_ratio=self.cfg.min_patch_wall_ratio,
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
            metadata=Metadata(candidate_materials=self.cfg.materials, model=self.cfg.model_id),
        )

    def classify_instances(
        self, detections: dict[str, list[Bbox]]
    ) -> PerObjectMaterialResult:
        """Classify each detection's crop independently and group by class.

        Args:
            detections: Mapping of class label (`window`, `door`, `house`) to a
                list of xyxy pixel bboxes.

        Returns:
            `PerObjectMaterialResult` with one entry per instance.
        """
        rgb, width, height = self._load_rgb()
        objects: dict[str, list[ObjectMaterial]] = {}
        min_size = self.cfg.per_object_min_size_px
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
                crop = rgb[y1:y2, x1:x2]
                scored = self._classify_one(crop)
                entries.append(
                    ObjectMaterial(
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        dominant_material=scored[0],
                        properties=self._properties_for(scored[0].label),
                    )
                )
            objects[label] = entries
        return PerObjectMaterialResult(
            image=ImageInfo(id=self.image_path.stem, width=width, height=height),
            view_type=self.view_type,
            objects=objects,
            metadata=Metadata(candidate_materials=self.cfg.materials, model=self.cfg.model_id),
        )
