"""Module 3 (custom) — material classification via a MINC-trained ConvNeXt.

Approach: zero-shot vision-language models read facade *material* unreliably —
texture is fine-grained and easily confused with windows and reflections — so
this backend is supervised. A ConvNeXt-Tiny (a modern, efficient CNN, strong and
data-efficient at texture/material recognition) is fine-tuned on MINC-2500
(Materials in Context) restricted to a facade subset: brick, glass, metal,
painted, plastic, stone, tile, wood. It trains in minutes on one GPU and runs
fast at inference. To name the *wall* material rather than glass/doors, inference
votes over many small, high-purity patches sampled inside the facade-minus-
openings region. `MincMaterials` is a drop-in for the zero-shot SigLIP2
`BuildingMaterials` (same `classify()` → `MaterialClassificationResult`),
selected via `config.building_materials.material_backend == "minc"`. Train with
`train.py`.
"""
from .classifier import MincMaterials
from .labels import NUM_CLASSES, TARGET_CLASSES

__all__ = ["MincMaterials", "TARGET_CLASSES", "NUM_CLASSES"]
