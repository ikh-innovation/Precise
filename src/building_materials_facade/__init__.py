"""Module 3 backend — DINOv3 Facade-8 material classifier.

A facade-domain replacement for the MINC backend. A self-supervised DINOv3
ViT-B/16 backbone (frozen probe by default) + MLP head, trained on a pooled,
facade-specific corpus (MINC patches + OpenFACADES + URC + London/Scotland)
mapped to the Facade-8 taxonomy — which, unlike MINC, includes `concrete` and
`render` (plaster/stucco). Same inference contract as the other Module 3
backends; respects the red-region-only rule via the shared region sampler.
"""
from .classifier import FacadeMaterials
from .labels import NUM_CLASSES, TARGET_CLASSES

__all__ = ["FacadeMaterials", "TARGET_CLASSES", "NUM_CLASSES"]
