"""Scene-level gate for Module 1's facade mask (Cityscapes SegFormer).

Module 1's CMP checkpoint is trained on tightly-cropped, head-on facade
photographs where essentially the whole frame is facade, so it has never had to
learn what "not a facade" looks like. On street-level captures it returns a
blob that includes hedges, parked cars, pavement, sky and neighbouring wings —
and since Module 2's height is now measured from that mask's extent, the blob
costs height directly.

This package runs a Cityscapes-finetuned SegFormer, a model that *is* in domain
for street photography, and uses it only as a gate: keep the parts of Module 1's
mask that the street model also calls `building`. Module 1 keeps its real job —
window, door, sill, balcony — which is what CMP actually taught it.

What the gate cannot do is separate an abutting neighbour at the same depth;
both wings are genuinely `building` to any segmenter. That residual is reported
(`components_kept`) rather than hidden, and is a depth problem, not a semantic
one.
"""
from .labels import BUILDING, CLASS_NAMES, GROUND, GROUPS, OCCLUDER, SKY, ids_for
from .parser import SceneMasks, SceneParser
from .refine import BuildingRegion, refine_building_region

__all__ = [
    "SceneParser",
    "SceneMasks",
    "BuildingRegion",
    "refine_building_region",
    "CLASS_NAMES",
    "GROUPS",
    "BUILDING",
    "OCCLUDER",
    "GROUND",
    "SKY",
    "ids_for",
]
