"""Cityscapes 19-class label scheme, grouped by what each class means to us.

Module 1's CMP SegFormer is trained on tightly-cropped, head-on facade
photographs in which essentially the whole frame *is* facade. Applied to
street-level captures it has no notion of "not a facade", so it labels
oleander, parked cars, pavement and sky as `facade` and hands Module 2 a blob.

Cityscapes is the opposite domain — vehicle-height photographs of real streets
— and carries exactly the classes needed to undo that: `building` and `wall`
are the building; `vegetation`, vehicles, people and street furniture are
things standing *in front of* it; `road`/`sidewalk`/`terrain` are the ground it
stands on; `sky` is behind it. Grouping the 19 classes this way is all the
scene understanding the refinement needs.
"""
from __future__ import annotations

# Cityscapes train ids 0..18, in their canonical order.
CLASS_NAMES: list[str] = [
    "road",          # 0
    "sidewalk",      # 1
    "building",      # 2
    "wall",          # 3
    "fence",         # 4
    "pole",          # 5
    "traffic light",  # 6
    "traffic sign",  # 7
    "vegetation",    # 8
    "terrain",       # 9
    "sky",           # 10
    "person",        # 11
    "rider",         # 12
    "car",           # 13
    "truck",         # 14
    "bus",           # 15
    "train",         # 16
    "motorcycle",    # 17
    "bicycle",       # 18
]

# The building itself. `wall` is Cityscapes' free-standing wall, which is a
# building surface often segmented where a facade meets a boundary wall.
BUILDING: frozenset[str] = frozenset({"building", "wall"})

# Things that stand between the camera and the facade. Where these overlap the
# facade the building is *occluded*, not absent — which is why removing them
# raises the mask's lower edge rather than revealing the true base.
OCCLUDER: frozenset[str] = frozenset({
    "vegetation", "fence", "pole", "traffic light", "traffic sign",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
})

# The ground plane the building stands on. Its visible upper edge in front of a
# facade is at or below the building's base, so it bounds how far down the
# base can reasonably be extrapolated.
GROUND: frozenset[str] = frozenset({"road", "sidewalk", "terrain"})

# Behind everything. Facade mask bleeding into sky is the commonest way the
# top edge goes wrong.
SKY: frozenset[str] = frozenset({"sky"})

GROUPS: dict[str, frozenset[str]] = {
    "building": BUILDING,
    "occluder": OCCLUDER,
    "ground": GROUND,
    "sky": SKY,
}


def ids_for(names: frozenset[str] | set[str] | list[str]) -> list[int]:
    """Train ids for the given class names, ignoring any that are unknown."""
    return [i for i, n in enumerate(CLASS_NAMES) if n in set(names)]
