"""Recover the part of the entrance the argmax label map leaves behind.

Module 1 emits one label per pixel (`pred = probs.argmax(0)`), so a door only
exists where `door` outscores all eleven other CMP classes. On the street-level
captures in `data/package_runs` it often does not, and two things go wrong at
once.

**The prior is against it.** `door` is the rarest class in CMP — 1.26% of the
training pixels, against 2.98% for `shop` and 38.6% for `facade`. A model
fitted with plain cross-entropy learns that imbalance and carries it into
every argmax, so on a pixel the two classes describe equally well, `shop` wins
because shopfronts were more common in training, not because the pixel is more
shopfront than door. And `shop` is the wrong class to lose to: it is the
shopfront glazing, and a glass entrance set into a shopfront is the same thing
to within a mullion. In 17 of the 20 packages the `shop` region is 4-30x the
`door` region and covers the same rows.

**What survives is the wrong part.** The fragment that outscores `shop` is the
dark recessed lintel *above* the leaves. Package 19 is the clean example — 42
px of header for an entrance about 145 px tall, read as a 0.72 m door and
turned into 42.7 m of building on a facade the storey rhythm puts at 25 m. A
door that is too short makes the building too tall, through the most heavily
weighted anchor there is.

Two steps, both working off probabilities the same forward pass already
produced — the checkpoint is not retrained:

1. `promote_door` removes the training prior from the decision (logit
   adjustment: score by `p(c|x) / prior(c)**tau`), for `door` alone and only
   against the classes it is allowed to outbid.
2. `extend_to_opening_base` walks each door down through the rows below it
   that are still predominantly shopfront, stopping at the first row that is
   not — on these images, the pavement.

The result is written back into `pred` rather than carried alongside it, so
the emitted masks, the polygons, the overlay and Module 3's wall all see one
consistent labelling in which the classes still do not overlap.

Both steps are **downward-biased on purpose**, and that is the design. A door
reaches the ground by definition and the glazing it is set into reaches the
ground with it, so downward is the one direction in which growing a mask is an
argument rather than a guess. Every version allowed to grow the other way was
measured and was worse; see `promote_door` and the note at the end.

Measured over all 20 packages against the storey-pitch ruler, with the
`door_max_aspect` gate in Module 2's picker (which this pass *requires*): the
door anchor's median error against 2.05 m falls from 0.266 to 0.245 in log
terms, and over the anchors actually fused from 0.209 to 0.170. Package 9 goes
1.26 m -> 2.01 m, package 19 0.72 m -> 2.07 m, package 16's 9 px-wide sliver
is dropped rather than believed; nothing regresses and nothing falls back to
`floor_count * 3.2`. Ten of the twenty get a visibly fuller door in the
overlay, most of them without their anchor moving at all.

**This pass needs the aspect gate.** Growing doors changes which one is
tallest, and "tallest wins" is the one contest a sliver is good at: on package
19 a 12 px-wide strip beside the ATM grew past the real Santander entrance and
took the anchor. `door_max_aspect` in `_pick_door_anchor` keeps the choice on
something door-shaped; turning this pass on without it makes package 19 worse
than doing nothing.

Two things this module does **not** do.

It does not judge whether its seed is a door at all. Packages 9 and 14 seed on
a blob in a tree and a shadow between two shopfronts, and these steps measure
those more accurately rather than rejecting them. The base-contact and
occluder tests that would reject them need the scene gate's masks, which do
not exist at this point in the pipeline.

It does not grow a door *sideways*, and package 20's left entrance is why that
is a separate decision rather than an oversight. There the mask is 30 px wide
on a ~107 px gated opening, and the iron gate beside it is confidently `shop`
(p(shop) = 0.93, p(door) = 0.03 — the prior correction lifts door only to
0.06, nowhere near). A lateral walk mirroring `extend_to_opening_base` does
capture it, and the overlay is plainly better for it, but measured over all 20
it costs height accuracy: the median goes 0.245 -> 0.256 and package 16's
sliver comes back, because widening a sliver is exactly what makes it pass the
aspect gate. If it is added it should widen the *mask* while the anchor is
still chosen and measured on the un-widened component.

An earlier version also grew the mask sideways by hysteresis, seeding on
argmax-door and absorbing neighbouring pixels above a lower probability. It is
not here because it was measured and it was worse: absorbing from `facade`
took the wall above the lintel and pushed packages 1 and 2 from 2.07 m and
2.11 m — essentially exact — to 4.04 m and 4.39 m, and restricting it to
`shop` still cost package 2 (2.11 -> 3.99). Letting `door` outbid `facade` is
worse still, and not only for the span: `facade` *is* the building extent, so
door eating it shrinks the mask the height is measured from — three packages
fell back and package 10 collapsed from 30.0 m to 12.8 m.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .labels import prior_ratio


@dataclass(frozen=True)
class DoorRefinement:
    """What the refinement did, so a run can be audited from its log alone.

    Attributes:
        seed_px: Door pixels the argmax produced, before any correction.
        promoted_px: Pixels won from the classes door was allowed to outbid.
        extended_px: Pixels added by the walk to the opening base.
        components: Door components the walk ran on.
        extended: Components whose bottom edge actually moved.
    """

    seed_px: int = 0
    promoted_px: int = 0
    extended_px: int = 0
    components: int = 0
    extended: int = 0

    @property
    def changed(self) -> bool:
        """True when the door mask is not simply the argmax mask."""
        return bool(self.promoted_px or self.extended_px)

    def as_dict(self) -> dict[str, int]:
        """Plain mapping for logging or embedding in a result payload."""
        return {
            "seed_px": self.seed_px,
            "promoted_px": self.promoted_px,
            "extended_px": self.extended_px,
            "components": self.components,
            "extended": self.extended,
        }


def promote_door(
    probs: np.ndarray,
    pred: np.ndarray,
    door_id: int,
    outbid: dict[int, float],
    min_prob: float,
    downward_only: bool = True,
) -> np.ndarray:
    """Let the door class outbid the classes it systematically loses to.

    `door` is the rarest class in CMP — 1.26% of the training pixels, against
    2.98% for `shop` and 38.6% for `facade`. A model fitted with plain
    cross-entropy learns that imbalance as a prior and carries it into every
    argmax, so on a pixel the two classes describe equally well, `shop` wins
    because shopfronts were more common in the training set, not because this
    pixel is more shopfront than door.

    Scoring by `p(c|x) / prior(c)**tau` instead of `p(c|x)` removes that prior
    from the decision. Applied only to `door`, and only against the classes
    named in `outbid`, it becomes: a pixel currently labelled `shop` flips to
    `door` when `p(door)` is within the classes' prior ratio of `p(shop)`.
    Everything else keeps the argmax it had, so `window`, `balcony` and the
    storey-line classes Module 2 measures rows from are untouched.

    `min_prob` is the floor that keeps this honest. The prior ratio is a large
    multiplier, and without a floor it would flip pixels where the model gave
    door 0.01 and shop 0.02 — a contest between two numbers that both mean
    "not this".

    `downward_only` is what makes the correction safe, and it was measured
    rather than assumed. A prior correction is pointwise: it has no sense of
    direction, and the pixels it wins sit above and beside the door as readily
    as below it. Unrestricted it took the shopfront band above the entrance in
    package 2 and pushed a 2.11 m door — essentially exact — to 3.65 m, and on
    package 19 it grew the door just enough to look complete, which stopped
    Module 2 snapping it to the ground and left the entrance half-recovered at
    1.72 m where the snap alone reached 2.29 m. Restricted to pixels below an
    existing door pixel in the same column it can only ever do the thing the
    physics supports: carry a door down toward the ground it must stand on.

    Args:
        probs: `[C, H, W]` per-class probabilities.
        pred: `[H, W]` argmax label map.
        door_id: Class id of `door`.
        outbid: Class id -> the factor door's probability is multiplied by
            when contesting that class.
        min_prob: Lowest `p(door)` that may win a contest.
        downward_only: Restrict wins to pixels below an existing door pixel in
            the same column.

    Returns:
        `[H, W]` bool door mask: the argmax door pixels plus everything won.
    """
    door = pred == door_id
    if not outbid:
        return door
    door_p = probs[door_id]
    winner_p = np.take_along_axis(probs, pred[None], axis=0)[0]
    factor = np.ones_like(door_p)
    for class_id, value in outbid.items():
        factor[pred == class_id] = value
    contested = np.isin(pred, list(outbid))
    won = contested & (door_p >= min_prob) & (door_p * factor >= winner_p)
    if downward_only:
        won &= np.cumsum(door, axis=0) > 0
    return door | won


def extend_to_opening_base(
    door: np.ndarray,
    opening: np.ndarray,
    bottom_limit: int,
    max_height: int,
    min_row_ratio: float,
) -> tuple[np.ndarray, int]:
    """Walk each door component down through the shopfront it is set into.

    When the rows immediately below a door component are still shopfront
    glazing, the entrance did not end there — the label did. The walk stops at
    the first row that is not predominantly glazing, and a gap ends it, so a
    separate opening further down the facade cannot be annexed.

    Args:
        door: `[H, W]` bool door mask to extend.
        opening: `[H, W]` bool mask of the classes the walk may pass through —
            the configured ground-floor opening classes, plus `door` itself.
        bottom_limit: Last row the walk may reach; the foot of the facade.
        max_height: Largest total pixel height any one door may end up with.
        min_row_ratio: Fraction of a row, across the component's own width,
            that must be `opening` for the walk to continue.

    Returns:
        `(mask, extended)` — the extended mask and the number of components
        whose bottom edge moved.
    """
    out = door.copy()
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        door.astype(np.uint8), connectivity=8
    )
    moved = 0
    for i in range(1, count):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        start = y + h
        stop = min(bottom_limit, y + max_height - 1)
        if stop < start:
            continue

        band = opening[start:stop + 1, x:x + w]
        if band.size == 0:
            continue
        rows = band.mean(axis=1) >= min_row_ratio
        run = int(rows.size if rows.all() else np.argmin(rows))
        if run == 0:
            continue
        out[start:start + run, x:x + w] |= band[:run]
        moved += 1
    return out, moved


def widen_to_opening(
    door: np.ndarray,
    opening: np.ndarray,
    max_width_ratio: float,
    min_col_ratio: float,
) -> tuple[np.ndarray, int]:
    """Grow each door sideways through the opening it is set into.

    The mirror of `extend_to_opening_base`, along x. Where the columns beside
    a door are still shopfront over the door's own rows, the entrance is wider
    than the label: package 20's left entrance is a 30 px strip of mask on a
    ~107 px gated opening, because the model reads the see-through iron gate
    beside the leaf as glazing (p(shop) = 0.93 against p(door) = 0.03 — not
    something any threshold or prior correction reaches).

    Growth is bounded by the door's own height. A double leaf is about 1.8 m
    wide against 2.05 m tall, so a real entrance stays under roughly 1 —
    `max_width_ratio` is deliberately a little looser, and it is the only
    thing standing between this and annexing a whole shopfront, which is
    metres wide and would swallow the ground floor.

    **This result is for the mask, not for the measurement.** Widening never
    changes a component's row span, so it cannot change a door's height — but
    it does change its *aspect*, and aspect is what `_pick_door_anchor` uses
    to tell a door from a drainpipe. Measured over all 20 packages, feeding
    widened doors to Module 2 puts package 16's 9 px sliver back in play (it
    widens into a door shape) and moves the door-anchor median from 0.245 to
    0.256. So the adapter keeps this out of the geometry Module 2 reads and
    uses it only where a truer outline helps: the overlay, and the opening
    region Module 3 subtracts from the wall.

    Args:
        door: `[H, W]` bool door mask to widen.
        opening: `[H, W]` bool mask of classes the growth may pass through.
        max_width_ratio: Widest a door may become, as a multiple of its own
            height.
        min_col_ratio: Fraction of a column, over the component's own rows,
            that must be `opening` for the growth to continue through it.

    Returns:
        `(mask, widened)` — the widened mask and how many components grew.
    """
    out = door.copy()
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        door.astype(np.uint8), connectivity=8
    )
    widened = 0
    for i in range(1, count):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        budget = int(round(max_width_ratio * h))
        if w >= budget:
            continue
        band = opening[y:y + h]
        usable = band.mean(axis=0) >= min_col_ratio
        left, right = x, x + w - 1
        while left > 0 and usable[left - 1] and (right - left + 1) < budget:
            left -= 1
        while right + 1 < door.shape[1] and usable[right + 1] and (right - left + 1) < budget:
            right += 1
        if left == x and right == x + w - 1:
            continue
        out[y:y + h, left:right + 1] |= band[:, left:right + 1]
        widened += 1
    return out, widened


def widen_doors(pred: np.ndarray, name_to_id: dict[str, int], cfg) -> np.ndarray:
    """The door mask of `pred`, widened into its opening. Presentation only.

    Returns the input's own door mask unchanged when widening is disabled or
    there is no door, so callers can use the result unconditionally.
    """
    door_id = name_to_id.get("door")
    if door_id is None:
        return np.zeros(pred.shape, dtype=bool)
    door = pred == door_id
    if not cfg.door_widen or not door.any():
        return door
    extend_ids = [name_to_id[n] for n in cfg.door_extend_into if n in name_to_id]
    opening = np.isin(pred, extend_ids) | door if extend_ids else door
    wide, _ = widen_to_opening(
        door, opening, cfg.door_widen_max_width_ratio, cfg.door_widen_col_ratio
    )
    return wide


def refine_doors(
    probs: np.ndarray,
    pred: np.ndarray,
    name_to_id: dict[str, int],
    cfg,
) -> tuple[np.ndarray, DoorRefinement]:
    """Undo the class prior on `door`, then walk each door to its opening base.

    Args:
        probs: `[C, H, W]` per-class probabilities at the image's resolution.
        pred: `[H, W]` argmax label map to refine.
        name_to_id: CMP class name -> class id, from the checkpoint's config.
        cfg: `FacadeParsingSegformerConfig`.

    Returns:
        `(pred, report)`. `pred` is a new array; the input is not modified.
        With no door to work with the input is returned unchanged alongside an
        all-zero report.
    """
    door_id = name_to_id.get("door")
    if door_id is None:
        return pred, DoorRefinement()

    seeds = pred == door_id
    seed_px = int(seeds.sum())

    outbid = {
        name_to_id[n]: prior_ratio("door", n, cfg.door_prior_tau)
        for n in cfg.door_outbids
        if n in name_to_id and n != "door"
    }
    door = promote_door(
        probs, pred, door_id, outbid, cfg.door_min_prob,
        downward_only=cfg.door_promote_downward_only,
    )
    promoted_px = int(door.sum()) - seed_px
    if not door.any():
        return pred, DoorRefinement()

    # The facade's extent sets both the floor the walk may reach and the scale
    # the height ceiling is expressed in, so the ceiling means the same thing
    # on a two-storey terrace as on a twelve-storey block.
    facade_id = name_to_id.get("facade")
    facade_rows = (
        np.flatnonzero((pred == facade_id).any(axis=1))
        if facade_id is not None
        else np.array([], dtype=int)
    )
    if facade_rows.size:
        facade_top, facade_bottom = int(facade_rows[0]), int(facade_rows[-1])
    else:
        facade_top, facade_bottom = 0, pred.shape[0] - 1
    max_height = max(
        1, int(round(cfg.door_max_facade_ratio * (facade_bottom - facade_top + 1)))
    )

    extend_ids = [name_to_id[n] for n in cfg.door_extend_into if n in name_to_id]
    opening = np.isin(pred, extend_ids) | door if extend_ids else door
    extended, moved = extend_to_opening_base(
        door,
        opening,
        bottom_limit=facade_bottom,
        max_height=max_height,
        min_row_ratio=cfg.door_extend_row_ratio,
    )

    refined = pred.copy()
    refined[extended] = door_id
    return refined, DoorRefinement(
        seed_px=seed_px,
        promoted_px=promoted_px,
        extended_px=int(extended.sum()) - int(door.sum()),
        components=int(cv2.connectedComponents(door.astype(np.uint8), 8)[0] - 1),
        extended=moved,
    )
