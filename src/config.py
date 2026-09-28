"""Typed configuration for the Precise pipeline.

Loads `config.yaml` at the repo root into a nested pydantic model. Each
module class receives its own typed section in `__init__`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


ViewType = Literal["facade", "topdown"]

PRECISE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PRECISE_ROOT / "config.yaml"


class PipelineConfig(BaseModel):
    """Shared pipeline settings."""

    default_image: str
    view_type: ViewType = "facade"
    materials_all: bool = True


class FacadeParsingConfig(BaseModel):
    """Module 1 (SEEM) configuration."""

    threshold: float = 0.6
    backbone: str = "samvit-l"
    crop_foreground: bool = True
    save_masks: bool = False
    prompts: list[str] = Field(default_factory=lambda: ["house", "window", "door"])
    prompt_synonyms: dict[str, list[str]] = Field(default_factory=dict)
    input_resize_short_side: int = 512
    min_polygon_area_px: float = 50.0
    poly_simplify_eps_ratio: float = 0.002
    overlay_alpha: float = 0.1
    label_colors_bgr: dict[str, list[int]] = Field(default_factory=dict)
    model_version: str = "v1.0"


class SceneParsingConfig(BaseModel):
    """Cityscapes scene gate for Module 1's facade mask."""

    enabled: bool = True
    # Cityscapes-finetuned SegFormer from the Hub; b0..b5 trade speed for
    # accuracy. Weights download on first run and are then cached.
    model_id: str = "nvidia/segformer-b4-finetuned-cityscapes-1024-1024"
    device: Literal["auto", "cpu"] = "auto"
    # Cityscapes checkpoints are trained around 1024 px; a 413 px facade crop
    # segments noticeably better upsampled to that scale first.
    input_long_side: int = 1024
    # Refuse the gate when it would delete all but this fraction of Module 1's
    # mask — more likely a bad scene parse than a building that is not one.
    min_kept_ratio: float = 0.25
    # Speckle cleanup, then components smaller than this fraction of the
    # largest are dropped (a tree splitting a facade leaves two real halves,
    # so it is not "keep only the largest").
    morph_kernel_px: int = 5
    min_component_ratio: float = 0.15
    # A column needs this fraction of the image height in building pixels
    # before its upper/lower edge is trusted.
    min_column_pixels_ratio: float = 0.05
    # Extent percentile per edge: the bottom takes the 100-p'th of per-column
    # lower edges, favouring columns that reached the base; the top takes the
    # p'th. Robust to a stray row without truncating a real penthouse.
    edge_percentile: float = 5.0
    # How far below a column's lower edge to look for an occluder when
    # deciding whether that column ever found the building's base.
    base_probe_ratio: float = 0.04
    # Slack around the refined extent when deciding whether a window or door
    # belongs to this building, as a fraction of the image's longer side.
    opening_margin_ratio: float = 0.02
    model_version: str = "v1.0"


class BuildingFeaturesConfig(BaseModel):
    """Module 2 (geometry heuristics) configuration."""

    assumed_door_height_m: float = 2.05
    assumed_floor_height_m: float = 3.2
    window_cluster_tol_ratio: float = 0.6
    plausible_floor_height_m: list[float] = Field(
        default_factory=lambda: [2.4, 4.5], min_length=2, max_length=2
    )
    bottom_door_tol_ratio: float = 0.15
    # Largest height/width a door candidate may have and still be picked as
    # the scale anchor. A 0.9 m x 2.05 m leaf is 2.3, and obliquity does not
    # take a real door much past 3; anything far beyond that is a shadow down
    # a reveal or a drainpipe, and "tallest wins" is exactly the contest a
    # sliver is best at. Only the upper bound is applied — a *wide* candidate
    # is a double door, a garage, or an entrance whose foot has not been put
    # back on the ground yet.
    door_max_aspect: float = Field(4.0, gt=0.0)
    # A door reaches the ground, so one whose mask stops short of the resolved
    # base row is truncated — by a parked car, or by the shopfront glazing that
    # took its lower half — and a short span makes the building tall. Its foot
    # is moved down to the base, but only when the storey pitch says it is too
    # short to be a whole door: 2.05 m in a 3 m storey is 0.68 of a pitch, so
    # a door already past `door_short_storeys` is taken as complete and left
    # alone, and a correction that would push one past `door_max_storeys` is
    # refused. `door_base_snap_ratio` is the fallback bound on facades too
    # irregular to fit a pitch; 0 disables the correction there.
    # See `_snap_doors_to_base`.
    door_short_storeys: float = Field(0.55, ge=0.0)
    door_max_storeys: float = Field(0.85, gt=0.0)
    door_base_snap_ratio: float = Field(0.5, ge=0.0)

    # --- metric scale recovery (src/building_features/scale.py) -------------
    # Reference lengths for the door-free scale anchors, and the relative
    # 1-sigma uncertainty each one deserves. Sigmas set the inverse-variance
    # weights when several anchors are fused, so their *ratios* matter more
    # than their absolute values.
    assumed_floor_pitch_m: float = 3.0
    assumed_window_height_m: float = 1.45
    sigma_rel_door: float = 0.08
    sigma_rel_floor_pitch: float = 0.10
    sigma_rel_window: float = 0.25
    # Anchors further than this from the median (in log space) are discarded
    # before fusing: 0.35 ~ +/-42%.
    scale_agreement_log_tol: float = 0.35
    # Window rows needed before solving for the vertical vanishing point, and
    # the fraction of the affine residual the perspective fit must beat.
    min_rows_for_perspective: int = 4
    perspective_improvement: float = 0.85
    # Largest scale ratio across the detected window rows that a candidate
    # vanishing point may imply. ~3 is a normal street; 8 allows a narrow
    # street and a steep look-up.
    max_vertical_scale_ratio: float = 8.0
    # Storeys of facade the model may place beyond the outermost window rows.
    # These bound the extrapolation, which is where a bad vanishing point
    # does its damage: above the top row sits half a storey plus a parapet or
    # setback penthouse, below the bottom row half a storey plus the taller
    # shopfront storey.
    max_storeys_above_top_row: float = 2.5
    max_storeys_below_bottom_row: float = 3.0
    # A row holding fewer than this fraction of the median row's boxes is a
    # segmentation artefact, not a storey, and is dropped.
    min_row_members_ratio: float = 0.34
    # Module 1 classes whose polygons mark storey lines. Used as a source of
    # rows only when windows alone are too few to fit the vertical model.
    floor_line_labels: list[str] = Field(
        default_factory=lambda: ["sill", "balcony", "cornice", "molding"]
    )
    # v2.0: height measured through a fitted vertical model + fused scale
    # anchors instead of door-or-constant; output gains the `scale` block.
    model_version: str = "v2.0"


MaterialCategory = Literal[
    "metal",
    "ceramic",
    "masonry",
    "concrete",
    "gypsum",
    "natural-stone",
    "wood",
    "polymer",
    "composite",
    "earthen",
]


class MaterialProperties(BaseModel):
    """Fixed, representative mechanical properties for one material.

    These are static reference values (single representative figure per
    property for the grade described in `note`) — they are looked up by
    material label, never computed at runtime, so a given material always
    reports the same numbers. Sourced from standard engineering references
    (Engineering ToolBox, MatWeb, Callister, Eurocode/AISC, Wood Handbook).
    """

    category: MaterialCategory
    density_kg_m3: float = Field(gt=0, description="Bulk density (kg/m³).")
    youngs_modulus_gpa: float = Field(
        gt=0, description="Modulus of elasticity / stiffness (GPa)."
    )
    compressive_strength_mpa: float = Field(
        gt=0, description="Representative compressive strength (MPa)."
    )
    tensile_strength_mpa: float = Field(
        gt=0, description="Representative tensile strength (MPa)."
    )
    poisson_ratio: float = Field(
        ge=0.0, le=0.5, description="Poisson's ratio (dimensionless)."
    )
    note: str = Field(description="Assumed grade / condition the values represent.")


class BuildingMaterialsConfig(BaseModel):
    """Module 3 (CLIP material) configuration."""

    # Module 3 backend:
    #   "siglip2" — zero-shot open_clip (open-vocab, no training)
    #   "minc"    — timm ConvNeXt fine-tuned on MINC-2500 (8 classes, no concrete)
    #   "facade"  — DINOv3 fine-tuned on the pooled Facade-8 corpus (adds
    #               concrete + render; src/building_materials_facade)
    material_backend: Literal["siglip2", "minc", "facade"] = "siglip2"
    model_name: str = "ViT-L-16-SigLIP2-384"
    pretrained: str = "webli"
    model_id: str = "siglip2-large-16-384"
    # MINC backend
    minc_checkpoint: str = "src/building_materials_minc/runs/best.pt"
    minc_arch: str = "convnext_tiny"
    # DINOv3 Facade-8 backend
    facade_checkpoint: str = "src/building_materials_facade/runs/best.pt"
    facade_arch: str = "vit_base_patch16_dinov3.lvd1689m"
    facade_patch_size: int = 256
    facade_max_patches: int = 16
    facade_min_wall_ratio: float = Field(0.7, ge=0.0, le=1.0)
    per_object_min_size_px: int = 24
    # Patch-voting (SigLIP2 backend): native-resolution wall patches, averaged.
    patch_size: int = 384
    max_patches: int = 12
    min_patch_wall_ratio: float = Field(0.6, ge=0.0, le=1.0)
    # Patch-voting (MINC backend): smaller patches fit the stone between windows,
    # and a high wall-purity threshold keeps actual glass/doors out of the crop.
    minc_patch_size: int = 128
    minc_max_patches: int = 24
    minc_min_wall_ratio: float = Field(0.9, ge=0.0, le=1.0)
    # Exclude a margin around window/door openings from the wall region (ratio of
    # the image's short side) so patches stay off reflective frames/edges.
    wall_opening_dilation_ratio: float = Field(0.012, ge=0.0)
    materials: list[str] = Field(default_factory=list)
    prompt_templates: dict[str, list[str]] = Field(default_factory=dict)
    material_properties: dict[str, MaterialProperties] = Field(default_factory=dict)


class FacadeParsingSegformerConfig(BaseModel):
    """Optional Module 1 backend: the trained 12-class CMP SegFormer.

    Used when `client.py` runs with `manual_seg=True`. The 12 CMP classes are
    reduced to house(=facade)/window/door for the pipeline; overlay colors are
    taken from `facade_parsing.label_colors_bgr` for parity with SEEM.
    """

    checkpoint: str = "src/facade_parsing_segm/segformer_cmp/runs/best"
    prob_threshold: float = Field(0.5, ge=0.0, le=1.0)
    overlay_alpha: float = Field(0.1, ge=0.0, le=1.0)
    min_polygon_area_px: float = Field(80.0, ge=0.0)
    poly_simplify_eps_ratio: float = Field(0.01, ge=0.0)

    # --- door recovery (segformer_cmp/door_refine.py) ------------------------
    # `door` is the rarest class in CMP, and a glass entrance set into a
    # shopfront loses the argmax to `shop`, leaving Module 2 the dark lintel
    # above the leaves as its 2.05 m scale anchor. Two steps recover the rest
    # from the same probabilities: undo the class prior, then walk the door
    # down to the base of the opening. Requires `door_max_aspect` in Module 2.
    door_refine: bool = True
    door_prior_tau: float = Field(1.0, ge=0.0)
    door_outbids: list[str] = Field(default_factory=lambda: ["shop"])
    door_min_prob: float = Field(0.10, ge=0.0, le=1.0)
    # A prior correction is pointwise and has no sense of direction, so it
    # wins pixels above the door as readily as below it — which is the one
    # thing the door's span cannot afford. Restricting wins to pixels below an
    # existing door pixel in the same column leaves only the correction the
    # physics supports: a door carried down toward the ground it stands on.
    door_promote_downward_only: bool = True
    # Step 2, the walk. Classes it may pass through, and the fraction of a row
    # (across the door's own width) that must belong to them to continue.
    door_extend_into: list[str] = Field(default_factory=lambda: ["shop"])
    door_extend_row_ratio: float = Field(0.5, ge=0.0, le=1.0)
    # Ceiling on the recovered door, as a fraction of the facade's pixel
    # height. A 2.05 m door is ~0.29 of a two-storey facade and ~0.06 of a
    # twelve-storey one, so this only ever binds on low buildings.
    door_max_facade_ratio: float = Field(0.35, gt=0.0, le=1.0)
    # Step 3, widening — the mask only, never the measurement. Where the
    # columns beside a door are still shopfront, the entrance is wider than
    # the label (package 20's left entrance is 30 px of mask on a ~107 px
    # gated opening). Bounded by the door's own height, since a double leaf is
    # 1.8 m wide against 2.05 m tall. Module 2 is deliberately not shown the
    # result: widening cannot change a door's height, but it does change its
    # aspect, which is what tells a door from a drainpipe — feeding widened
    # doors to the anchor picker put package 16's 9 px sliver back in play and
    # moved the door-anchor median from 0.245 to 0.256. It is used for the
    # overlay and for the opening region Module 3 subtracts from the wall.
    door_widen: bool = True
    door_widen_max_width_ratio: float = Field(1.3, gt=0.0)
    door_widen_col_ratio: float = Field(0.5, ge=0.0, le=1.0)

    # --- rectified second opinion (segformer_cmp/rectify.py) ----------------
    # CMP is rectified head-on facades; these are street captures taken from
    # across a road. Measured on the held-out CMP split, a 0.10 oblique warp
    # costs door IoU 31% while even severe look-up costs 17% — obliquity is
    # what the door class cannot survive, and door is what Module 2 measures
    # scale from. So the facade is warped fronto-parallel and asked again.
    #
    # The merge is deliberately ADD-ONLY: door pixels the rectified pass is
    # confident about are taken, nothing is ever removed on its word. The
    # checkpoint is unstable enough that the rectified pass loses doors about
    # as often as it finds them (it gains on packages 8, 14, 20 and loses on
    # 5, 10, 18, 19), and add-only turns every one of those losses into a
    # no-op while keeping the gains — package 20's gated entrance goes from
    # p(door) 0.02 to 0.80. It costs one extra forward pass per image.
    door_rectified_pass: bool = True
    # A rectified door must win its own argmax and clear this, so only
    # confident disagreements are imported.
    rect_min_door_prob: float = Field(0.5, ge=0.0, le=1.0)
    # Line detection and vanishing-point RANSAC.
    rect_min_segment_ratio: float = Field(0.02, ge=0.0)
    rect_angle_tol_deg: float = Field(25.0, gt=0.0, le=45.0)
    rect_ransac_iters: int = Field(500, ge=1)
    rect_inlier_tol: float = Field(0.012, gt=0.0)
    rect_seed: int = 0
    # Guards. Each rejects a warp that was observed to be worse than none:
    # an implausible focal length means the two vanishing points were not
    # orthogonal directions, a large rotation means the axes came out swapped,
    # and high anisotropy is the affine-rectification wedge failure.
    rect_min_focal_ratio: float = Field(0.3, gt=0.0)
    rect_max_focal_ratio: float = Field(8.0, gt=0.0)
    rect_max_rotation_deg: float = Field(20.0, gt=0.0)
    rect_max_anisotropy: float = Field(1.6, ge=1.0)

    model_version: str = "v1.0"


class PreciseConfig(BaseModel):
    """Top-level configuration for the Precise pipeline."""

    pipeline: PipelineConfig
    facade_parsing: FacadeParsingConfig
    building_features: BuildingFeaturesConfig
    building_materials: BuildingMaterialsConfig
    # Optional: only used when client.py runs with manual_seg=True.
    facade_parsing_segformer: FacadeParsingSegformerConfig = Field(
        default_factory=FacadeParsingSegformerConfig
    )
    # Scene gate applied to Module 1's facade mask before Modules 2 and 3.
    scene_parsing: SceneParsingConfig = Field(default_factory=SceneParsingConfig)


def load_config(path: str | Path | None = None) -> PreciseConfig:
    """Load and validate `config.yaml`.

    Args:
        path: Optional explicit path; defaults to `<repo>/config.yaml`.

    Returns:
        Parsed `PreciseConfig`.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return PreciseConfig(**raw)
