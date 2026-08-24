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


class BuildingFeaturesConfig(BaseModel):
    """Module 2 (geometry heuristics) configuration."""

    assumed_door_height_m: float = 2.05
    assumed_floor_height_m: float = 3.2
    window_cluster_tol_ratio: float = 0.6
    plausible_floor_height_m: list[float] = Field(
        default_factory=lambda: [2.4, 4.5], min_length=2, max_length=2
    )
    bottom_door_tol_ratio: float = 0.15
    model_version: str = "v1.0"


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
