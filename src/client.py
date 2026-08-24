"""End-to-end Precise pipeline.

Runs Modules 1–3 on a single image:
  * Module 1 — semantic masks for `house`/`facade`, `window`, `door`
               (SegFormer when manual_seg=True, else SEEM).
  * Module 2 — floor count + building height from those bboxes.
  * Module 3 — dominant material on the building wall, read STRICTLY from the
               red facade/house region (config `material_backend`:
               "facade" DINOv3 Facade-8 | "minc" classifier | "siglip2" zero-shot).

Outputs next to the input image (overwritten each run):
  * `m1.json`, `m2.json`, `m3.json` — pydantic dumps from each module.
  * `<stem>_pipeline.png`           — annotated visualization.

Usage (run with the GPU conda env — see README):
    PY=~/miniconda3/envs/precise-seem-gpu/bin/python
    $PY src/client.py                      # every ./base/*.jpg
    $PY src/client.py path/to/facade.jpg   # a single image
    $PY src/client.py path/to/dir          # every .jpg in a directory

The Module 3 backend is selected by `building_materials.material_backend` in
config.yaml (default: the DINOv3 Facade-8 model).
"""
from __future__ import annotations

import os
import sys
import time
from functools import wraps
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from building_features import BuildingFeatures, Detection
from building_materials import BuildingMaterials, MaterialProperties, ObjectMaterial
from config import PreciseConfig, load_config
from facade_parsing import (
    BUILDING_LABELS,
    OPENING_LABELS,
    ClassMask,
    FacadeParser,
    ensure_backbone_assets,
)


def timer(fn):
    """Decorator that prints wall-clock seconds taken by `fn`."""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        console = Console()
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        console.print(Panel.fit(
            f"[bold]{fn.__name__}[/bold] finished in "
            f"[bold yellow]{elapsed:.2f}s[/bold yellow]",
            title="[bold]Timing[/bold]",
            border_style="magenta",
        ))
        return result
    return wrapped


class Pipeline:
    """Orchestrate Modules 1–3 and write JSON + annotated PNG outputs."""

    def __init__(self, cfg: PreciseConfig, use_segformer: bool = False) -> None:
        """Bind configuration and select the Module 1 backend.

        Args:
            cfg: Parsed `PreciseConfig` from `config.yaml`.
            use_segformer: When True, Module 1 uses the trained CMP SegFormer
                (`facade_parsing_segm/segformer_cmp`) instead of SEEM.
        """
        self.cfg = cfg
        self.use_segformer = use_segformer
        self.console = Console()

    def _bboxes_from_class(self, class_mask: ClassMask) -> list[Detection]:
        """Convert one SEEM class's polygons into per-instance xyxy bboxes."""
        out: list[Detection] = []
        score = float(class_mask.confidence)
        for poly in class_mask.polygons:
            pts = np.asarray(poly.pixel, dtype=np.float32)
            if pts.size == 0:
                continue
            x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
            x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
            out.append((np.array([x1, y1, x2, y2], dtype=np.float32), score))
        return out

    def _house_bbox(self, parse_result) -> tuple[float, float, float, float] | None:
        """Return the building bbox: `house` (SEEM) or `facade` (SegFormer)."""
        for cls in parse_result.classes:
            if cls.label in BUILDING_LABELS and cls.bbox is not None:
                x1, y1, x2, y2 = cls.bbox.pixel
                return float(x1), float(y1), float(x2), float(y2)
        return None

    def _wall_region_mask(self, parse_result, height: int, width: int) -> np.ndarray:
        """Rasterize the wall region for Module 3: facade/house minus window/door.

        Returns an `[H, W]` uint8 mask (1 = wall surface) so Module 3 classifies
        the building material on real wall pixels only. Empty when Module 1
        produced no facade/house — Module 3 then fails closed (reports "none")
        rather than classifying the whole image.
        """
        building = np.zeros((height, width), dtype=np.uint8)
        openings = np.zeros((height, width), dtype=np.uint8)
        for cls in parse_result.classes:
            if cls.label in BUILDING_LABELS:
                target = building
            elif cls.label in OPENING_LABELS:
                target = openings
            else:
                continue
            for poly in cls.polygons:
                pts = np.asarray(poly.pixel, dtype=np.int32)
                if pts.shape[0] >= 3:
                    cv2.fillPoly(target, [pts], 1)
        # Grow window/door regions by a margin so the wall mask stays off their
        # (often reflective/dark) frames and any segmentation slop.
        ratio = self.cfg.building_materials.wall_opening_dilation_ratio
        if ratio > 0 and openings.any():
            k = max(3, int(ratio * min(height, width)))
            openings = cv2.dilate(openings, np.ones((k, k), np.uint8))
        return ((building > 0) & (openings == 0)).astype(np.uint8)

    def _write_json(self, path: Path, model) -> None:
        """Dump a pydantic model to `path` as pretty JSON, overwriting."""
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")

    def _scaled_font(self, w: int, h: int) -> tuple[float, int, int]:
        """Return `(scale, thickness, line_height_px)` sized for image `w × h`."""
        scale = max(0.6, min(w, h) / 1200.0)
        thickness = max(1, int(round(scale * 2)))
        line_h = int(30 * scale) + 8
        return scale, thickness, line_h

    def _annotate_top_left(
        self, img: np.ndarray, lines: list[str]
    ) -> None:
        """Draw `lines` in a black box at the top-left of `img` (in place)."""
        if not lines:
            return
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness, line_h = self._scaled_font(w, h)
        pad = max(10, int(12 * scale))
        sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
        box_w = max(s[0] for s in sizes) + pad * 2
        box_h = line_h * len(lines) + pad * 2
        cv2.rectangle(img, (0, 0), (box_w, box_h), (0, 0, 0), -1)
        y = pad + int(line_h * 0.75)
        for line in lines:
            cv2.putText(
                img, line, (pad, y), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA,
            )
            y += line_h

    def _annotate_per_object_materials(
        self,
        img: np.ndarray,
        per_object: dict[str, list[ObjectMaterial]],
    ) -> None:
        """Label each detected mask with its material, in the class's overlay color.

        Labels go just above the bbox; if there is no room above, they are
        placed inside the bbox at the top. Text color matches the class's
        SEEM overlay color (BGR from config).
        """
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness, _ = self._scaled_font(w, h)
        pad = max(4, int(4 * scale))
        colors_bgr = {k: tuple(v) for k, v in self.cfg.facade_parsing.label_colors_bgr.items()}

        for label, entries in per_object.items():
            color = colors_bgr.get(label, (255, 255, 255))
            for obj in entries:
                x1, y1, x2, y2 = (int(round(v)) for v in obj.bbox)
                text = f"{label}: {obj.dominant_material.label}"
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                y_text = y1 - pad
                if y_text - th < 0:
                    y_text = y1 + th + pad
                x_text = max(0, min(x1, w - tw - 1))
                cv2.rectangle(
                    img,
                    (x_text - 2, y_text - th - 2),
                    (x_text + tw + 2, y_text + 2),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    img, text, (x_text, y_text), font, scale,
                    color, thickness, cv2.LINE_AA,
                )

    def _save_annotated(
        self,
        overlay: np.ndarray,
        text_lines: list[str],
        per_object: dict[str, list[ObjectMaterial]] | None,
        out_path: Path,
    ) -> Path:
        """Compose the final PNG: SEEM overlay array + top-left text + (optional) per-object labels."""
        img = overlay.copy()
        if per_object is not None:
            self._annotate_per_object_materials(img, per_object)
        self._annotate_top_left(img, text_lines)
        out_path = out_path.with_suffix(".png")
        cv2.imwrite(str(out_path), img)
        return out_path

    def _print_material_properties(
        self, props_by_material: dict[str, MaterialProperties]
    ) -> None:
        """Print fixed mechanical properties for the detected material(s).

        Renders one column per distinct material and one row per property.
        Does nothing when the mapping is empty (e.g. labels that have no
        `material_properties` entry in the config).

        Args:
            props_by_material: Material label → its configured properties.
        """
        if not props_by_material:
            return
        table = Table(
            title="[bold]Mechanical properties[/bold] "
            "[dim](fixed reference values from config.yaml)[/dim]",
            title_justify="left",
            border_style="blue",
            header_style="bold cyan",
        )
        table.add_column("Property", style="cyan", no_wrap=True)
        for label in props_by_material:
            table.add_column(label, justify="right")

        rows = [
            ("Category", lambda p: p.category),
            ("Density", lambda p: f"{p.density_kg_m3:g} kg/m³"),
            ("Young's modulus E", lambda p: f"{p.youngs_modulus_gpa:g} GPa"),
            ("Compressive strength", lambda p: f"{p.compressive_strength_mpa:g} MPa"),
            ("Tensile strength", lambda p: f"{p.tensile_strength_mpa:g} MPa"),
            ("Poisson's ratio ν", lambda p: f"{p.poisson_ratio:g}"),
            ("Representative of", lambda p: p.note),
        ]
        props = list(props_by_material.values())
        for name, getter in rows:
            table.add_row(name, *(str(getter(p)) for p in props))
        self.console.print(table)

    @timer
    def run(self, image: str | Path) -> Path:
        """Run Modules 1–3 on `image` and write JSON + annotated PNG outputs.

        Args:
            image: Path to the input facade image.

        Returns:
            Path to the annotated `<stem>_pipeline.png`.
        """
        image_path = Path(image)
        m1 = "SegFormer" if self.use_segformer else "SEEM"
        m3 = {"minc": "MINC", "facade": "DINOv3 Facade-8"}.get(
            self.cfg.building_materials.material_backend, "SigLIP2"
        )
        self.console.print(Panel.fit(
            "[bold cyan]Precise[/bold cyan] · Facade Pipeline\n"
            f"[dim]Module 1 {m1}  →  Module 2 geometry  +  Module 3 {m3}[/dim]",
            border_style="cyan",
        ))

        # Module 1 backend: trained CMP SegFormer (manual_seg=True) or SEEM.
        # Both expose `.parse()` -> FacadeParsingResult and
        # `.last_visualization_image`, and emit house/window/door.
        if self.use_segformer:
            with self.console.status(
                "[bold cyan]Module 1: SegFormer (trained CMP model)...", spinner="dots"
            ):
                from facade_parsing_segm.segformer_cmp.pipeline_adapter import (
                    FacadeSegformerParser,
                )

                parser = FacadeSegformerParser(
                    image_path=image_path,
                    cfg=self.cfg.facade_parsing_segformer,
                    colors_bgr=self.cfg.facade_parsing.label_colors_bgr,
                    view_type=self.cfg.pipeline.view_type,
                )
                parse_result = parser.parse()
        else:
            ensure_backbone_assets(self.cfg.facade_parsing.backbone)
            with self.console.status(
                "[bold cyan]Module 1: SEEM semantic parsing...", spinner="dots"
            ):
                parser = FacadeParser(
                    image_path=image_path,
                    cfg=self.cfg.facade_parsing,
                    view_type=self.cfg.pipeline.view_type,
                )
                parse_result = parser.parse()

        overlay = parser.last_visualization_image
        if overlay is None:
            raise RuntimeError("Module 1 did not produce an overlay image.")

        per_class = {c.label: self._bboxes_from_class(c) for c in parse_result.classes}
        windows = per_class.get("window", [])
        doors = per_class.get("door", [])
        house_bbox = self._house_bbox(parse_result)

        with self.console.status(
            "[bold cyan]Module 2: floor count & building height...", spinner="dots"
        ):
            features = BuildingFeatures(
                image_path=image_path,
                cfg=self.cfg.building_features,
                view_type=self.cfg.pipeline.view_type,
            ).extract(windows=windows, doors=doors, house_bbox=house_bbox)

        # Module 3 backend: SigLIP2 zero-shot (default), the custom MINC timm
        # classifier, or the DINOv3 Facade-8 classifier. All expose the same
        # classify(region_mask) / classify_instances surface.
        backend = self.cfg.building_materials.material_backend
        if backend == "minc":
            from building_materials_minc import MincMaterials

            materials = MincMaterials(
                image_path=image_path,
                cfg=self.cfg.building_materials,
                view_type=self.cfg.pipeline.view_type,
            )
        elif backend == "facade":
            from building_materials_facade import FacadeMaterials

            materials = FacadeMaterials(
                image_path=image_path,
                cfg=self.cfg.building_materials,
                view_type=self.cfg.pipeline.view_type,
            )
        else:
            materials = BuildingMaterials(
                image_path=image_path,
                cfg=self.cfg.building_materials,
                view_type=self.cfg.pipeline.view_type,
            )

        materials_all = self.cfg.pipeline.materials_all
        if materials_all:
            # Classify the building wall only: facade/house region minus
            # window/door openings (no GrabCut). Empty mask -> Module 3 fails
            # closed ("none"); it never classifies the whole image.
            wall_mask = self._wall_region_mask(
                parse_result, parse_result.image.height, parse_result.image.width
            )
            with self.console.status(
                "[bold cyan]Module 3: material classification (building wall)...",
                spinner="dots",
            ):
                m3_result = materials.classify(region_mask=wall_mask)
            per_object_labels = None
        else:
            with self.console.status(
                "[bold cyan]Module 3: per-object material classification...",
                spinner="dots",
            ):
                detections = {
                    label: [tuple(b.tolist()) for b, _ in dets]
                    for label, dets in per_class.items()
                }
                m3_result = materials.classify_instances(detections)
                per_object_labels = m3_result.objects

        json_dir = image_path.parent
        self._write_json(json_dir / "m1.json", parse_result)
        self._write_json(json_dir / "m2.json", features)
        self._write_json(json_dir / "m3.json", m3_result)

        floors = features.predictions.floor_count.value
        height_m = features.predictions.building_height_m.value
        height_src = features.height_source

        lines = [
            f"Floors:   {floors}",
            f"Height:   {height_m:.1f} m  ({height_src})",
        ]
        if materials_all:
            if m3_result.classified_region == "none":
                lines.append("Material: n/a (no building region)")
            else:
                dom = m3_result.dominant_material
                # Flag the degraded read so a masked-bbox result isn't mistaken
                # for a confident multi-patch wall classification.
                approx = " ~approx" if m3_result.classified_region == "masked_bbox" else ""
                lines.append(f"Material: {dom.label} ({dom.score * 100:.1f}%){approx}")

        out_path = image_path.with_name(f"{image_path.stem}_pipeline.png")
        final = self._save_annotated(overlay, lines, per_object_labels, out_path)

        summary = [
            f"[dim]Windows:[/dim] {len(windows)}   [dim]Doors:[/dim] {len(doors)}",
            *lines,
            f"[dim]Output PNG:[/dim] {final}",
            f"[dim]JSON outputs:[/dim] m1.json, m2.json, m3.json (in {json_dir})",
        ]
        self.console.print(Panel.fit(
            "\n".join(summary),
            title="[bold]Pipeline result[/bold]",
            border_style="green",
        ))

        # Module 3 add-on: print fixed mechanical properties of whatever
        # material(s) were detected (one column each, distinct labels only).
        if materials_all:
            props = m3_result.dominant_material_properties
            if props is not None:
                self._print_material_properties(
                    {m3_result.dominant_material.label: props}
                )
        else:
            props_by_material: dict[str, MaterialProperties] = {}
            for entries in m3_result.objects.values():
                for obj in entries:
                    if obj.properties is not None:
                        props_by_material.setdefault(
                            obj.dominant_material.label, obj.properties
                        )
            self._print_material_properties(props_by_material)
        return final


def main(img_path: str, materials_all=True, manual_seg: bool = True) -> None:
    """Entry point: run the pipeline on a single image.

    Args:
        img_path: Path to the input facade image.
        materials_all: Whole-image material classification (True) vs. per-object.
        manual_seg: Module 1 backend.
            True  -> the trained CMP SegFormer (segformer_cmp/runs/best).
            False -> SEEM (zero-shot).
    """

    cfg = load_config()
    cfg.pipeline.materials_all = materials_all
    Pipeline(cfg, use_segformer=manual_seg).run(image=img_path)


if __name__ == "__main__":
    # Run a single image, every .jpg in a directory, or (no arg) all of ./base.
    target = sys.argv[1] if len(sys.argv) > 1 else "./base"
    if os.path.isdir(target):
        images = [
            os.path.join(target, f)
            for f in sorted(os.listdir(target))
            if f.lower().endswith(".jpg")
        ]
    else:
        images = [target]

    if not images:
        raise SystemExit(f"No images to process at: {target}")
    for img in images:
        main(img)
