"""End-to-end Precise pipeline.

Runs Modules 1–3 on a folder of paired top-down/facade **packages** (the
default), or on plain facade images with `--images`.

On a single image:
  * Module 1 — semantic masks for `house`/`facade`, `window`, `door`
               (SegFormer when manual_seg=True, else SEEM).
  * Module 2 — floor count + building height from those bboxes.
  * Module 3 — dominant material on the building wall, read STRICTLY from the
               red facade/house region (config `material_backend`:
               "facade" DINOv3 Facade-8 | "minc" classifier | "siglip2" zero-shot).

Outputs next to the input image (overwritten each run):
  * `m1.json`, `m2.json`, `m3.json` — pydantic dumps from each module.
  * `<stem>_pipeline.png`           — annotated visualization.

**Package mode is the default.** It walks a folder where each building is a
pair of images — `<id>.png` top-down and `Fac<id>.png` facade — showing both in
one window while feeding the facade half to Modules 1–3. Only one pair is
decoded at a time. The window shows nothing but the two images; package names,
results and the key reference are printed here on the terminal.

Usage (run with the GPU conda env — see README):
    PY=~/miniconda3/envs/precise-seem-gpu/bin/python
    $PY src/client.py                      # browse data/Precise-Data  (default)
    $PY src/client.py path/to/pairs        # a different package folder
    $PY src/client.py --auto               # batch every pair
    $PY src/client.py --show               # display only, no models loaded
    $PY src/client.py --ids 3 7            # only these ids

    keys:  d / right  next     a / left  previous
           r          run Modules 1–3    q / esc  quit

Image mode needs `--images`, except that a positional *file* selects it
automatically (a file can never be a folder of pairs):

    $PY src/client.py path/to/facade.jpg   # a single image (no flag needed)
    $PY src/client.py --images             # every ./base/*.jpg
    $PY src/client.py --images path/to/dir # every .jpg in a directory

The Module 3 backend is selected by `building_materials.material_backend` in
config.yaml (default: the DINOv3 Facade-8 model).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from building_packages import (
    DEFAULT_OUT_ROOT,
    DEFAULT_ROOT,
    KEY_HELP,
    KEY_HELP_VIEW_ONLY,
    PackageCursor,
    PackageWindow,
    compose_package_view,
    discover_packages,
    iter_loaded,
    pipeline_blockers,
    run_package,
    select,
)
from config import PreciseConfig, load_config

# Modules 1-3 pull in torch, open_clip, transformers and rich at import time.
# They are imported inside the methods that use them so that package browsing
# (`--packages --show`) needs nothing but cv2/numpy — it loads no model and
# renders no rich output, and must stay usable in an environment that cannot
# run inference. `from __future__ import annotations` makes every annotation
# below a string, so these names are only needed by a type checker.
if TYPE_CHECKING:
    from building_features import Detection
    from building_materials import MaterialProperties, ObjectMaterial
    from facade_parsing import ClassMask


def timer(fn):
    """Decorator that prints wall-clock seconds taken by `fn`."""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        from rich.console import Console
        from rich.panel import Panel

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
        from rich.console import Console

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
        from facade_parsing import BUILDING_LABELS

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
        from facade_parsing import BUILDING_LABELS, OPENING_LABELS

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
        from rich.table import Table

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
    def run(self, image: str | Path, out_dir: str | Path | None = None) -> Path:
        """Run Modules 1–3 on `image` and write JSON + annotated PNG outputs.

        Args:
            image: Path to the input facade image.
            out_dir: Directory for `m1/m2/m3.json` and the annotated PNG,
                created if missing. Defaults to the image's own directory —
                pass a per-image directory when batching a folder, otherwise
                every run overwrites the same three JSON files.

        Returns:
            Path to the annotated `<stem>_pipeline.png`.
        """
        from rich.panel import Panel

        from building_features import BuildingFeatures

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
            from facade_parsing import FacadeParser, ensure_backbone_assets

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
            from building_materials import BuildingMaterials

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

        json_dir = Path(out_dir) if out_dir is not None else image_path.parent
        json_dir.mkdir(parents=True, exist_ok=True)
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

        out_path = json_dir / f"{image_path.stem}_pipeline.png"
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
    """Run the pipeline on a single image.

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


# ---------------------------------------------------------------------------
# Package mode: paired top-down / facade images from a `Precise-Data` folder
# ---------------------------------------------------------------------------

def _print_keys(can_run: bool) -> None:
    """List the interactive keys on the terminal.

    The window itself draws nothing but the two images, so the key reference
    lives here rather than as an on-canvas legend. Plain `print` throughout
    package mode: `rich` is a pipeline dependency, and browsing must not
    require it.
    """
    for key, action in (KEY_HELP if can_run else KEY_HELP_VIEW_ONLY):
        print(f"  {key:<10} {action}")


def _report_blockers(cfg: PreciseConfig, use_segformer: bool) -> bool:
    """Print why inference is unavailable, if it is.

    Returns:
        True when the pipeline looks runnable.
    """
    blockers = pipeline_blockers(cfg, use_segformer=use_segformer)
    if not blockers:
        return True
    print("Pipeline unavailable - display only:")
    for item in blockers:
        print(f"  - {item}")
    print("  (--show silences this)")
    return False


def _build_pipeline(
    cfg: PreciseConfig, use_segformer: bool, materials_all: bool
) -> Pipeline:
    """Configure one `Pipeline` for a whole package run.

    The caller's config is left untouched, and `view_type` is pinned to
    `facade`: only the facade half of a package is ever analysed.
    """
    run_cfg = cfg.model_copy(deep=True)
    run_cfg.pipeline.materials_all = materials_all
    run_cfg.pipeline.view_type = "facade"
    return Pipeline(run_cfg, use_segformer=use_segformer)


def _packages_interactive(packages, cfg, args, can_run: bool) -> int:
    """Browse packages in a window, running the pipeline on demand.

    Returns:
        Process exit code.
    """
    total = len(packages)
    pipeline = _build_pipeline(cfg, not args.seem, not args.per_object) if can_run else None

    with PackageCursor(packages) as cursor, PackageWindow() as window:
        while True:
            package = cursor.current
            print(f"showing {package.name}  [{cursor.position + 1}/{total}]", flush=True)
            action = window.wait_for_action(
                compose_package_view(package, cell_size=(args.cell, args.cell))
            )

            if action == "quit":
                return 0
            if action == "next":
                cursor.move(1)
            elif action == "prev":
                cursor.move(-1)
            elif action == "unknown":
                # Surfaced rather than ignored so a backend whose arrow codes
                # differ from the ones we accept is diagnosable on the spot.
                print(f"unrecognized key code {window.last_unknown_key}")
                _print_keys(can_run)
            elif action == "run":
                if pipeline is None:
                    print("pipeline unavailable in this environment")
                    continue
                result = run_package(package, pipeline, out_root=args.out_root)
                print(result.summary(), flush=True)


def _packages_auto(packages, cfg, args, can_run: bool) -> int:
    """Walk every package once, running the pipeline and reporting a tally.

    Returns:
        Process exit code — non-zero when any package failed.
    """
    total = len(packages)
    pipeline = _build_pipeline(cfg, not args.seem, not args.per_object) if can_run else None
    window = None if args.no_window else PackageWindow()
    failures = 0

    try:
        for position, package in enumerate(iter_loaded(packages)):
            print(f"[{position + 1}/{total}] {package.name}", flush=True)
            if pipeline is not None:
                result = run_package(package, pipeline, out_root=args.out_root)
                print(f"    {result.summary()}", flush=True)
                failures += 0 if result.ok else 1

            if window is not None:
                canvas = compose_package_view(package, cell_size=(args.cell, args.cell))
                # q / esc during the hold aborts the batch.
                if window.is_quit(window.show(canvas, delay_ms=max(1, args.delay))):
                    print("aborted by user")
                    break
    finally:
        if window is not None:
            window.close()

    if pipeline is not None:
        print(f"done: {total - failures}/{total} succeeded -> {args.out_root}")
    return 1 if failures else 0


def _run_packages(args: argparse.Namespace) -> int:
    """Index a package folder, then browse or batch it.

    This is the default mode: bare `python src/client.py` lands here on
    `data/Precise-Data`.
    """
    # `--packages DIR` wins over a positional folder; neither given -> the
    # default dataset. `discover_packages(None)` resolves to data/Precise-Data.
    root = args.packages or args.target
    index = discover_packages(root)
    print(index.summary())
    if not index.packages:
        print("No <id>/Fac<id> image pairs found.")
        return 1

    packages = select(index, args.ids)
    if args.list:
        for package in packages:
            print(
                f"  {package.name}  top-down={package.topdown_path.name}"
                f"  facade={package.facade_path.name}"
            )
        return 0

    cfg = load_config()
    can_run = False
    if not args.show_only:
        can_run = _report_blockers(cfg, use_segformer=not args.seem)

    if not args.auto:
        _print_keys(can_run)
        return _packages_interactive(packages, cfg, args, can_run)
    return _packages_auto(packages, cfg, args, can_run)


def _run_images(args: argparse.Namespace) -> int:
    """Run the pipeline over a single image or every .jpg in a directory."""
    target = args.target or "./base"
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
        main(img, materials_all=not args.per_object, manual_seg=not args.seem)
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build and parse the command line."""
    parser = argparse.ArgumentParser(
        prog="client.py",
        description=(
            "Run Modules 1-3 on a folder of paired top-down/facade packages "
            "(the default), or on plain facade images with --images."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "target", nargs="?", default=None,
        help="Package folder to iterate (default: data/Precise-Data). An "
             "existing image FILE switches to image mode automatically; with "
             "--images this is the image or directory of .jpg files instead.",
    )
    parser.add_argument(
        "--images", action="store_true",
        help="Image mode: treat the target as a facade image or a directory "
             "of .jpg files (default: ./base) rather than a package folder.",
    )

    group = parser.add_argument_group("package mode (default)")
    group.add_argument(
        "--packages", nargs="?", const=str(DEFAULT_ROOT), default=None,
        metavar="DIR",
        help="Explicitly select package mode, optionally naming the folder of "
             "<id>/Fac<id> pairs. Redundant now that it is the default, but "
             "takes precedence over the positional target.",
    )
    group.add_argument(
        "--ids", type=int, nargs="+", default=None,
        help="Only these package ids, in this order.",
    )
    group.add_argument(
        "--auto", action="store_true",
        help="Batch every package unattended instead of waiting on keys.",
    )
    group.add_argument(
        "--show", "--no-run", dest="show_only", action="store_true",
        help="Display only: iterate and show the pairs, never invoke the "
             "pipeline (and skip its availability check). With --auto this is "
             "an unattended slideshow.",
    )
    group.add_argument(
        "--list", action="store_true",
        help="Print the discovered packages and exit.",
    )
    group.add_argument(
        "--no-window", action="store_true",
        help="Do not open a window (batch passes on a headless machine).",
    )
    group.add_argument(
        "--out-root", default=str(DEFAULT_OUT_ROOT),
        help="Parent directory for per-package pipeline outputs.",
    )
    group.add_argument(
        "--cell", type=int, default=620,
        help="Pixel size of each image panel in the window.",
    )
    group.add_argument(
        "--delay", type=int, default=400,
        help="Milliseconds to hold each package on screen in --auto mode.",
    )

    common = parser.add_argument_group("backends (both modes)")
    common.add_argument(
        "--seem", action="store_true",
        help="Module 1 backend: zero-shot SEEM instead of the CMP SegFormer.",
    )
    common.add_argument(
        "--per-object", action="store_true",
        help="Module 3 classifies each detected object, not the whole wall.",
    )
    return parser.parse_args(argv)


def cli(argv: list[str] | None = None) -> int:
    """Entry point. Package mode is the default; `--images` selects image mode.

    A positional target that is an existing *file* also selects image mode: a
    file can never be a folder of pairs, so `client.py photo.jpg` keeps working
    without the flag. A directory is read as a package folder, since that is
    now the default — pass `--images DIR` for the old "every .jpg in DIR" pass.
    """
    args = _parse_args(argv)
    if args.images or (args.target is not None and Path(args.target).is_file()):
        return _run_images(args)
    return _run_packages(args)


if __name__ == "__main__":
    raise SystemExit(cli())
