"""Feed a `BuildingPackage` into the Modules 1-3 pipeline.

Only the **facade** image is analysed. Module 1 is the CMP-trained SegFormer
(facade / window / door) or SEEM, and Module 2 derives floors and height from
window/door rows — neither is meaningful on an aerial crop, so the top-down
image of a package is carried for display and context only, never pushed
through the facade models.

`run_package()` takes the pipeline as an argument rather than importing it:
`client.py` owns the CLI and imports this module, so importing `client` back
from here would be circular. Anything exposing
`run(image=..., out_dir=...) -> path` satisfies it.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .loader import PRECISE_ROOT, BuildingPackage

# Written next to the data by default, one subdirectory per package, so the 20
# pairs in a single folder do not overwrite each other's m1/m2/m3.json.
DEFAULT_OUT_ROOT = PRECISE_ROOT / "data" / "package_runs"


class SupportsRun(Protocol):
    """The slice of `client.Pipeline` that `run_package()` depends on."""

    def run(self, image: str | Path, out_dir: str | Path | None = ...) -> Path:
        ...


@dataclass(frozen=True)
class PackageRun:
    """Outcome of one pipeline invocation on one package."""

    package_id: int
    out_dir: Path
    annotated: Path | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the pipeline completed and wrote its annotated PNG."""
        return self.error is None and self.annotated is not None

    def summary(self) -> str:
        """One-line result, for the terminal."""
        if self.ok:
            return f"package-{self.package_id:02d}: OK -> {self.out_dir}"
        return f"package-{self.package_id:02d}: FAILED - {self.error}"


def _missing_modules(names: tuple[str, ...]) -> list[str]:
    """Return which of `names` cannot be imported, without importing them."""
    missing = []
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


def pipeline_blockers(cfg, use_segformer: bool = True) -> list[str]:
    """List reasons the pipeline cannot run here, cheaply and before any run.

    Checks the Python packages the pipeline imports and the trained checkpoints
    the selected Module 1 / Module 3 backends load. Returning these strings
    lets the CLI warn once at startup instead of raising on the first package.

    Args:
        cfg: A loaded `PreciseConfig`.
        use_segformer: Whether Module 1 will be the trained CMP SegFormer.

    Returns:
        Human-readable blockers; empty when the pipeline looks runnable.
    """
    blockers: list[str] = []

    missing = _missing_modules(("rich", "cv2", "numpy", "torch"))
    if missing:
        blockers.append(f"missing Python package(s): {', '.join(missing)}")

    if use_segformer:
        ckpt = Path(cfg.facade_parsing_segformer.checkpoint)
        if not ckpt.is_absolute():
            ckpt = PRECISE_ROOT / ckpt
        if not ckpt.exists():
            blockers.append(f"Module 1 SegFormer checkpoint not found: {ckpt}")
        if _missing_modules(("transformers",)):
            blockers.append("Module 1 SegFormer needs `transformers`")

    backend = cfg.building_materials.material_backend
    if backend == "facade":
        ckpt = Path(cfg.building_materials.facade_checkpoint)
        if not ckpt.is_absolute():
            ckpt = PRECISE_ROOT / ckpt
        if not ckpt.exists():
            blockers.append(f"Module 3 facade checkpoint not found: {ckpt}")
        if _missing_modules(("timm",)):
            blockers.append("Module 3 facade backend needs `timm`")
    elif backend == "minc":
        ckpt = Path(cfg.building_materials.minc_checkpoint)
        if not ckpt.is_absolute():
            ckpt = PRECISE_ROOT / ckpt
        if not ckpt.exists():
            blockers.append(f"Module 3 MINC checkpoint not found: {ckpt}")
        if _missing_modules(("timm",)):
            blockers.append("Module 3 MINC backend needs `timm`")
    elif _missing_modules(("open_clip",)):
        blockers.append("Module 3 SigLIP2 backend needs `open_clip_torch`")

    return blockers


def package_out_dir(package: BuildingPackage, out_root: str | Path | None = None) -> Path:
    """Directory this package's pipeline outputs belong in."""
    root = Path(out_root) if out_root is not None else DEFAULT_OUT_ROOT
    return root / package.name


def run_package(
    package: BuildingPackage,
    pipeline: SupportsRun,
    out_root: str | Path | None = None,
) -> PackageRun:
    """Run `pipeline` on this package's facade image.

    The pipeline reads the facade from disk (its modules are path-driven), and
    writes `m1.json`, `m2.json`, `m3.json` and `Fac<id>_pipeline.png` into a
    per-package directory. Inference failures are captured in the returned
    `PackageRun` rather than raised, so a batch pass over a folder is not ended
    by one bad image.

    Args:
        package: The package whose facade to analyse.
        pipeline: A configured pipeline exposing `run(image=, out_dir=)`.
        out_root: Parent for the per-package output directory. Defaults to
            `<repo>/data/package_runs`.

    Returns:
        A `PackageRun` describing where the outputs went, or the error text.
    """
    out_dir = package_out_dir(package, out_root)
    try:
        annotated = pipeline.run(image=package.facade_path, out_dir=out_dir)
        return PackageRun(package.package_id, out_dir, annotated=Path(annotated))
    except Exception as exc:  # noqa: BLE001 - reported per package, never fatal
        return PackageRun(
            package.package_id, out_dir, error=f"{type(exc).__name__}: {exc}"
        )
