"""Paired top-down / facade image packages: iterate, display, and analyse.

`data/Precise-Data` holds one building per numeric id as `<id>.png` (top-down)
and `Fac<id>.png` (facade). This package turns that folder into an iteration of
`BuildingPackage` pairs that each load into memory one at a time, shows the pair
side by side in a window, and feeds the facade half to Modules 1-3.

    from building_packages import discover_packages, iter_packages
    from building_packages import compose_package_view, PackageWindow

    index = discover_packages()                 # filenames only, no pixels
    for package in iter_packages(index=index):  # one pair resident at a time
        canvas = compose_package_view(package)   # just the two images, no chrome

Run it from the CLI with `python src/client.py --packages` (see `client.py` for
the interactive keys and the `--auto` / `--show` modes).

The window draws only the two images; `client.py` reports package names and
pipeline results on the terminal.
"""
from .loader import (
    DEFAULT_ROOT,
    IMAGE_SUFFIXES,
    BuildingPackage,
    PackageCursor,
    PackageIndex,
    discover_packages,
    iter_loaded,
    iter_packages,
    select,
)
from .runner import (
    DEFAULT_OUT_ROOT,
    PackageRun,
    package_out_dir,
    pipeline_blockers,
    run_package,
)
from .viewer import (
    KEY_HELP,
    KEY_HELP_VIEW_ONLY,
    WINDOW_NAME,
    PackageWindow,
    compose_package_view,
)

__all__ = [
    # loader
    "BuildingPackage",
    "PackageCursor",
    "PackageIndex",
    "discover_packages",
    "iter_loaded",
    "iter_packages",
    "select",
    "DEFAULT_ROOT",
    "IMAGE_SUFFIXES",
    # viewer
    "compose_package_view",
    "PackageWindow",
    "WINDOW_NAME",
    "KEY_HELP",
    "KEY_HELP_VIEW_ONLY",
    # runner
    "run_package",
    "package_out_dir",
    "pipeline_blockers",
    "PackageRun",
    "DEFAULT_OUT_ROOT",
]
