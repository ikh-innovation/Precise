"""Paired top-down / facade image packages from a `Precise-Data` folder.

`data/Precise-Data` stores one building per numeric id as two files:
`<id>.png` (top-down / aerial) and `Fac<id>.png` (street-level facade). A
`BuildingPackage` is that pair.

Both access patterns here hold *exactly one* package's pixels in memory at a
time, so a folder of any size costs no more RAM than its largest pair:

  * `iter_packages()` — forward-only generator; decodes a package on entry to
    the iteration and releases it before the next id is decoded. Use for batch
    passes over the whole folder.
  * `PackageCursor` — random access by position (needed for back/forward
    browsing), releasing the outgoing package as the incoming one is decoded.

Discovery is metadata-only: `discover_packages()` matches filenames and never
decodes pixels, so indexing a folder is cheap.
"""
from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PRECISE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PRECISE_ROOT / "data" / "Precise-Data"

# Extensions we will decode, in the order a pair is resolved when one numeric id
# happens to exist under more than one of them.
IMAGE_SUFFIXES: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
)

# `12.png` -> top-down for id 12;  `Fac12.png` -> facade for id 12.
# The `Fac` prefix and the extension are both matched case-insensitively, and an
# optional `_`/`-`/space separator is tolerated (`Fac_12`, `fac-12`).
_TOPDOWN_RE = re.compile(r"^(\d+)$")
_FACADE_RE = re.compile(r"^fac[_\-\s]?(\d+)$", re.IGNORECASE)


def _read_bgr(path: Path) -> np.ndarray:
    """Decode `path` to an `[H, W, 3]` uint8 BGR array.

    Grayscale is expanded to 3 channels and RGBA is flattened onto white, so a
    transparent PNG margin does not reach the pipeline as black pixels (which a
    material classifier would read as a dark wall).
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    if raw.shape[2] == 4:
        rgb = raw[:, :, :3].astype(np.float32)
        alpha = raw[:, :, 3:4].astype(np.float32) / 255.0
        return np.rint(rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return np.ascontiguousarray(raw[:, :, :3])


@dataclass
class BuildingPackage:
    """One building: a top-down image and its matching facade image.

    Constructed unloaded (paths only). Pixels are decoded on first access to
    `topdown` / `facade`, or eagerly via `load()`, and dropped by `release()`.
    Also usable as a context manager, which releases on exit.
    """

    package_id: int
    topdown_path: Path
    facade_path: Path
    _topdown: np.ndarray | None = field(default=None, repr=False, compare=False)
    _facade: np.ndarray | None = field(default=None, repr=False, compare=False)

    # ----- pixel access ------------------------------------------------------
    @property
    def topdown(self) -> np.ndarray:
        """Top-down image as BGR uint8, decoding on first access."""
        if self._topdown is None:
            self._topdown = _read_bgr(self.topdown_path)
        return self._topdown

    @property
    def facade(self) -> np.ndarray:
        """Facade image as BGR uint8, decoding on first access."""
        if self._facade is None:
            self._facade = _read_bgr(self.facade_path)
        return self._facade

    @property
    def loaded(self) -> bool:
        """True when both images are currently resident in memory."""
        return self._topdown is not None and self._facade is not None

    @property
    def nbytes(self) -> int:
        """Bytes of decoded pixels currently held (0 once released)."""
        return sum(a.nbytes for a in (self._topdown, self._facade) if a is not None)

    def load(self) -> BuildingPackage:
        """Decode both images now and return self, for chaining."""
        _, _ = self.topdown, self.facade
        return self

    def release(self) -> None:
        """Drop decoded pixels so the next package can reuse the memory."""
        self._topdown = None
        self._facade = None

    def __enter__(self) -> BuildingPackage:
        return self.load()

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    # ----- description -------------------------------------------------------
    @property
    def name(self) -> str:
        """Stable human-readable id, e.g. `package-07`."""
        return f"package-{self.package_id:02d}"

    def sizes(self) -> dict[str, tuple[int, int] | None]:
        """`(width, height)` per view for whatever is loaded, else None."""
        out: dict[str, tuple[int, int] | None] = {}
        for view, arr in (("topdown", self._topdown), ("facade", self._facade)):
            out[view] = (arr.shape[1], arr.shape[0]) if arr is not None else None
        return out

    def __repr__(self) -> str:
        state = f"{self.nbytes / 1e6:.1f} MB" if self.loaded else "released"
        return f"BuildingPackage(id={self.package_id}, {state})"


@dataclass(frozen=True)
class PackageIndex:
    """The result of scanning a folder: paired packages plus what did not pair."""

    root: Path
    packages: tuple[BuildingPackage, ...]
    unpaired_topdown: tuple[int, ...] = ()
    unpaired_facade: tuple[int, ...] = ()

    def __len__(self) -> int:
        return len(self.packages)

    @property
    def ids(self) -> tuple[int, ...]:
        """Package ids, in iteration order."""
        return tuple(p.package_id for p in self.packages)

    def summary(self) -> str:
        """One-line description of the scan, including unpaired ids."""
        parts = [f"{len(self.packages)} package(s) in {self.root}"]
        if self.unpaired_topdown:
            parts.append(f"top-down with no facade: {list(self.unpaired_topdown)}")
        if self.unpaired_facade:
            parts.append(f"facade with no top-down: {list(self.unpaired_facade)}")
        return " | ".join(parts)


def _collect(root: Path) -> tuple[dict[int, Path], dict[int, Path]]:
    """Map id -> path for top-down and facade files in `root`.

    When an id exists under several extensions the earliest entry in
    `IMAGE_SUFFIXES` wins, so the choice is deterministic rather than
    filesystem-order dependent.
    """
    rank = {suffix: i for i, suffix in enumerate(IMAGE_SUFFIXES)}
    topdown: dict[int, Path] = {}
    facade: dict[int, Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in rank:
            continue
        stem = path.stem
        if (match := _FACADE_RE.match(stem)) is not None:
            target = facade
        elif (match := _TOPDOWN_RE.match(stem)) is not None:
            target = topdown
        else:
            continue
        key = int(match.group(1))
        current = target.get(key)
        if current is None or rank[suffix] < rank[current.suffix.lower()]:
            target[key] = path
    return topdown, facade


def discover_packages(root: str | Path | None = None) -> PackageIndex:
    """Scan `root` for `<id>` / `Fac<id>` image pairs, without decoding pixels.

    Args:
        root: Folder to scan. Defaults to `<repo>/data/Precise-Data`.

    Returns:
        A `PackageIndex` whose packages are sorted by numeric id and are all
        unloaded; unmatched ids are reported separately rather than dropped
        silently.

    Raises:
        FileNotFoundError: `root` does not exist or is not a directory.
    """
    root_path = Path(root) if root is not None else DEFAULT_ROOT
    if not root_path.is_dir():
        raise FileNotFoundError(f"Package folder not found: {root_path}")

    topdown, facade = _collect(root_path)
    paired = sorted(set(topdown) & set(facade))
    packages = tuple(
        BuildingPackage(
            package_id=pid, topdown_path=topdown[pid], facade_path=facade[pid]
        )
        for pid in paired
    )
    return PackageIndex(
        root=root_path,
        packages=packages,
        unpaired_topdown=tuple(sorted(set(topdown) - set(facade))),
        unpaired_facade=tuple(sorted(set(facade) - set(topdown))),
    )


def select(
    index: PackageIndex, ids: Sequence[int] | None = None
) -> tuple[BuildingPackage, ...]:
    """Return the index packages, optionally restricted to `ids` (in `ids` order).

    Raises:
        KeyError: One of `ids` has no paired package.
    """
    if ids is None:
        return index.packages
    by_id = {p.package_id: p for p in index.packages}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(
            f"No paired package for id(s) {missing}; available: {list(index.ids)}"
        )
    return tuple(by_id[i] for i in ids)


def iter_loaded(packages: Sequence[BuildingPackage]) -> Iterator[BuildingPackage]:
    """Yield each of `packages` loaded, releasing it before the next is decoded.

    The package handed to the consumer is fully decoded; it is released as soon
    as the consumer asks for the following one (and on early exit or an
    exception), so peak memory stays at a single pair. Keep pixels past the
    iteration only by copying them.

    Args:
        packages: Packages to walk, in the order given.

    Yields:
        One loaded package at a time.
    """
    for package in packages:
        try:
            yield package.load()
        finally:
            package.release()


def iter_packages(
    root: str | Path | None = None,
    ids: Sequence[int] | None = None,
    index: PackageIndex | None = None,
) -> Iterator[BuildingPackage]:
    """Scan a folder and iterate its packages, one resident at a time.

    Convenience wrapper over `discover_packages()` + `iter_loaded()`.

    Args:
        root: Folder to scan; defaults to `<repo>/data/Precise-Data`. Ignored
            when `index` is given.
        ids: Optional subset of package ids, iterated in the given order.
        index: A pre-built `PackageIndex`, to avoid re-scanning the folder.

    Yields:
        Loaded packages, in ascending id order (or `ids` order).
    """
    idx = index if index is not None else discover_packages(root)
    yield from iter_loaded(select(idx, ids))


class PackageCursor:
    """A movable position over a package list, one package resident at a time.

    `iter_packages()` only moves forward; interactive browsing needs to step
    back too, so the cursor keeps the (cheap, unloaded) list and decodes on
    demand — releasing the outgoing package whenever it moves.
    """

    def __init__(self, packages: Sequence[BuildingPackage], start: int = 0) -> None:
        """Bind a non-empty package list and load the package at `start`.

        Raises:
            ValueError: `packages` is empty.
        """
        if not packages:
            raise ValueError("PackageCursor needs at least one package.")
        self._packages = tuple(packages)
        self._pos = start % len(self._packages)
        self._packages[self._pos].load()

    def __len__(self) -> int:
        return len(self._packages)

    @property
    def position(self) -> int:
        """Zero-based index of the current package."""
        return self._pos

    @property
    def current(self) -> BuildingPackage:
        """The loaded package at the cursor."""
        return self._packages[self._pos].load()

    def goto(self, position: int) -> BuildingPackage:
        """Move to `position` (wrapping) and return the newly loaded package."""
        target = position % len(self._packages)
        if target != self._pos:
            self._packages[self._pos].release()
            self._pos = target
        return self.current

    def move(self, delta: int) -> BuildingPackage:
        """Step `delta` packages (negative goes back, wrapping at both ends)."""
        return self.goto(self._pos + delta)

    def close(self) -> None:
        """Release the resident package."""
        self._packages[self._pos].release()

    def __enter__(self) -> PackageCursor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
