"""Side-by-side display of a `BuildingPackage` in an OpenCV window.

The window shows **only the two images** — no titles, captions, filenames,
counters or key legend. Which package is on screen, and what the pipeline made
of it, is reported on the terminal instead.

The two views of a package have unrelated resolutions — in `data/Precise-Data`
the top-down crops run from 107x127 up to 1528x682 while the facades are mostly
tall — so each view is letterboxed into its own fixed cell instead of being
concatenated raw. That keeps the window a stable size as the iteration moves,
which matters when `cv2.imshow` reuses one window across packages.

`compose_package_view()` is pure (array in, array out); `PackageWindow` adds the
interactive window and key handling on top of it. Key reads go through
`cv2.waitKeyEx`, not `cv2.waitKey`, because the latter is defined as
`waitKeyEx(delay) & 0xff` and so collapses every arrow key to 0.
"""
from __future__ import annotations

import cv2
import numpy as np

from .loader import BuildingPackage

WINDOW_NAME = "Precise"

# Dark ground so letterbox padding reads as background rather than as image.
_BG = (24, 24, 24)

# Keys the interactive loop understands. Nothing is drawn on the canvas any
# more, so callers print this on the terminal instead of rendering a legend.
KEY_HELP: tuple[tuple[str, str], ...] = (
    ("d / right", "next package"),
    ("a / left", "previous package"),
    ("r", "run pipeline on this facade"),
    ("q / esc", "quit"),
)

# For display-only browsing, which does not honour `r`.
KEY_HELP_VIEW_ONLY: tuple[tuple[str, str], ...] = tuple(
    row for row in KEY_HELP if row[0] != "r"
)


def _fit_into(img: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
    """Letterbox `img` into a `cell_h x cell_w` panel, preserving aspect ratio.

    Upscales small images (several top-down crops are barely 100 px wide) but
    never past 3x, beyond which the interpolation shows more than the photo.
    """
    h, w = img.shape[:2]
    scale = min(cell_w / w, cell_h / h)
    scale = min(scale, 3.0)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    cell = np.full((cell_h, cell_w, 3), _BG, dtype=np.uint8)
    y0 = (cell_h - new_h) // 2
    x0 = (cell_w - new_w) // 2
    cell[y0:y0 + new_h, x0:x0 + new_w] = resized
    return cell


def compose_package_view(
    package: BuildingPackage,
    cell_size: tuple[int, int] = (620, 620),
    gap: int = 14,
) -> np.ndarray:
    """Render `package` as a single BGR canvas: top-down beside facade.

    Images only — nothing is drawn over or around them beyond the neutral
    padding needed to make two different aspect ratios share a row.

    Args:
        package: The package to draw. Accessing its pixels decodes them if the
            package is not already loaded.
        cell_size: `(width, height)` of each image panel in pixels.
        gap: Pixels of background between and around the panels.

    Returns:
        An `[H, W, 3]` uint8 BGR canvas ready for `cv2.imshow`.
    """
    cell_w, cell_h = cell_size
    canvas_w = cell_w * 2 + gap * 3
    canvas_h = cell_h + gap * 2
    canvas = np.full((canvas_h, canvas_w, 3), _BG, dtype=np.uint8)

    for i, img in enumerate((package.topdown, package.facade)):
        x0 = gap + i * (cell_w + gap)
        canvas[gap:gap + cell_h, x0:x0 + cell_w] = _fit_into(img, cell_w, cell_h)
    return canvas


class PackageWindow:
    """One reused OpenCV window plus the key decoding for interactive browsing."""

    # Arrow codes are the extended values a Win32 build reports; the Qt/GTK
    # values (81/83) are deliberately NOT accepted because they collide with
    # ord("Q") and ord("S"). Reading them at all requires `waitKeyEx` — plain
    # `waitKey` is defined as `waitKeyEx(delay) & 0xff`, which collapses every
    # arrow to 0. An unmatched code is reported, not swallowed.
    _NEXT = {ord("d"), ord("D"), 2555904}
    _PREV = {ord("a"), ord("A"), 2424832}
    _RUN = {ord("r"), ord("R"), 13, 32}
    _QUIT = {ord("q"), ord("Q"), 27}

    def __init__(self, window_name: str = WINDOW_NAME) -> None:
        self.window_name = window_name
        self._opened = False
        # Raw code of the last unrecognized press, for the caller to surface.
        self.last_unknown_key: int | None = None

    def _ensure_open(self) -> None:
        if not self._opened:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self._opened = True

    @classmethod
    def is_quit(cls, key: int) -> bool:
        """True when `key` is a quit press, for callers driving `show()` directly."""
        return key in cls._QUIT

    def show(self, canvas: np.ndarray, delay_ms: int = 1) -> int:
        """Display `canvas` and pump the GUI event loop for `delay_ms`.

        Returns:
            The raw `cv2.waitKeyEx` code (-1 when nothing was pressed).
        """
        self._ensure_open()
        cv2.imshow(self.window_name, canvas)
        return cv2.waitKeyEx(delay_ms)

    def wait_for_action(self, canvas: np.ndarray) -> str:
        """Display `canvas` and block until the user presses a recognized key.

        The arrow-key codes `cv2.waitKeyEx` reports are backend-specific, so an
        unrecognized press returns `unknown` with the raw code in
        `last_unknown_key` rather than being swallowed — otherwise a build that
        emits a different code would just look like a dead keyboard.

        Returns:
            One of `next`, `prev`, `run`, `quit`, `unknown`. Closing the window
            with its title-bar button also yields `quit`.
        """
        self._ensure_open()
        cv2.imshow(self.window_name, canvas)
        while True:
            key = cv2.waitKeyEx(50)
            if key in self._QUIT:
                return "quit"
            if key in self._NEXT:
                return "next"
            if key in self._PREV:
                return "prev"
            if key in self._RUN:
                return "run"
            if key != -1:
                self.last_unknown_key = key
                return "unknown"
            # The window was closed from its title bar: treat as quit so the
            # loop cannot spin forever on a dead window.
            if not self._window_alive():
                return "quit"

    def _window_alive(self) -> bool:
        """True while the window still exists (title-bar close sets it to 0)."""
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def close(self) -> None:
        """Destroy the window if it was opened."""
        if self._opened:
            cv2.destroyWindow(self.window_name)
            # A few event-loop pumps are needed for the window to actually go away.
            for _ in range(3):
                cv2.waitKeyEx(1)
            self._opened = False

    def __enter__(self) -> PackageWindow:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
