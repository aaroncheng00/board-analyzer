"""
Building blocks for synthesising realistic board-cell training images.

Use to fix alpha bug where transparent backgrounds are automatically mapped to black.
Overlay sprites onto real square colours instead.
"""

import colorsys
import random
from collections import Counter

import cv2
import numpy as np

try:
    from board_slicer import extract_board_cells
    from slicer_config import SlicerConfig
except ImportError:  # imported as part of the `src` package
    from .board_slicer import extract_board_cells
    from .slicer_config import SlicerConfig


# ---------------------------------------------------------------------------
# Some hardcoded square colours, as BGR triples (cv2 order, matching board_slicer's output).
#
# Fallback themes for boards we have no screenshot of.
# ---------------------------------------------------------------------------
SQUARE_PALETTE = {
    "lichess_light":   (181, 217, 240),   # #F0D9B5
    "lichess_dark":    (99, 136, 181),    # #B58863
    "blue_light":      (222, 227, 234),   # #EAE3DE
    "blue_dark":       (160, 130, 111),   # #6F82A0
    "grey_light":      (220, 220, 220),
    "grey_dark":       (130, 130, 130),
    "wood_light":      (151, 183, 216),   # the tan the *?l45 sprites already use
    "wood_dark":       (118, 150, 182),
}


def sample_square_colors(image_path, config=None, empty_rows=None):
    """
    Read the actual light/dark square colours out of a board screenshot.

    Slices the board, takes the median of each cell's central region (avoiding
    gridline pixels at the borders), and splits by luminance.

    Only pass boards you know are checkerboards to avoid icons getting sampled.

    empty_rows: row indices to sample, to avoid pieces polluting the medians.
                Defaults to every row.

    Returns {"<stem>_light": bgr, "<stem>_dark": bgr}.
    """
    cfg = config or SlicerConfig()
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read board image: {image_path}")

    cells, rows, cols, _ = extract_board_cells(image, cfg)
    rows_to_use = range(rows) if empty_rows is None else empty_rows

    medians = []
    for r in rows_to_use:
        for c in range(cols):
            cell = cells.get((r, c))
            if cell is None or cell.size == 0:
                continue
            h, w = cell.shape[:2]
            core = cell[h // 4:3 * h // 4, w // 4:3 * w // 4]
            if core.size:
                medians.append(np.median(core.reshape(-1, 3), axis=0))

    if not medians:
        raise ValueError(f"No usable cells sampled from {image_path}")

    vals = np.array(medians)
    lum = vals.mean(axis=1)
    light = vals[lum > lum.mean()]
    dark = vals[lum <= lum.mean()]

    def dominant(group):
        """Most common cell colour in the group, not the mean.
        """
        counts = Counter(map(tuple, group.round().astype(int).tolist()))
        return tuple(counts.most_common(1)[0][0])

    stem = str(image_path).split("/")[-1].rsplit(".", 1)[0]
    return {f"{stem}_light": dominant(light), f"{stem}_dark": dominant(dark)}


def jitter_color(bgr, rng=None, hue=0.03, sat=0.12, val=0.10):
    """Small random shift in HSV, covering rendering and compression variation."""
    rng = rng or random
    b, g, r = [c / 255.0 for c in bgr]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + rng.uniform(-hue, hue)) % 1.0
    s = min(1.0, max(0.0, s * (1 + rng.uniform(-sat, sat))))
    v = min(1.0, max(0.0, v * (1 + rng.uniform(-val, val))))
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return tuple(int(round(c * 255)) for c in (b, g, r))


def composite_sprite(rgba, bg_color, size=128, scale=0.85, offset=(0, 0)):
    """
    Alpha-composite an RGBA sprite onto a solid square.

    rgba:     (H, W, 4) uint8, as cv2.imread(..., IMREAD_UNCHANGED) returns
    bg_color: BGR triple for the square behind the piece
    scale:    fraction of the cell the sprite occupies.
    offset:   (dx, dy) in pixels, for a little positional variation

    Returns a (size, size, 3) BGR image.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"expected an (H, W, 4) RGBA sprite, got shape {rgba.shape}")

    canvas = np.full((size, size, 3), bg_color, dtype=np.uint8)

    side = max(1, int(round(size * scale)))
    sprite = cv2.resize(rgba, (side, side), interpolation=cv2.INTER_AREA)
    fg = sprite[:, :, :3].astype(np.float32)
    alpha = (sprite[:, :, 3:4].astype(np.float32)) / 255.0

    x0 = (size - side) // 2 + offset[0]
    y0 = (size - side) // 2 + offset[1]
    x0 = max(0, min(size - side, x0))
    y0 = max(0, min(size - side, y0))

    region = canvas[y0:y0 + side, x0:x0 + side].astype(np.float32)
    canvas[y0:y0 + side, x0:x0 + side] = (
        alpha * fg + (1.0 - alpha) * region
    ).round().astype(np.uint8)
    return canvas


def has_alpha(image_rgba, min_transparent_frac=0.05):
    """True when an image carries enough real transparency to be worth compositing."""
    if image_rgba is None or image_rgba.ndim != 3 or image_rgba.shape[2] != 4:
        return False
    return float((image_rgba[:, :, 3] < 250).mean()) > min_transparent_frac
