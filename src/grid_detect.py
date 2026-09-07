"""
Grid-size inference from periodicity, as an alternative to Hough line detection.

This module keys on periodicity instead. For each pixel row y, take the mean of 
the row difference across the full width.

At real cell boundary every square should change shade at once, so this value should spike.

Sweep N and score each by how strong the profile is at the boundaries N predicts.
Take the largest valid candidate since the true N may have divisors.
"""

import cv2
import numpy as np

try:
    from slicer_config import SlicerConfig
except ImportError:  # imported as part of the `src` package
    from .slicer_config import SlicerConfig


class GridDetectionError(ValueError):
    """
    Raised when the image carries no usable grid signal.
    """


def boundary_profile(board_image, axis):
    """
    1-D profile of how much the image changes across each candidate boundary.

    axis 0 -> row boundaries (differences down the image)
    axis 1 -> column boundaries (the same, transposed)

    Returns (raw, contrast):
      raw       the un-normalised profile, length (dimension - 1)
      contrast  peak / median of raw
    """
    gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    if axis == 1:
        gray = gray.T
    raw = np.abs(np.diff(gray, axis=0)).mean(axis=1)

    peak = float(raw.max())
    median = float(np.median(raw))

    if peak <= 1e-9:
        contrast = 0.0
    elif median <= 1e-9:
        contrast = float("inf")
    else:
        contrast = peak / median
    return raw, contrast


def score_grid(profile, n_cells, tol=2, percentile=20.0):
    """
    How well a uniform division into `n_cells` explains the profile.

    Samples the profile at each of the n_cells-1 interior boundaries used by a uniform grid.
    Take the max within +/- tol px to absorb rounding and anti-aliasing.

    Determine final score with `percentile`.

    `profile` should be normalised to 0..1 so scores are comparable across axes.
    """
    n = len(profile)
    if n_cells < 2:
        return 0.0
    peaks = []
    for i in range(1, n_cells):
        b = int(round(i * n / n_cells))
        peaks.append(profile[max(0, b - tol):min(n, b + tol + 1)].max())
    return float(np.percentile(peaks, percentile))


def infer_axis_size(board_image, axis, config=None):
    """
    Infer the number of cells along one axis.

    Returns (n_cells, scores) where scores maps N -> score, for reporting.

    Since divisors tie, pick largest scoring N within `profile_rel_threshold`.

    Contrast guard checks that image is not completely uniform
    """
    cfg = config or SlicerConfig()
    raw, contrast = boundary_profile(board_image, axis)

    if contrast < cfg.profile_min_contrast:
        name = "rows" if axis == 0 else "columns"
        raise GridDetectionError(
            f"no usable grid signal along {name}: contrast {contrast:.1f} is "
            f"below profile_min_contrast ({cfg.profile_min_contrast}). Real "
            f"boards measure 36-547; a featureless image measures ~1. The image "
            f"likely has neither shade alternation nor drawn gridlines."
        )

    profile = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    scores = {
        n: score_grid(profile, n, cfg.profile_peak_tol, cfg.profile_percentile)
        for n in range(cfg.profile_n_min, cfg.profile_n_max + 1)
    }

    # The contrast guard above only rejects flat images.
    best_score = max(scores.values())
    if best_score < cfg.profile_min_score:
        name = "rows" if axis == 0 else "columns"
        raise GridDetectionError(
            f"no regular grid along {name}: best score {best_score:.3f} is below "
            f"profile_min_score ({cfg.profile_min_score}). The image has "
            f"contrast ({contrast:.1f}) but its boundaries are not evenly "
            f"spaced -- a single strong edge looks like this."
        )

    cutoff = cfg.profile_rel_threshold * best_score
    return max(n for n, s in scores.items() if s >= cutoff), scores


def infer_grid_size(board_image, config=None):
    """
    Infer (rows, cols) for a cropped board.

    Axes are inferred independently, so non-square NxM grids work.
    """
    cfg = config or SlicerConfig()
    rows, _ = infer_axis_size(board_image, 0, cfg)
    cols, _ = infer_axis_size(board_image, 1, cfg)
    return rows, cols


def detect_grid_lines_by_profile(board_image, config=None):
    """
    Grid line positions from the inferred cell count.

    Returns (row_lines, col_lines) spanning the full crop.
    """
    cfg = config or SlicerConfig()
    h, w = board_image.shape[:2]
    rows, cols = infer_grid_size(board_image, cfg)
    row_lines = [int(round(v)) for v in np.linspace(0, h, rows + 1)]
    col_lines = [int(round(v)) for v in np.linspace(0, w, cols + 1)]
    return row_lines, col_lines
