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


def peak_tolerance(n_cells, span, tol_frac=0.04, tol_min=2):
    """
    Search window, in px, when sampling a predicted boundary.

    Scaled to cell size rather than fixed.
    """
    if n_cells < 1:
        return tol_min
    return max(tol_min, int(round(span / n_cells * tol_frac)))


def normalise(raw, percentile=99.0):
    """
    Scale a raw profile to 0..1 using a high percentile as the ceiling.

    Not the max. A single strong edge -- a board frame, a UI selection
    highlight -- otherwise sets the ceiling and squashes every genuine gridline
    beneath it. Both checkers boards scored 0.19 against a 0.20 threshold purely
    because their wooden frame edge was ~5x stronger than their gridlines; at
    the 99th percentile they score 1.00.
    """
    top = float(np.percentile(raw, percentile))
    lo = float(raw.min())
    return np.clip((raw - lo) / (top - lo + 1e-9), 0.0, 1.0)


def score_grid(profile, n_cells, lo=None, hi=None, tol_frac=0.04,
               percentile=20.0, tol_min=2):
    """
    How well a grid of `n_cells` spanning lo..hi explains the profile.

    Iterate over possible lo/hi values since grid may not fill the bounding box.

    Samples the profile at each of the n_cells-1 interior boundaries, taking the
    max within a window scaled to cell size (see peak_tolerance), then
    aggregates with `percentile`.

    `profile` should be normalised to 0..1 so scores are comparable across axes.
    """
    n = len(profile)
    if n_cells < 2:
        return 0.0
    lo = 0 if lo is None else lo
    hi = n if hi is None else hi
    if hi - lo < n * 0.5:
        return 0.0
    cell = (hi - lo) / n_cells
    tol = peak_tolerance(n_cells, hi - lo, tol_frac, tol_min)
    peaks = []
    for i in range(1, n_cells):
        b = int(round(lo + i * cell))
        peaks.append(profile[max(0, b - tol):min(n, b + tol + 1)].max())
    return float(np.percentile(peaks, percentile))


def fit_axis_grid(profile, config=None):
    """
    Fit (n_cells, lo, hi) -- how many cells, and where the grid sits.

    Searched in two stages:

      1. symmetric coarse pass, hi = n - lo.
      2. small asymmetric search around the winner, for borders that are
         not symmetric.

    Returns (best_n, best_lo, best_hi, scores) where scores maps n -> its best
    score over all offsets, for the divisor tie-break and for reporting.
    """
    cfg = config or SlicerConfig()
    n = len(profile)
    max_inset = int(n * cfg.profile_max_inset_frac)
    step = max(1, cfg.profile_inset_step)
    refine = cfg.profile_refine_px

    def sc(nc, lo, hi):
        return score_grid(profile, nc, lo, hi, cfg.profile_peak_tol_frac,
                          cfg.profile_percentile, cfg.profile_peak_tol_min)

    scores, spans = {}, {}
    for nc in range(cfg.profile_n_min, cfg.profile_n_max + 1):
        best = None
        for inset in range(0, max_inset + 1, step):
            v = sc(nc, inset, n - inset)
            if best is None or v > best[0]:
                best = (v, inset, n - inset)
        _, lo0, hi0 = best
        for lo in range(max(0, lo0 - refine), lo0 + refine + 1, step):
            for hi in range(max(0, hi0 - refine), min(n, hi0 + refine) + 1, step):
                v = sc(nc, lo, hi)
                if v > best[0]:
                    best = (v, lo, hi)
        scores[nc], spans[nc] = best[0], (best[1], best[2])

    # Divisor tie-break
    cutoff = cfg.profile_rel_threshold * max(scores.values())
    best_n = max(nc for nc, v in scores.items() if v >= cutoff)
    return (best_n,) + spans[best_n] + (scores,)


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

    profile = normalise(raw, cfg.profile_norm_percentile)
    n_cells, lo, hi, scores = fit_axis_grid(profile, cfg)

    # The contrast guard above only rejects flat images.
    best_score = scores[n_cells]
    if best_score < cfg.profile_min_score:
        name = "rows" if axis == 0 else "columns"
        raise GridDetectionError(
            f"no regular grid along {name}: best score {best_score:.3f} is below "
            f"profile_min_score ({cfg.profile_min_score}). The image has "
            f"contrast ({contrast:.1f}) but its boundaries are not evenly "
            f"spaced -- a single strong edge looks like this."
        )

    return n_cells, lo, hi, scores


def infer_grid_size(board_image, config=None):
    """
    Infer (rows, cols) for a cropped board.

    Axes are inferred independently, so non-square NxM grids work.
    """
    cfg = config or SlicerConfig()
    rows, r_lo, r_hi, _ = infer_axis_size(board_image, 0, cfg)
    cols, c_lo, c_hi, _ = infer_axis_size(board_image, 1, cfg)
    return rows, cols, (r_lo, r_hi), (c_lo, c_hi)


def detect_grid_lines_by_profile(board_image, config=None):
    """
    Grid line positions from the inferred cell count.

    Returns (row_lines, col_lines) spanning the full crop.
    """
    cfg = config or SlicerConfig()
    rows, cols, (r_lo, r_hi), (c_lo, c_hi) = infer_grid_size(board_image, cfg)
    # spans the FITTED grid, not the whole crop -- see fit_axis_grid
    row_lines = [int(round(v)) for v in np.linspace(r_lo, r_hi, rows + 1)]
    col_lines = [int(round(v)) for v in np.linspace(c_lo, c_hi, cols + 1)]
    return row_lines, col_lines
