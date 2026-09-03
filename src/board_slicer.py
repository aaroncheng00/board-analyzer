"""
Board grid detection and cell-slicing pipeline.

Step 1: find_board_bbox   -- locate the board's outer boundary in a screenshot
Step 2: detect_grid_lines -- find row/column gridlines via Hough transform
Step 3: slice_cells       -- crop out individual cell images based on
                              detected line positions

Every tunable number lives in SlicerConfig (slicer_config.py)

Requires: pip install opencv-python-headless numpy
"""

import argparse

import cv2
import numpy as np

try:
    from slicer_config import SlicerConfig
except ImportError:
    from .slicer_config import SlicerConfig


# Step 1: Find the board's bounding box within a larger image
def find_board_bbox(image, config=None):
    """
    Locate the board as the largest roughly-square contour in the image.

    image: BGR image (as read by cv2.imread)
    config: SlicerConfig -- uses the bbox_* fields; see slicer_config.py

    Returns (x, y, w, h) bounding box of the detected board.
    """
    cfg = config or SlicerConfig()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, cfg.bbox_canny_low, cfg.bbox_canny_high)
    # Dilate edges slightly so broken/antialiased border lines 
    # connect into one continuous contour
    kernel = np.ones((cfg.bbox_dilate_kernel, cfg.bbox_dilate_kernel), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=cfg.bbox_dilate_iterations)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found -- is this image mostly blank?")

    img_area = image.shape[0] * image.shape[1]
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < img_area * cfg.bbox_min_area_frac:
            continue
        aspect = w / h if h > 0 else 0
        # Accepted boards should be close to square
        sq = cfg.bbox_aspect_tol
        if sq <= aspect <= 1 / sq:
            candidates.append((area, x, y, w, h))

    if not candidates:
        raise ValueError("No board-like (square-ish, large) contour found")

    # Pick the candidate with largest area
    candidates.sort(reverse=True, key=lambda c: c[0])
    _, x, y, w, h = candidates[0]
    return x, y, w, h


# Step 2: Detect grid lines within the board crop
def _cluster_positions(values, gap):
    """Group nearby line positions (Hough often reports the same physical
    line multiple times with slight pixel offsets) into single positions."""
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [int(np.mean(c)) for c in clusters]


def _uniform_positions(lines):
    """
    Keep the number of detected lines, but respace them evenly.

    Assumes cells are equally sized and square.
    """
    if len(lines) < 3:
        return lines 
    return [int(round(v)) for v in np.linspace(lines[0], lines[-1], len(lines))]


def detect_grid_lines(board_image, config=None):
    """
    Detect horizontal and vertical gridlines within a cropped board image.

    board_image: BGR image containing only the board
    config: SlicerConfig, see slicer_config.py

    The Hough vote threshold is derived from the computed minLineLength

    Returns (row_lines, col_lines), both sorted in ascending order
    """
    cfg = config or SlicerConfig()

    h, w = board_image.shape[:2]
    gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, cfg.line_canny_low, cfg.line_canny_high)

    min_len_h = int(w * cfg.min_line_frac)
    min_len_v = int(h * cfg.min_line_frac)
    min_line_length = min(min_len_h, min_len_v)

    max_line_gap = int(min(w, h) * cfg.max_line_gap_frac)

    hough_threshold = max(
        cfg.hough_threshold_min,
        int(min_line_length * cfg.hough_threshold_frac),
    )

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    horizontal_ys, vertical_xs = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))
            angle = abs(angle) % 180

            if angle < cfg.angle_tol_deg and abs(dx) >= min_len_h * cfg.line_run_frac:
                horizontal_ys.append((y1 + y2) // 2)
            elif abs(angle - 90) < cfg.angle_tol_deg and abs(dy) >= min_len_v * cfg.line_run_frac:
                vertical_xs.append((x1 + x2) // 2)

    cluster_gap = max(cfg.cluster_gap_min, int(min(w, h) * cfg.cluster_gap_frac))

    row_lines = _cluster_positions(horizontal_ys, gap=cluster_gap)
    col_lines = _cluster_positions(vertical_xs, gap=cluster_gap)

    # Include outer edges as lines in case the outermost gridline wasn't detected
    if not row_lines or row_lines[0] > cluster_gap:
        row_lines = [0] + row_lines
    if not row_lines or row_lines[-1] < h - cluster_gap:
        row_lines = row_lines + [h]
    if not col_lines or col_lines[0] > cluster_gap:
        col_lines = [0] + col_lines
    if not col_lines or col_lines[-1] < w - cluster_gap:
        col_lines = col_lines + [w]

    # Re-cluster after inserting fallback boundaries
    row_lines = _cluster_positions(row_lines, gap=cluster_gap)
    col_lines = _cluster_positions(col_lines, gap=cluster_gap)

    if cfg.uniform_spacing:
        row_lines = _uniform_positions(row_lines)
        col_lines = _uniform_positions(col_lines)

    return row_lines, col_lines


# Step 3: Slice the board into individual cell images
def slice_cells(board_image, row_lines, col_lines):
    """
    Crop out each cell using detected gridline positions.
    
    Returns a dict {(row, col): cell_image}, 0-indexed from top-left.
    Emits cells in BGR (not RGB)
    """
    cells = {}
    num_rows = len(row_lines) - 1
    num_cols = len(col_lines) - 1

    for r in range(num_rows):
        for c in range(num_cols):
            y1, y2 = row_lines[r], row_lines[r + 1]
            x1, x2 = col_lines[c], col_lines[c + 1]
            cells[(r, c)] = board_image[y1:y2, x1:x2]

    return cells


# End-to-end wrapper from image to cell generation
def extract_board_cells(image, config=None):
    """
    Full pipeline: raw screenshot -> dict of {(row, col): cell_image}.

    config: SlicerConfig, threaded through to steps 1 and 2.
    """
    cfg = config or SlicerConfig()

    x, y, w, h = find_board_bbox(image, cfg)
    board_crop = image[y:y + h, x:x + w]

    row_lines, col_lines = detect_grid_lines(board_crop, cfg)
    num_rows, num_cols = len(row_lines) - 1, len(col_lines) - 1

    cells = slice_cells(board_crop, row_lines, col_lines)
    return cells, num_rows, num_cols, board_crop


# Debug helper
# Draws the detected bbox + gridlines on the original image
def draw_debug_overlay(image, bbox, row_lines, col_lines):
    debug_img = image.copy()
    x, y, w, h = bbox

    # board bounding box is red
    cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # gridlines are green and drawn relative to the board crop, offset back
    # into the original image's coordinate space
    for ry in row_lines:
        cv2.line(debug_img, (x, y + ry), (x + w, y + ry), (0, 255, 0), 1)
    for cx in col_lines:
        cv2.line(debug_img, (x + cx, y), (x + cx, y + h), (0, 255, 0), 1)

    return debug_img


# ---------------------------------------------------------------------------
# CLI
# Save every sliced cell as its own file
# Plus one debug overlay image showing the detected bbox + gridlines on the original.
#
# Every SlicerConfig field gets its own --flag automatically
#
# Usage:
#   python3 board_slicer.py path/to/screenshot.png
#   python3 board_slicer.py path/to/screenshot.png --out debug_output
#   python3 board_slicer.py path/to/screenshot.png --min-line-frac 0.6
# ---------------------------------------------------------------------------
def _add_config_arguments(parser):
    """Add one --flag per SlicerConfig field, so the CLI stays in sync with
    the dataclass as fields are added or removed."""
    from dataclasses import fields

    defaults = SlicerConfig()
    group = parser.add_argument_group(
        "slicer config",
        "Per-run overrides for SlicerConfig; defaults come from slicer_config.py",
    )
    for f in fields(SlicerConfig):
        flag = "--" + f.name.replace("_", "-")
        current = getattr(defaults, f.name)
        if isinstance(current, bool):
            group.add_argument(flag, dest=f.name, default=current,
                               action=argparse.BooleanOptionalAction)
        else:
            group.add_argument(flag, dest=f.name, default=current,
                               type=type(current))


def _config_from_args(args):
    """Build a SlicerConfig from the parsed namespace, and report which values
    were overridden away from the slicer_config.py defaults."""
    from dataclasses import fields

    defaults = SlicerConfig()
    values = {f.name: getattr(args, f.name) for f in fields(SlicerConfig)}
    overrides = {k: v for k, v in values.items() if v != getattr(defaults, k)}
    return SlicerConfig(**values), overrides


def main():
    import os

    parser = argparse.ArgumentParser(
        description="Run the board-slicing pipeline on a single image and "
                    "save the results for visual inspection."
    )
    parser.add_argument("image_path", help="Path to a screenshot containing a board")
    parser.add_argument("--out", default="slicer_debug_output",
                        help="Output directory for sliced cells + debug overlay")
    _add_config_arguments(parser)
    args = parser.parse_args()

    cfg, overrides = _config_from_args(args)

    image = cv2.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image_path}")

    os.makedirs(args.out, exist_ok=True)

    print(f"Loaded image: {args.image_path}  shape={image.shape}")
    if overrides:
        pretty = ", ".join(f"{k}={v}" for k, v in overrides.items())
        print(f"Config overrides: {pretty}")
    else:
        print("Config: all slicer_config.py defaults")

    try:
        bbox = find_board_bbox(image, cfg)
    except ValueError as e:
        print(f"FAILED at Step 1 (find_board_bbox): {e}")
        return

    x, y, w, h = bbox
    print(f"Step 1 -- board bbox: x={x}, y={y}, w={w}, h={h}")
    board_crop = image[y:y + h, x:x + w]
    cv2.imwrite(os.path.join(args.out, "01_board_crop.png"), board_crop)

    row_lines, col_lines = detect_grid_lines(board_crop, cfg)
    num_rows, num_cols = len(row_lines) - 1, len(col_lines) - 1
    print(f"Step 2 -- detected grid: {num_rows} rows x {num_cols} cols")
    print(f"  row_lines: {row_lines}")
    print(f"  col_lines: {col_lines}")

    debug_img = draw_debug_overlay(image, bbox, row_lines, col_lines)
    debug_path = os.path.join(args.out, "02_debug_overlay.png")
    cv2.imwrite(debug_path, debug_img)
    print(f"  saved debug overlay -> {debug_path}")

    # Save the raw Canny edge map that detect_grid_lines is working from
    gray = cv2.cvtColor(board_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, cfg.line_canny_low, cfg.line_canny_high)
    edges_path = os.path.join(args.out, "03_canny_edges.png")
    cv2.imwrite(edges_path, edges)
    print(f"  saved raw edge map -> {edges_path}")

    cells = slice_cells(board_crop, row_lines, col_lines)
    cells_dir = os.path.join(args.out, "cells")
    os.makedirs(cells_dir, exist_ok=True)
    for (r, c), cell_img in cells.items():
        if cell_img.size == 0:
            print(f"  WARNING: empty crop at ({r},{c}), skipping")
            continue
        cv2.imwrite(os.path.join(cells_dir, f"cell_{r:02d}_{c:02d}.png"), cell_img)

    print(f"Step 3 -- saved {len(cells)} cell images -> {cells_dir}")
    print(f"\nDone. Inspect {debug_path} first -- if the red box and green "
          f"lines don't align with the real board, tune the thresholds in "
          f"slicer_config.py (or pass them as --flags, see --help) before "
          f"trusting the cells output.")


if __name__ == "__main__":
    main()
