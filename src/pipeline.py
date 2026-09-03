"""
Full board-analysis pipeline: screenshot in, labelled matrix out.

    screenshot -> find_board_bbox -> detect_grid_lines -> slice_cells
               -> CellClassifier.predict -> {(row, col): label}

Usage:
    python3 src/pipeline.py assets/boards/chess_1.png
    python3 src/pipeline.py assets/boards/chess_1.png --out pipeline_output
    python3 src/pipeline.py assets/boards/tango_1.png --no-uniform-spacing
"""

import argparse
import os
from dataclasses import dataclass, fields
from typing import Optional

import cv2
import numpy as np

from board_slicer import (
    detect_grid_lines,
    draw_debug_overlay,
    find_board_bbox,
    slice_cells,
)
from classifier import CellClassifier
from slicer_config import SlicerConfig

DEFAULT_CHECKPOINT = "models/board_cnn.pt"


@dataclass
class BoardResult:
    """
    Everything the pipeline learned about one board image.
    """

    rows: int
    cols: int
    labels: dict          # (row, col) -> class name, or None for an empty crop
    confidences: dict     # (row, col) -> softmax probability of that label
    bbox: tuple           # (x, y, w, h) in original image coordinates
    row_lines: list
    col_lines: list
    board_crop: np.ndarray
    cells: dict           # (row, col) -> BGR crop

    def label_grid(self):
        """labels as a list of lists, row-major."""
        return [[self.labels[(r, c)] for c in range(self.cols)]
                for r in range(self.rows)]


def analyze_board(image, classifier, slicer_config=None):
    """
    Run the whole chain on one BGR image and return a BoardResult.
    """
    cfg = slicer_config or SlicerConfig()

    bbox = find_board_bbox(image, cfg)
    x, y, w, h = bbox
    board_crop = image[y:y + h, x:x + w]

    row_lines, col_lines = detect_grid_lines(board_crop, cfg)
    cells = slice_cells(board_crop, row_lines, col_lines)

    preds = classifier.predict(cells)
    return BoardResult(
        rows=len(row_lines) - 1,
        cols=len(col_lines) - 1,
        labels={k: v[0] for k, v in preds.items()},
        confidences={k: v[1] for k, v in preds.items()},
        bbox=bbox,
        row_lines=row_lines,
        col_lines=col_lines,
        board_crop=board_crop,
        cells=cells,
    )


def format_matrix(result, width=13):
    """The label grid as aligned text."""
    out = []
    for r in range(result.rows):
        row = []
        for c in range(result.cols):
            name = result.labels[(r, c)]
            row.append(f"{'--' if name is None else name[:width]:>{width}}")
        out.append("  " + " ".join(row))
    return "\n".join(out)


def format_confidences(result):
    out = []
    for r in range(result.rows):
        out.append("  " + " ".join(f"{result.confidences[(r, c)]:.2f}"
                                   for c in range(result.cols)))
    return "\n".join(out)


def contact_sheet(result, tile=120):
    """
    Every cell crop with its predicted label and confidence written on it.
    """
    pad, bar = 6, 26
    cell_w = tile + pad * 2
    cell_h = tile + pad * 2 + bar
    sheet = np.full((cell_h * result.rows, cell_w * result.cols, 3), 255, np.uint8)

    for (r, c), crop in result.cells.items():
        y0, x0 = r * cell_h, c * cell_w
        if crop.size:
            sheet[y0 + pad:y0 + pad + tile, x0 + pad:x0 + pad + tile] = \
                cv2.resize(crop, (tile, tile))
        label = result.labels[(r, c)] or "EMPTY"
        conf = result.confidences[(r, c)]
        # low-confidence cells in red; see the note in main() about why this is
        # a display cue only and not a correctness signal
        color = (0, 0, 200) if conf < 0.5 else (0, 0, 0)
        cv2.putText(sheet, label[:16], (x0 + pad, y0 + pad + tile + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
        cv2.putText(sheet, f"{conf:.2f}", (x0 + pad, y0 + pad + tile + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
    return sheet


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_slicer_arguments(parser):
    """One --flag per SlicerConfig field, as in board_slicer.py and train.py."""
    defaults = SlicerConfig()
    group = parser.add_argument_group(
        "slicer config", "Per-run overrides; defaults from slicer_config.py")
    for f in fields(SlicerConfig):
        flag = "--" + f.name.replace("_", "-")
        current = getattr(defaults, f.name)
        if isinstance(current, bool):
            group.add_argument(flag, dest=f.name, default=current,
                               action=argparse.BooleanOptionalAction)
        else:
            group.add_argument(flag, dest=f.name, default=current,
                               type=type(current))


def main():
    parser = argparse.ArgumentParser(
        description="Slice a board screenshot and classify every cell.")
    parser.add_argument("image_path")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", default="pipeline_output",
                        help="Directory for the debug overlay and contact sheet")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu")
    _add_slicer_arguments(parser)
    args = parser.parse_args()

    slicer_cfg = SlicerConfig(**{f.name: getattr(args, f.name)
                                 for f in fields(SlicerConfig)})

    image = cv2.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image_path}")

    clf = CellClassifier.load(args.checkpoint, args.device)
    result = analyze_board(image, clf, slicer_cfg)

    os.makedirs(args.out, exist_ok=True)
    print(f"image       {args.image_path}  {image.shape[1]}x{image.shape[0]}")
    print(f"checkpoint  {args.checkpoint}  ({len(clf.classes)} classes, {clf.device})")
    x, y, w, h = result.bbox
    print(f"board bbox  x={x} y={y} w={w} h={h}")
    print(f"grid        {result.rows} rows x {result.cols} cols "
          f"({len(result.cells)} cells)")

    print(f"\nlabels:\n{format_matrix(result)}")
    print(f"\nconfidence:\n{format_confidences(result)}")

    confs = list(result.confidences.values())
    print(f"\n  confidence  min {min(confs):.2f}  median {np.median(confs):.2f}  "
          f"max {max(confs):.2f}")
    print(f"  distinct labels predicted: {len({l for l in result.labels.values()})}")

    overlay_path = os.path.join(args.out, "01_overlay.png")
    cv2.imwrite(overlay_path,
                draw_debug_overlay(image, result.bbox,
                                   result.row_lines, result.col_lines))
    sheet_path = os.path.join(args.out, "02_contact_sheet.png")
    cv2.imwrite(sheet_path, contact_sheet(result))
    print(f"\nsaved  {overlay_path}   <- did the SLICER work?")
    print(f"saved  {sheet_path}   <- did the CLASSIFIER work?")


if __name__ == "__main__":
    main()
