"""
Turn a BoardResult into things that can cross an HTTP boundary.

BoardResult holds numpy arrays and tuple-keyed dicts, neither of which survives
json.dumps, so every conversion the API needs lives here rather than being
inlined into the route handlers.
"""

import base64
import io
import json
import zipfile

import cv2


def png_bytes(bgr_image):
    """
    Encode a BGR array as PNG bytes.
    """
    ok, buf = cv2.imencode(".png", bgr_image)
    if not ok:
        raise ValueError("failed to PNG-encode an image")
    return buf.tobytes()


def png_data_url(bgr_image):
    """
    Encode a BGR array as a data: URL that an <img> or canvas can consume.
    """
    return "data:image/png;base64," + base64.b64encode(png_bytes(bgr_image)).decode("ascii")


def grid(mapping, rows, cols, default=None):
    """
    A tuple-keyed {(row, col): value} dict as a row-major list of lists.

    BoardResult keys its labels and confidences by (row, col) tuples, which JSON
    has no representation for.
    """
    return [[mapping.get((r, c), default) for c in range(cols)]
            for r in range(rows)]


def matrix_dict(result):
    """
    Just the classification: size, labels, confidences.

    Shared by the JSON response and the matrix.json inside the cell zip, so the
    download and the on-screen board can never disagree.
    """
    return {
        "rows": result.rows,
        "cols": result.cols,
        "labels": grid(result.labels, result.rows, result.cols),
        "confidences": [[round(float(v), 4) for v in row]
                        for row in grid(result.confidences, result.rows,
                                        result.cols, default=0.0)],
    }


def result_to_dict(result):
    """
    BoardResult -> a JSON-safe dict.

    The board crop is the only image sent. row_lines/col_lines are relative to
    that crop, so the client can draw gridlines and labels itself and every view
    costs one image over the wire rather than one per view.
    """
    return {
        **matrix_dict(result),
        "bbox": [int(v) for v in result.bbox],
        "row_lines": [int(v) for v in result.row_lines],
        "col_lines": [int(v) for v in result.col_lines],
        "board_crop_png": png_data_url(result.board_crop),
    }


def cells_zip(result):
    """
    Every sliced cell as a zip, plus the matrix that labelled them.

    Cells are stored flat as cells/r<row>c<col>.png rather than filed into
    per-class folders. Filing by prediction would quietly turn a wrong
    prediction into someone's ground truth the moment they reused the download
    as training data.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("matrix.json", json.dumps(matrix_dict(result), indent=2))
        for (r, c), cell in sorted(result.cells.items()):
            if cell.size == 0:      # a degenerate slice; slice_cells can return one
                continue
            zf.writestr(f"cells/r{r}c{c}.png", png_bytes(cell))
    return buf.getvalue()
