"""
HTTP wrapper around pipeline.analyze_board, for the web UI in web/.

`app` is a bare module-level ASGI object rather than something started by a
__main__ block, so any host can consume it directly.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))     # before the src imports below;
                                               # matches how the CLI scripts
                                               # find each other
import cv2                                     # noqa: E402
import numpy as np                             # noqa: E402
from fastapi import FastAPI, File, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402

from classifier import CellClassifier         # noqa: E402
from pipeline import analyze_board            # noqa: E402

from .serialize import cells_zip, result_to_dict

CHECKPOINT = REPO_ROOT / "models" / "board_cnn.pt"

app = FastAPI(title="board_analyzer")

# Loaded once, at import, not per request. CellClassifier.load rebuilds the
# backbone and reads 1.6MB from disk -- roughly as long as the inference it
# would be serving. 
if not CHECKPOINT.exists():
    raise SystemExit(f"no checkpoint at {CHECKPOINT} -- run: python3 src/train.py")
CLASSIFIER = CellClassifier.load(str(CHECKPOINT))


def decode_upload(data):
    """
    Raw upload bytes -> the BGR array the pipeline expects.

    IMREAD_COLOR drops any alpha channel and yields 3-channel BGR 
    to align with find_board_bbox and bgr_array_to_pil 
    """
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            "could not read that file as an image -- PNG and JPEG screenshots work"
        )
    return image


def run(data):
    """
    Decode and analyse, or raise with a message worth showing the user.

    Both expected failures contain specific diagnostic text
    """
    return analyze_board(decode_upload(data), CLASSIFIER)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        return result_to_dict(run(await file.read()))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


@app.post("/api/cells")
async def cells(file: UploadFile = File(...)):
    """
    The sliced cells as a zip.

    Re-analyses the upload rather than caching the previous /api/analyze result
    """
    try:
        payload = cells_zip(run(await file.read()))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    return Response(
        payload,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="board.zip"'},
    )
