# board_analyzer

Read a board game screenshot and return a labeled grid of what is on each square.

Given a screenshot, the pipeline locates the board, infers the
grid size, slices out each cell, and classifies the contents.

```
screenshot ──▶ find board border ──▶ infer grid lines ──▶ slice cells ──▶ classify ──▶ N×M matrix
```

## Supported games

| game | classes | training images | end-to-end tested |
|---|---|---|---|
| Chess | 12 | 131 | yes |
| Tango | 4 | 22 | yes |
| Queens | 2 | 20 | no |
| Checkers | 2 | 6 | no |

Plus `empty` and ten `digit_*` classes.

## Setup

Requires a **native arm64** environment on Apple Silicon. An x86_64 conda env under Rosetta
will not work, since PyTorch's last macOS x86_64 release is 2.2.2, which is incompatible with
NumPy 2.x, and `torch.from_numpy` fails.

```bash
CONDA_SUBDIR=osx-arm64 conda create -n board_analyzer_arm python=3.11
conda activate board_analyzer_arm
conda config --env --set subdir osx-arm64          # keep future installs native
pip install torch torchvision "opencv-python-headless<5"
```

Run everything from the repo root so the default data paths resolve.

## Usage

```bash
python3 src/pipeline.py assets/boards/chess_2.png --out pipeline_output
```

Prints the label matrix and a matching confidence matrix, and writes two diagnostics that
separate a slicing failure from a classification failure:

- `01_overlay.png` — Did the slicer work? Contains the bounding box and gridlines on the original
- `02_contact_sheet.png` — Did the classifier work? Contains every cell crop with its label and confidence

```python
from classifier import CellClassifier
from pipeline import analyze_board
import cv2

clf = CellClassifier.load("models/board_cnn.pt")
result = analyze_board(cv2.imread("board.png"), clf)

result.rows, result.cols        # inferred grid size
result.labels[(0, 0)]           # 'chess_black_rook', or None for an empty crop
result.confidences[(0, 0)]      # softmax probability
result.label_grid()             # list of lists, row-major
```

`analyze_board` prints nothing and returns a `BoardResult`, so it is safe to call from a loop.

To retrain: drop cell images into `data/train/cells/<class>/` and run `python3 src/train.py`.
Every `SlicerConfig`, `ModelConfig` and `TrainConfig` field is exposed as a CLI flag
automatically.
Use `--help` on any script for more info.

## Model

A frozen ImageNet backbone with a small trainable head. 
Swapping backbones is one config field.
The feature width is measured with a dummy forward pass rather than hardcoded.

```
shufflenet_v2_x0_5   ImageNet weights, FROZEN        341,792 params
  └── head           Dropout(0.2) → Linear(1024, 33)  33,825 params  ← the only trainable part

input      128×128 RGB   (board cells are ~90-180px, so 224 would just upscale)
output     33 logits     softmax applied at inference only, not in the model
latency    ~257 ms for a 64-cell board (CPU); faster on MPS
```

## Results

TBD.