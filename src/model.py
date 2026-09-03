"""
Model architecture and preprocessing for board-cell classification.

A pretrained backbone is used as a frozen feature extractor, with its ImageNet
classifier head replaced by a small head sized to our own class count. Because
the backbone is frozen, training is a linear probe.

Requires: pip install torch torchvision
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.models import get_model, get_model_weights


# ---------------------------------------------------------------------------
# Supported backbones: arch name -> the attribute holding its classifier head.
#
# This is an explicit allowlist, NOT a hasattr() probe, because torchvision is
# inconsistent here and some models fail silently. squeezenet1_1, for example,
# flattens AFTER its classifier, so stripping the head yields an 86,528-dim
# tensor and a 2.8M-parameter "head" -- with no error raised.
#
# Verified head attr / feature dim for other candidates, if one is ever wanted:
#   resnet18            fc           512
#   mobilenet_v3_small  classifier   576
#   mobilenet_v2        classifier  1280
#   efficientnet_b0     classifier  1280
# Adding one is a single dict entry; run this file's __main__ to confirm it.
# ---------------------------------------------------------------------------
BACKBONES = {
    "shufflenet_v2_x0_5": "fc",
}


@dataclass
class ModelConfig:
    arch: str = "shufflenet_v2_x0_5"   # must be a key in BACKBONES
    input_size: int = 128              # board cells are ~93px, so 224 would just
                                        # upscale for nothing. Feature dim is
                                        # invariant to this (adaptive pooling),
                                        # so it never affects the head
    pretrained: bool = True            # ImageNet weights; False gives random init
    freeze_backbone: bool = True       # True => linear probe, the right choice
                                        # while the dataset is ~150 images
    dropout: float = 0.2               # applied before the final Linear


@dataclass
class TrainConfig:
    epochs: int = 300
    batch_size: int = 32
    lr: float = 1e-3                   # head-only, so this can be relatively high
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05      # mild, helps overconfidence with few
                                        # examples per class
    seed: int = 0
    out_path: str = "models/board_cnn.pt"


class PadToSquare:
    """
    Pad a PIL image to a square with its own border color, without scaling.
    The pad color is the median of the existing border pixels
    """

    def __call__(self, img):
        img = img.convert("RGB")
        w, h = img.size
        if w == h:
            return img

        side = max(w, h)
        canvas = Image.new("RGB", (side, side), self._border_color(img))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas

    @staticmethod
    def _border_color(img):
        px = img.load()
        w, h = img.size
        edge = (
            [px[x, 0] for x in range(w)] + [px[x, h - 1] for x in range(w)]
            + [px[0, y] for y in range(h)] + [px[w - 1, y] for y in range(h)]
        )
        # median per channel -- robust to a few outlier pixels on the border
        return tuple(sorted(c[i] for c in edge)[len(edge) // 2] for i in range(3))


def bgr_array_to_pil(cell):
    """
    Convert a cv2/numpy BGR crop into a PIL RGB image.
    """
    if not isinstance(cell, np.ndarray):
        raise ValueError(f"expected a numpy array, got {type(cell).__name__}")
    if cell.ndim != 3 or cell.shape[2] != 3:
        raise ValueError(
            f"expected an (H, W, 3) BGR array, got shape {cell.shape}"
        )
    if cell.size == 0:
        raise ValueError(
            f"empty cell crop (shape {cell.shape}) means two detected gridlines "
            "likely landed on the same coordinate, inspect sliced output"
        )
    if cell.dtype != np.uint8:
        raise ValueError(f"expected uint8 pixel data, got dtype {cell.dtype}")

    # [:, :, ::-1] leaves negative strides. Pillow accepts those directly, so
    # the reflexive np.ascontiguousarray() copy is pure cost here.
    return Image.fromarray(cell[:, :, ::-1])


def backbone_normalization(arch):
    """
    Read from torchvision's weight metadata

    Return (mean, std) that the pretrained checkpoint was trained with
    """
    _require_known(arch)
    meta = get_model_weights(arch).DEFAULT.transforms()
    return list(meta.mean), list(meta.std)


def build_eval_transform(config=None):
    """
    The deterministic preprocessing pipeline used by both training and inference.
    """
    cfg = config or ModelConfig()
    mean, std = backbone_normalization(cfg.arch)
    return T.Compose([
        PadToSquare(),
        T.Resize((cfg.input_size, cfg.input_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def pick_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BoardCellNet(nn.Module):
    """
    Frozen feature extractor + trainable classification head for board cells.
    """

    def __init__(self, backbone, head, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.freeze_backbone = freeze_backbone

    def train(self, mode=True):
        """Put the module in training mode, but hold a frozen backbone in eval.

        Keeping the backbone in eval() makes "frozen" actually mean frozen.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model(config=None, num_classes=None):
    """
    Build a BoardCellNet for `num_classes` categories.

    Replace the backbone head with nn.Identity to measure feature width

    Returns (model, feature_dim).
    """
    cfg = config or ModelConfig()
    if num_classes is None:
        raise ValueError("num_classes is required (pass len(dataset.classes))")
    _require_known(cfg.arch)

    backbone = get_model(cfg.arch, weights="DEFAULT" if cfg.pretrained else None)
    setattr(backbone, BACKBONES[cfg.arch], nn.Identity())

    backbone.eval()
    with torch.no_grad():
        probe = backbone(torch.zeros(1, 3, cfg.input_size, cfg.input_size))
    if probe.ndim != 2:
        raise RuntimeError(
            f"{cfg.arch}: expected (N, D) features after stripping the head, got "
            f"{tuple(probe.shape)}. This backbone needs pooling before the head."
        )
    feature_dim = probe.shape[1]

    if cfg.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

    head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(feature_dim, num_classes))
    return BoardCellNet(backbone, head, cfg.freeze_backbone), feature_dim


def _require_known(arch):
    if arch not in BACKBONES:
        raise ValueError(
            f"Unsupported arch {arch!r}. Known: {sorted(BACKBONES)}. "
            "Add it to BACKBONES with its head attribute name, then run "
            "`python3 src/model.py` to verify the feature dim is sane."
        )


# ---------------------------------------------------------------------------
# Standalone check: build the model and report what came out. 
# Touches no data and trains nothing.
#
#   python3 src/model.py
#   python3 src/model.py --arch shufflenet_v2_x0_5 --input-size 96
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build the model and report its shape.")
    parser.add_argument("--arch", default=ModelConfig.arch, choices=sorted(BACKBONES))
    parser.add_argument("--input-size", type=int, default=ModelConfig.input_size)
    parser.add_argument("--num-classes", type=int, default=33,
                        help="Defaults to the 33 folders currently in data/train/cells")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip the weight download (random init)")
    args = parser.parse_args()

    cfg = ModelConfig(arch=args.arch, input_size=args.input_size,
                      pretrained=not args.no_pretrained)
    model, dim = build_model(cfg, args.num_classes)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    mean, std = backbone_normalization(cfg.arch)

    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, cfg.input_size, cfg.input_size))

    print(f"arch            {cfg.arch}  (head attr: {BACKBONES[cfg.arch]})")
    print(f"input size      {cfg.input_size}x{cfg.input_size}")
    print(f"pretrained      {cfg.pretrained}")
    print(f"feature dim     {dim}   (measured, not hardcoded)")
    print(f"num classes     {args.num_classes}")
    print(f"output shape    {tuple(out.shape)}")
    print(f"params          {total:,} total / {trainable:,} trainable "
          f"({100 * trainable / total:.2f}%)")
    print(f"normalization   mean={mean} std={std}")
    print(f"transform       {' -> '.join(t.__class__.__name__ for t in build_eval_transform(cfg).transforms)}")


if __name__ == "__main__":
    main()
