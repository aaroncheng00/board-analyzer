"""
Run a trained checkpoint over board cells.

Input is {(row, col): array}
Output is a label and confidence per cell.

Usage:
    from classifier import CellClassifier
    clf = CellClassifier.load("models/board_cnn.pt")
    preds = clf.predict(cells)          # {(r, c): (label, confidence)}

Or from the command line, to sanity-check a checkpoint:
    python3 src/classifier.py models/board_cnn.pt data/val/cells/empty/*.png
"""

import torch
import torch.nn.functional as F

from model import (
    ModelConfig,
    bgr_array_to_pil,
    build_eval_transform,
    build_model,
    pick_device,
)

CHECKPOINT_VERSION = 1


class CellClassifier:
    """
    A loaded checkpoint, ready to classify cells.
    """

    def __init__(self, model, classes, model_cfg, device):
        self.model = model
        self.classes = list(classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.model_cfg = model_cfg
        self.device = device
        self.transform = build_eval_transform(model_cfg)

    @classmethod
    def load(cls, path, device=None):
        """
        Rebuild the model described by a checkpoint and load its weights.

        Model architecture comes from ckpt["model_cfg"]
        """
        device = pick_device(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)

        missing = {"state_dict", "classes", "model_cfg"} - set(ckpt)
        if missing:
            raise ValueError(f"{path}: checkpoint is missing {sorted(missing)}")
        if ckpt.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"{path}: checkpoint version {ckpt.get('version')!r}, expected "
                f"{CHECKPOINT_VERSION}. Retrain with the current train.py."
            )

        model_cfg = ModelConfig(**ckpt["model_cfg"])
        model, _ = build_model(model_cfg, len(ckpt["classes"]))
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()

        return cls(model, ckpt["classes"], model_cfg, device)

    def predict(self, cells):
        """
        Classify the {(row, col): bgr_array} dict from board_slicer.slice_cells.

        Returns {(row, col): (label, confidence)} with the same keys.

        Empty crops come back as (None, 0.0)
        """
        keys = list(cells)
        usable = [k for k in keys if cells[k].size > 0]

        results = {k: (None, 0.0) for k in keys}
        if usable:
            for key, pred in zip(usable, self.predict_batch([cells[k] for k in usable])):
                results[key] = pred
        return results

    def predict_batch(self, images):
        """
        Classify a list of BGR arrays in one forward pass.

        Returns [(label, confidence), ...] aligned with `images`.
        """
        if not images:
            return []

        batch = torch.stack([
            self.transform(bgr_array_to_pil(img)) for img in images
        ]).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            # Softmax here and not in the model
            probs = F.softmax(logits, dim=1)

        confidences, indices = probs.max(dim=1)
        return [
            (self.classes[i], float(c))
            for i, c in zip(indices.tolist(), confidences.tolist())
        ]


# ---------------------------------------------------------------------------
# Standalone check: classify some image files with a checkpoint.
#
#   python3 src/classifier.py models/board_cnn.pt data/val/cells/empty/*.png
# ---------------------------------------------------------------------------
def main():
    import argparse

    import cv2

    parser = argparse.ArgumentParser(
        description="Classify individual cell images with a trained checkpoint."
    )
    parser.add_argument("checkpoint")
    parser.add_argument("images", nargs="+", help="Cell image files")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu")
    args = parser.parse_args()

    clf = CellClassifier.load(args.checkpoint, args.device)
    print(f"checkpoint  {args.checkpoint}")
    print(f"arch        {clf.model_cfg.arch} @ {clf.model_cfg.input_size}px")
    print(f"classes     {len(clf.classes)}")
    print(f"device      {clf.device}\n")

    arrays, names = [], []
    for path in args.images:
        img = cv2.imread(path)
        if img is None:
            print(f"  SKIP  unreadable: {path}")
            continue
        arrays.append(img)
        names.append(path)

    for name, (label, conf) in zip(names, clf.predict_batch(arrays)):
        print(f"  {conf:6.1%}  {label:<24} {name}")


if __name__ == "__main__":
    main()
