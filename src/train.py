"""
Train the board-cell classifier.

Since the backbone is frozen (see model.py), only the classification head is trained.
Checkpoint stores the class list, architecture config, and normalization constants.

Usage (run from the repo root, so the default data paths resolve):
    python3 src/train.py
    python3 src/train.py --epochs 2                 # smoke run
    python3 src/train.py --arch shufflenet_v2_x0_5 --input-size 96
"""

import argparse
import os
from dataclasses import asdict, fields

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from model import (
    ModelConfig,
    TrainConfig,
    backbone_normalization,
    build_eval_transform,
    build_model,
    pick_device,
)

DEFAULT_TRAIN_ROOT = "data/train/cells"
DEFAULT_VAL_ROOT = "data/val/cells"


# ---------------------------------------------------------------------------
# CLI: one --flag per dataclass field, mirroring board_slicer.py, so the CLI
# stays in sync as config fields are added or removed.
# ---------------------------------------------------------------------------
def _add_config_arguments(parser, config_cls, title, description):
    defaults = config_cls()
    group = parser.add_argument_group(title, description)
    for f in fields(config_cls):
        flag = "--" + f.name.replace("_", "-")
        current = getattr(defaults, f.name)
        if isinstance(current, bool):
            group.add_argument(flag, dest=f.name, default=current,
                               action=argparse.BooleanOptionalAction)
        else:
            group.add_argument(flag, dest=f.name, default=current,
                               type=type(current))


def _config_from_args(args, config_cls):
    """Build a config from the parsed namespace, plus the values that were
    overridden away from the dataclass defaults."""
    defaults = config_cls()
    values = {f.name: getattr(args, f.name) for f in fields(config_cls)}
    overrides = {k: v for k, v in values.items() if v != getattr(defaults, k)}
    return config_cls(**values), overrides


def load_datasets(train_root, val_root, model_cfg):
    """
    ImageFolder for both splits, sharing model.py's deterministic transform.
    """
    transform = build_eval_transform(model_cfg)
    train_ds = datasets.ImageFolder(train_root, transform=transform)
    val_ds = datasets.ImageFolder(val_root, transform=transform)

    if train_ds.classes != val_ds.classes:
        only_train = sorted(set(train_ds.classes) - set(val_ds.classes))
        only_val = sorted(set(val_ds.classes) - set(train_ds.classes))
        raise ValueError(
            "train and val class sets differ, so their label indices do not "
            f"agree.\n  only in {train_root}: {only_train}\n"
            f"  only in {val_root}: {only_val}"
        )
    return train_ds, val_ds


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """
    One pass over `loader`. Trains when an optimizer is given, else evaluates.

    Returns (mean_loss, accuracy).
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)

    return total_loss / max(seen, 1), correct / max(seen, 1)


def extract_features(model, dataset, batch_size, device):
    """
    Run the frozen backbone over a dataset once and keep the features.

    Valid only while the backbone is frozen.

    Returns a TensorDataset of (features, label)
    """
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for images, targets in DataLoader(dataset, batch_size=batch_size):
            feats.append(model.backbone(images.to(device)).cpu())
            labels.append(targets)
    return TensorDataset(torch.cat(feats), torch.cat(labels))


def per_class_report(module, loader, classes, device):
    """
    Accuracy broken down by class, worst first.
    """
    module.eval()
    correct = [0] * len(classes)
    total = [0] * len(classes)
    with torch.no_grad():
        for inputs, targets in loader:
            preds = module(inputs.to(device)).argmax(dim=1).cpu()
            for t, p in zip(targets.tolist(), preds.tolist()):
                total[t] += 1
                correct[t] += int(t == p)

    rows = sorted(
        ((correct[i] / total[i] if total[i] else 0.0, classes[i], correct[i], total[i])
         for i in range(len(classes))),
        key=lambda r: (r[0], -r[3]),
    )
    print(f"\nper-class validation accuracy (worst first)")
    print(f"  {'class':<24}{'n':>5}{'correct':>9}{'acc':>8}")
    for acc, name, ok, n in rows:
        flag = "  <-" if acc < 1.0 else ""
        print(f"  {name:<24}{n:>5}{ok:>9}{acc:>8.2f}{flag}")
    print(f"  {'TOTAL':<24}{sum(total):>5}{sum(correct):>9}"
          f"{sum(correct)/max(sum(total),1):>8.3f}")


def warn_split_imbalance(train_ds, val_ds):
    """
    Flag classes with at least as many validation as training images.
    """
    tr = [0] * len(train_ds.classes)
    va = [0] * len(val_ds.classes)
    for _, t in train_ds.samples:
        tr[t] += 1
    for _, t in val_ds.samples:
        va[t] += 1

    bad = [(train_ds.classes[i], tr[i], va[i])
           for i in range(len(tr)) if va[i] >= tr[i]]
    if not bad:
        return
    print(f"WARNING  {len(bad)} of {len(tr)} classes have >= as many val as train "
          f"images, so val accuracy is optimistic:")
    for name, t, v in sorted(bad, key=lambda b: b[1] - b[2])[:8]:
        print(f"           {name:<24} {t:>3} train / {v:>3} val")
    if len(bad) > 8:
        print(f"           ... and {len(bad) - 8} more")
    print()


def save_checkpoint(path, model, classes, model_cfg):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mean, std = backbone_normalization(model_cfg.arch)
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "classes": classes,
            "arch": model_cfg.arch,
            "input_size": model_cfg.input_size,
            "norm": {"mean": mean, "std": std},
            "model_cfg": asdict(model_cfg),
            "version": 1,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train the board-cell classification head."
    )
    parser.add_argument("--train-root", default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--val-root", default=DEFAULT_VAL_ROOT)
    parser.add_argument("--device", default=None,
                        help="cuda / mps / cpu (default: best available)")
    _add_config_arguments(parser, ModelConfig, "model config",
                          "Architecture and preprocessing; defaults from model.py")
    _add_config_arguments(parser, TrainConfig, "train config",
                          "Optimization settings; defaults from model.py")
    args = parser.parse_args()

    model_cfg, model_over = _config_from_args(args, ModelConfig)
    train_cfg, train_over = _config_from_args(args, TrainConfig)

    torch.manual_seed(train_cfg.seed)
    device = pick_device(args.device)

    if train_cfg.cache_features and not model_cfg.freeze_backbone:
        raise ValueError(
            "cache_features requires freeze_backbone." \
            " Pass --no-cache-features to train the backbone."
        )

    train_ds, val_ds = load_datasets(args.train_root, args.val_root, model_cfg)

    model, feature_dim = build_model(model_cfg, len(train_ds.classes))
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=train_cfg.lr,
                                  weight_decay=train_cfg.weight_decay)
    # CrossEntropyLoss expects RAW LOGITS -- it applies log_softmax itself, and
    # the head in model.py ends in a bare nn.Linear precisely so that holds.
    loss_fn = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)

    print(f"device        {device}")
    print(f"arch          {model_cfg.arch} @ {model_cfg.input_size}px  "
          f"(feature dim {feature_dim})")
    print(f"classes       {len(train_ds.classes)}")
    print(f"images        {len(train_ds)} train / {len(val_ds)} val")
    print(f"trainable     {sum(p.numel() for p in trainable):,} params "
          f"of {sum(p.numel() for p in model.parameters()):,}")
    print(f"cached feats  {train_cfg.cache_features}")
    overrides = {**model_over, **train_over}
    print(f"overrides     {overrides if overrides else 'none (all defaults)'}")
    print()

    warn_split_imbalance(train_ds, val_ds)

    # With cached features the head trains directly on backbone outputs, so the
    # module passed to run_epoch is model.head rather than the whole model
    if train_cfg.cache_features:
        train_data = extract_features(model, train_ds, train_cfg.batch_size, device)
        val_data = extract_features(model, val_ds, train_cfg.batch_size, device)
        module = model.head
    else:
        train_data, val_data, module = train_ds, val_ds, model

    train_loader = DataLoader(train_data, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=train_cfg.batch_size, shuffle=False)

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss, train_acc = run_epoch(module, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc = run_epoch(module, val_loader, loss_fn, device)
        if epoch % max(1, train_cfg.epochs // 20) == 0 or epoch == train_cfg.epochs:
            print(f"  epoch {epoch:>4}/{train_cfg.epochs}   "
                  f"loss {train_loss:.4f}   train_acc {train_acc:.3f}   "
                  f"val_loss {val_loss:.4f}   val_acc {val_acc:.3f}")

    per_class_report(module, val_loader, train_ds.classes, device)

    save_checkpoint(train_cfg.out_path, model, train_ds.classes, model_cfg)
    print(f"\nsaved checkpoint -> {train_cfg.out_path}")


if __name__ == "__main__":
    main()
