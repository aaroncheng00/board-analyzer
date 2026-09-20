#!/usr/bin/env python3
"""
Extract a transparent sprite from flat-background cell crops.

Flow: data/train/cells/<class>/  (screenshots)  ->  this script  ->  assets/icons/
      then generate_composites.py turns the sprite into training images.

Usage:
    python3 src/extract_sprite.py "data/train/cells/queens_gold_crown/val_*.png" \
        --out assets/icons/crown_linkedin_gold.png --preview /tmp/preview.png

    python3 src/extract_sprite.py "cells/*.png" --out sprite.png --preview /tmp/p.png
"""

import argparse
import glob
import os

import cv2
import numpy as np

# Colour distance (L2 over BGR) at which a pixel stops being background and
# starts being sprite. Below `alpha_low` is fully transparent, above
# `alpha_high` fully opaque
ALPHA_LOW = 28.0
ALPHA_HIGH = 70.0

# Components smaller than this are keying noise (JPEG ringing, a stray
# anti-aliased pixel) and not part of the sprite.
MIN_COMPONENT_AREA = 20

# How well a crop's silhouette must agree with the group's before it
# is accepted as the same artwork. See drop_shape_outliers().
SHAPE_MIN_IOU = 0.90


def key_out_background(bgr, alpha_low=ALPHA_LOW, alpha_high=ALPHA_HIGH):
    """
    Split a flat-background crop into (rgba_sprite, background_colour).

    Slice out potential high contrast borderlines at edge of square
    """
    img = bgr.astype(np.float32)
    h, w = img.shape[:2]

    edge = 3
    border = np.concatenate([
        img[:edge].reshape(-1, 3), img[-edge:].reshape(-1, 3),
        img[:, :edge].reshape(-1, 3), img[:, -edge:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)

    distance = np.linalg.norm(img - bg, axis=2)
    alpha = np.clip((distance - alpha_low) / (alpha_high - alpha_low), 0.0, 1.0)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (alpha > 0.5).astype(np.uint8), 8)
    keep = []
    for i in range(1, count):
        x, y, cw, ch, area = stats[i]
        touches_border = x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1
        if not touches_border and area >= MIN_COMPONENT_AREA:
            keep.append(i)
    if not keep:
        raise ValueError("no sprite found: every component touches the border "
                         "or is below the noise threshold")

    mask = cv2.dilate(np.isin(labels, keep).astype(np.uint8),
                      np.ones((3, 3), np.uint8), iterations=2)
    alpha = alpha * mask

    ys, xs = np.where(alpha > 0.3)
    rgba = np.dstack([img, alpha * 255])[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return rgba, tuple(int(c) for c in bg)


def drop_shape_outliers(crops, names, min_iou=SHAPE_MIN_IOU):
    """
    Discard crops whose silhouette disagrees with the group's consensus.
    """
    size = (96, 96)
    masks = np.stack([
        cv2.resize((c[:, :, 3] > 128).astype(np.uint8) * 255, size,
                   interpolation=cv2.INTER_AREA) > 127
        for c in crops
    ])
    consensus = np.median(masks, axis=0) > 0.5

    kept, kept_names = [], []
    for crop, name, mask in zip(crops, names, masks):
        union = (mask | consensus).sum()
        iou = float((mask & consensus).sum() / union) if union else 0.0
        if iou < min_iou:
            print(f"  dropped   {os.path.basename(name):<44} shape IoU {iou:.2f} "
                  f"< {min_iou:.2f} -- different artwork?")
            continue
        kept.append(crop)
        kept_names.append(name)
    return kept, kept_names


def median_sprite(crops):
    """
    Combine several keyed crops of one sprite into a single clean RGBA image.
    """
    if not crops:
        raise ValueError("no crops to combine")

    heights = [c.shape[0] for c in crops]
    widths = [c.shape[1] for c in crops]
    size = (int(np.median(widths)), int(np.median(heights)))
    stack = np.stack([cv2.resize(c, size, interpolation=cv2.INTER_AREA)
                      for c in crops])
    sprite = np.median(stack, axis=0)

    alpha = sprite[:, :, 3:4] / 255.0
    opaque = sprite[:, :, :3][alpha[:, :, 0] > 0.9]
    if opaque.size:
        sprite[:, :, :3] = np.where(alpha < 0.05, opaque.mean(axis=0), sprite[:, :, :3])
    return sprite.astype(np.uint8)


def preview(rgba, backgrounds=((255, 0, 255), (218, 218, 218), (251, 112, 90))):
    """The sprite composited on a few squares, magenta first to expose fringing."""
    tiles = []
    fg = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4] / 255.0
    for bg in backgrounds:
        tiles.append((alpha * fg + (1 - alpha) * np.array(bg, np.float32)).astype(np.uint8))
    return np.hstack(tiles)


def main():
    parser = argparse.ArgumentParser(
        description="Key a sprite out of flat-background cell screenshots.")
    parser.add_argument("pattern",
                        help="Glob of crops of ONE sprite, quoted so the shell "
                             "does not expand it")
    parser.add_argument("--out", required=True, help="Destination RGBA PNG")
    parser.add_argument("--preview", default=None,
                        help="Also write a composite-on-several-squares preview here")
    parser.add_argument("--alpha-low", type=float, default=ALPHA_LOW)
    parser.add_argument("--alpha-high", type=float, default=ALPHA_HIGH)
    parser.add_argument("--min-shape-iou", type=float, default=SHAPE_MIN_IOU,
                        help="Reject crops whose silhouette agrees with the "
                             "group's by less than this IoU (they are probably "
                             "a different sprite)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files matched {args.pattern!r}")

    crops, used = [], []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  skipped   unreadable: {path}")
            continue
        try:
            rgba, bg = key_out_background(img, args.alpha_low, args.alpha_high)
        except ValueError as exc:
            print(f"  skipped   {os.path.basename(path)}  ({exc})")
            continue
        b, g, r = bg
        print(f"  keyed     {os.path.basename(path):<44} "
              f"square #{r:02X}{g:02X}{b:02X}  ->  {rgba.shape[1]}x{rgba.shape[0]}")
        crops.append(rgba)
        used.append(path)

    if not crops:
        raise SystemExit("No crop could be keyed. Are these flat-background cells?")

    crops, used = drop_shape_outliers(crops, used, args.min_shape_iou)
    if not crops:
        raise SystemExit("Every crop was dropped as a shape outlier.")

    sprite = median_sprite(crops)
    coverage = float((sprite[:, :, 3] > 128).mean())
    print(f"\nsprite      {sprite.shape[1]}x{sprite.shape[0]} from {len(crops)} crops, "
          f"{coverage:.0%} opaque")

    if args.dry_run:
        print(f"would write {args.out}")
        return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, sprite)
    print(f"wrote       {args.out}")
    if args.preview:
        cv2.imwrite(args.preview, preview(sprite))
        print(f"wrote       {args.preview}   <- check for a coloured fringe on magenta")


if __name__ == "__main__":
    main()
