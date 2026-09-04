#!/usr/bin/env python3
"""
One-time generator: composite transparent sprites onto realistic square colours.

Fixes a data bug where chess and queens sprites are stored with transparent
backgrounds, so PIL's convert("RGB") drops alpha and keeps the underlying RGB
which is black in these files.

This script walks the class folders, composites every alpha-bearing sprite onto N
square colours, and writes the results beside the originals as gen_*.png so
ImageFolder picks them up with no code change. The class is taken from the
FOLDER, so no filename parsing is needed and there is no ambiguity about which
crown.png belongs to which class.

Usage:
    python3 src/generate_composites.py --dry-run
    python3 src/generate_composites.py --count 6 --sample-from assets/boards/chess_1.png
    python3 src/generate_composites.py --clean --delete-sources
"""

import argparse
import glob
import os

import cv2

try:
    from augment import (
        SQUARE_PALETTE,
        composite_sprite,
        has_alpha,
        jitter_color,
        sample_square_colors,
    )
except ImportError:
    from .augment import (
        SQUARE_PALETTE,
        composite_sprite,
        has_alpha,
        jitter_color,
        sample_square_colors,
    )

DEFAULT_ROOT = "data/train/cells"
GEN_PREFIX = "gen_"


def find_sprites(root):
    """Alpha-bearing files under root, as (class_name, path) pairs."""
    found = []
    for class_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(class_dir):
            continue
        for path in sorted(glob.glob(os.path.join(class_dir, "*"))):
            if os.path.basename(path).startswith(GEN_PREFIX):
                continue
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if has_alpha(img):
                found.append((os.path.basename(class_dir), path))
    return found


def build_backgrounds(sample_from, count, rng_seed=0):
    """
    Square colours to composite onto: sampled first, then the curated palette.
    """
    import random

    rng = random.Random(rng_seed)
    colors = {}
    for board in sample_from or []:
        colors.update(sample_square_colors(board))
    colors.update(SQUARE_PALETTE)

    names = list(colors)
    chosen = {}
    for i in range(count):
        name = names[i % len(names)]
        key = name if i < len(names) else f"{name}_j{i // len(names)}"
        value = colors[name] if i < len(names) else jitter_color(colors[name], rng)
        chosen[key] = value
    return chosen


def main():
    parser = argparse.ArgumentParser(
        description="Composite transparent sprites onto realistic square colours.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--count", type=int, default=6,
                        help="Background colours per sprite")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--scale", type=float, default=0.85,
                        help="Fraction of the cell the sprite occupies")
    parser.add_argument("--sample-from", action="append", default=None,
                        metavar="BOARD_PNG",
                        help="Sample square colours from this board screenshot "
                             "(checkerboards only; repeatable)")
    parser.add_argument("--clean", action="store_true",
                        help="Delete existing gen_*.png first (makes reruns idempotent)")
    parser.add_argument("--delete-sources", action="store_true",
                        help="Remove the transparent originals after generating. Safe: "
                             "they are duplicates of assets/icons/")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.clean:
        stale = glob.glob(os.path.join(args.root, "*", GEN_PREFIX + "*"))
        print(f"clean        removing {len(stale)} existing {GEN_PREFIX}* files")
        if not args.dry_run:
            for p in stale:
                os.remove(p)

    sprites = find_sprites(args.root)
    backgrounds = build_backgrounds(args.sample_from, args.count)

    print(f"root         {args.root}")
    print(f"sprites      {len(sprites)} alpha-bearing files across "
          f"{len({c for c, _ in sprites})} classes")
    print(f"backgrounds  {len(backgrounds)}")
    for name, bgr in backgrounds.items():
        b, g, r = bgr
        print(f"               {name:<24} BGR {str(tuple(bgr)):<18} #{r:02X}{g:02X}{b:02X}")
    print(f"output       {len(sprites) * len(backgrounds)} images at "
          f"{args.size}x{args.size}, scale {args.scale}")
    if args.dry_run:
        print("\n(dry run -- nothing written)")

    written = 0
    for class_name, path in sprites:
        rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        stem = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
        for bg_name, bg in backgrounds.items():
            out = os.path.join(os.path.dirname(path),
                               f"{GEN_PREFIX}{stem}_{bg_name}.png")
            if not args.dry_run:
                cv2.imwrite(out, composite_sprite(rgba, bg, args.size, args.scale))
            written += 1

    print(f"\nwrote        {written} composited images")

    if args.delete_sources:
        # Safe because every one of these is a derived duplicate from the assets folder
        print(f"\ndelete-sources  removing {len(sprites)} transparent originals:")
        for class_name, path in sprites:
            print(f"  {class_name:<22} {os.path.basename(path)}")
            if not args.dry_run:
                os.remove(path)


if __name__ == "__main__":
    main()
