#!/usr/bin/env python3
"""
Composite transparent sprites onto realistic square colours.

Fixes a data bug where missing alpha is converted to black.
Dark pieces can incorrectly map to dark backgrounds losing information.

Output goes to data/train/cells/<class>/gen_*.png

Flow: assets/  (raw art)  ->  this script  ->  data/  (training images)

Regenerating with --clean removes only gen_* outputs, not sources.

Usage:
    python3 src/generate_composites.py --dry-run
    python3 src/generate_composites.py --clean --count 6 \
        --sample-from assets/boards/chess_1.png
"""

import argparse
import glob
import os
import random

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

DEFAULT_SPRITES = "assets/icons"
DEFAULT_OUT = "data/train/cells"
GEN_PREFIX = "gen_"

# Wikipedia chess sprites are named Chess_<piece><piece-colour><square>45:
#   piece        k q r b n p
#   piece colour l = light (white), d = dark (black)
#   square       l / d / t, where t is the transparent variant we composite
_PIECE = {"k": "king", "q": "queen", "r": "rook",
          "b": "bishop", "n": "knight", "p": "pawn"}

# The crowns cannot be parsed -- they were originally two files both called
# crown.png in different classes -- so they are mapped explicitly.
_EXPLICIT = {
    "crown_gold.png": "queens_gold_crown",
    "crown_gold_2.png": "queens_gold_crown",
    "crown_black.png": "queens_black_crown",
}


def sprite_class(filename):
    """
    Map a source filename in assets/icons to a dataset class, or None.

    Only PNG is recognised. Render SVG first with svg_to_png.py.
    """
    stem, ext = os.path.splitext(filename)
    if ext.lower() != ".png":
        return None
    if filename in _EXPLICIT:
        return _EXPLICIT[filename]
    if stem.startswith("Chess_") and stem.endswith("t45"):
        code = stem[len("Chess_"):-len("45")]        # e.g. "pdt"
        if len(code) == 3 and code[0] in _PIECE and code[1] in "ld":
            return f"chess_{'white' if code[1] == 'l' else 'black'}_{_PIECE[code[0]]}"
    return None


def find_sprites(sprite_dir):
    """
    Alpha-bearing sprites, as (class_name, path) pairs.

    Class comes from sprite_class(). 
    A file missing a class mapping is reported.
    """
    found, skipped = [], []
    for path in sorted(glob.glob(os.path.join(sprite_dir, "*.png"))):
        base = os.path.basename(path)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        alpha = has_alpha(img)
        cls = sprite_class(base)

        if cls is not None and alpha:
            found.append((cls, path))
        elif cls is not None:
            skipped.append((base, "mapped to a class but has no usable alpha"))
        elif alpha:
            skipped.append((base, "HAS ALPHA but no class mapping -- add it to "
                                  "_EXPLICIT or follow the Chess_*t45 convention"))
    return found, skipped


def build_backgrounds(sample_from, count, rng_seed=0):
    """
    Square colours to composite onto: sampled first, then the curated palette,
    then jittered repeats once both are exhausted.
    """
    rng = random.Random(rng_seed)
    colors = {}
    for board in sample_from or []:
        colors.update(sample_square_colors(board))
    colors.update(SQUARE_PALETTE)

    names = list(colors)
    chosen = {}
    for i in range(count):
        name = names[i % len(names)]
        if i < len(names):
            chosen[name] = colors[name]
        else:
            chosen[f"{name}_j{i // len(names)}"] = jitter_color(colors[name], rng)
    return chosen


def main():
    parser = argparse.ArgumentParser(
        description="Composite transparent sprites onto realistic square colours.")
    parser.add_argument("--sprites", default=DEFAULT_SPRITES,
                        help="Raw art directory. Never modified.")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Dataset root; images land in <out>/<class>/")
    parser.add_argument("--count", type=int, default=6,
                        help="Background colours per sprite")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--scale", type=float, default=0.70,
                        help="Fraction of the cell the sprite occupies")
    parser.add_argument("--sample-from", action="append", default=None,
                        metavar="BOARD_PNG",
                        help="Sample square colours from this board screenshot "
                             "(checkerboards only; repeatable)")
    parser.add_argument("--clean", action="store_true",
                        help=f"Remove existing {GEN_PREFIX}* outputs first. Only ever "
                             f"touches generated files, never the sources.")
    parser.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True,
                        help="Tight-crop sprites to their alpha bbox before scaling, so "
                             "--scale refers to the piece and not its padding")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sprites, skipped = find_sprites(args.sprites)
    if not sprites:
        raise SystemExit(
            f"No usable sprites in {args.sprites}. Expected Chess_*t45.png or a "
            f"file listed in _EXPLICIT, each with a real alpha channel. Render "
            f"the chess PNGs with: python3 src/svg_to_png.py {args.sprites} --size 128"
        )
    backgrounds = build_backgrounds(args.sample_from, args.count)

    print(f"sprites      {len(sprites)} from {args.sprites} "
          f"across {len({c for c, _ in sprites})} classes")
    for base, why in skipped:
        print(f"  skipped    {base}  ({why})")
    print(f"backgrounds  {len(backgrounds)}")
    for name, bgr in backgrounds.items():
        b, g, r = bgr
        print(f"               {name:<26} #{r:02X}{g:02X}{b:02X}")

    if args.clean:
        stale = glob.glob(os.path.join(args.out, "*", GEN_PREFIX + "*"))
        print(f"clean        removing {len(stale)} existing {GEN_PREFIX}* outputs")
        if not args.dry_run:
            for p in stale:
                os.remove(p)

    written, missing = 0, []
    for class_name, path in sprites:
        class_dir = os.path.join(args.out, class_name)
        if not os.path.isdir(class_dir):
            missing.append(class_name)
            continue
        rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        stem = os.path.splitext(os.path.basename(path))[0]
        for bg_name, bg in backgrounds.items():
            out = os.path.join(class_dir, f"{GEN_PREFIX}{stem}_{bg_name}.png")
            if not args.dry_run:
                cv2.imwrite(out, composite_sprite(rgba, bg, args.size, args.scale, crop=args.crop))
            written += 1

    for c in sorted(set(missing)):
        print(f"  WARNING    no class directory {os.path.join(args.out, c)} -- skipped")
    print(f"\n{'would write' if args.dry_run else 'wrote'}  {written} composited images "
          f"to {args.out}/<class>/")


if __name__ == "__main__":
    main()
