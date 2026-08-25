#!/usr/bin/env python3
"""
Batch-convert all .svg files under a directory (recursively) to .png.

Uses svglib + reportlab -- pure Python/bundled C-extensions, no system-level
Cairo install required (unlike cairosvg).

Usage:
    python3 svg_to_png.py /path/to/folder
    python3 svg_to_png.py /path/to/folder --size 256
    python3 svg_to_png.py /path/to/folder --size 256 --out /path/to/output_folder
    python3 svg_to_png.py /path/to/folder --delete-svg

Requires: pip install svglib reportlab
"""

import argparse
import os
import sys

try:
    import resvg_py
except ImportError:
    print("Missing dependency. Install it with:\n    pip install svglib resvg-py")
    sys.exit(1)


def find_svg_files(root_dir):
    svg_paths = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".svg"):
                svg_paths.append(os.path.join(dirpath, fname))
    return svg_paths


def convert_svg_to_png(svg_path, root_dir, out_dir, size, delete_svg):
# Figure out where the .png should go
    if out_dir is None:
        # write alongside the original .svg
        png_path = os.path.splitext(svg_path)[0] + ".png"
    else:
        # mirror the folder structure under out_dir
        rel_path = os.path.relpath(svg_path, root_dir)
        rel_png = os.path.splitext(rel_path)[0] + ".png"
        png_path = os.path.join(out_dir, rel_png)
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
 
    try:
        # resvg_py.svg_to_bytes renders directly to PNG bytes.
        # width/height set together forces a square canvas at --size;
        # left as None, resvg uses the SVG's native intrinsic size.
        png_bytes = resvg_py.svg_to_bytes(
            svg_path=svg_path,
            width=size,
            height=size,
        )
 
        with open(png_path, "wb") as f:
            f.write(bytes(png_bytes))
 
        print(f"converted: {svg_path} -> {png_path}")
 
        if delete_svg:
            os.remove(svg_path)
            print(f"deleted source: {svg_path}")
 
        return True
    except Exception as e:
        print(f"FAILED: {svg_path} ({e})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch convert .svg files to .png")
    parser.add_argument("root_dir", help="Directory to search recursively for .svg files")
    parser.add_argument("--size", type=int, default=None,
                         help="Output width/height in pixels (square). Default: SVG's native size.")
    parser.add_argument("--out", type=str, default=None,
                         help="Output directory (mirrors folder structure). Default: write PNGs next to source SVGs.")
    parser.add_argument("--delete-svg", action="store_true",
                         help="Delete the original .svg file after successful conversion.")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    if not os.path.isdir(root_dir):
        print(f"Not a directory: {root_dir}")
        sys.exit(1)

    svg_files = find_svg_files(root_dir)
    if not svg_files:
        print(f"No .svg files found under {root_dir}")
        return

    print(f"Found {len(svg_files)} .svg file(s). Converting...\n")

    success_count = 0
    for svg_path in svg_files:
        if convert_svg_to_png(svg_path, root_dir, args.out, args.size, args.delete_svg):
            success_count += 1

    print(f"\nDone. {success_count}/{len(svg_files)} converted successfully.")


if __name__ == "__main__":
    main()