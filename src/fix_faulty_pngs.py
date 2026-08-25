#!/usr/bin/env python3
"""
Re-convert faulty PNGs in train/validation folders using the matching
source .svg files from a raw-assets folder, and overwrite the PNGs in
place. Uses resvg-py for accurate rendering.

Assumes train/validation folders mirror the raw-assets subfolder
structure, e.g.:
    assets/icons/foo.svg
    train/cells/class/foo.png
    val/cells/class/foo.png

Usage:
    python3 fix_faulty_pngs.py --raw /path/to/raw_assets \
        --train /path/to/train --validation /path/to/validation

    # Preview what would happen without writing anything:
    python3 fix_faulty_pngs.py --raw ... --train ... --validation ... --dry-run

    # Match the --size used in the original faulty conversion:
    python3 fix_faulty_pngs.py --raw ... --train ... --validation ... --size 256

Requires: pip install resvg-py
"""

import argparse
import os
import sys
 
try:
    import resvg_py
except ImportError:
    print("Missing dependency. Install it with:\n    pip install resvg-py")
    sys.exit(1)
 
 
def find_png_files(root_dir):
    png_paths = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".png"):
                png_paths.append(os.path.join(dirpath, fname))
    return png_paths
 
 
def build_svg_index(raw_assets_root):
    """Walk raw_assets_root once and build a basename (lowercase, no
    extension) -> full path index. Warns on duplicate basenames since
    those are ambiguous to match against."""
    index = {}
    duplicates = set()
    for dirpath, _dirnames, filenames in os.walk(raw_assets_root):
        for fname in filenames:
            if fname.lower().endswith((".svg", ".svgz")):
                base = os.path.splitext(fname)[0].lower()
                full_path = os.path.join(dirpath, fname)
                if base in index and index[base] != full_path:
                    duplicates.add(base)
                index[base] = full_path
    return index, duplicates
 
 
def find_matching_svg(png_path, svg_index):
    """Match a PNG to a source SVG by basename only (case-insensitive,
    ignoring folder structure and extension) since dataset folders may
    be organized into category subfolders that don't exist in raw_assets."""
    base = os.path.splitext(os.path.basename(png_path))[0].lower()
    return svg_index.get(base)
 
 
def reconvert(png_path, svg_path, size, dry_run):
    try:
        if dry_run:
            print(f"[dry-run] would overwrite: {png_path}  <-  {svg_path}")
            return True
 
        png_bytes = resvg_py.svg_to_bytes(svg_path=svg_path, width=size, height=size)
        with open(png_path, "wb") as f:
            f.write(bytes(png_bytes))
        print(f"fixed: {png_path}  <-  {svg_path}")
        return True
    except Exception as e:
        print(f"FAILED: {png_path} ({e})")
        return False
 
 
def process_dataset_folder(dataset_root, svg_index, size, dry_run):
    png_files = find_png_files(dataset_root)
    fixed = 0
    missing = []
 
    for png_path in png_files:
        svg_path = find_matching_svg(png_path, svg_index)
        if svg_path is None:
            missing.append(png_path)
            continue
        if reconvert(png_path, svg_path, size, dry_run):
            fixed += 1
 
    return fixed, len(png_files), missing
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Fix faulty PNGs in train/validation folders using matching source SVGs."
    )
    parser.add_argument("--raw", required=True, help="Path to raw assets folder containing source .svg files")
    parser.add_argument("--train", required=True, help="Path to train folder containing faulty .png files")
    parser.add_argument("--validation", required=True, help="Path to validation folder containing faulty .png files")
    parser.add_argument("--size", type=int, default=None,
                         help="Output width/height in pixels (square). Should match what was used originally.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Show what would be done without writing any files.")
    args = parser.parse_args()
 
    raw_root = os.path.abspath(args.raw)
    for label, path in [("raw assets", raw_root), ("train", args.train), ("validation", args.validation)]:
        if not os.path.isdir(path):
            print(f"Not a directory ({label}): {path}")
            sys.exit(1)
 
    svg_index, duplicates = build_svg_index(raw_root)
    print(f"Indexed {len(svg_index)} unique SVG basename(s) under {raw_root}")
    if duplicates:
        print(f"WARNING: {len(duplicates)} basename(s) appear more than once in raw assets "
              f"(ambiguous match, last one found wins): {sorted(duplicates)}")
 
    total_fixed = 0
    total_found = 0
    all_missing = []
 
    for label, dataset_dir in [("train", os.path.abspath(args.train)),
                                ("validation", os.path.abspath(args.validation))]:
        print(f"\n=== Processing {label} folder: {dataset_dir} ===")
        fixed, found, missing = process_dataset_folder(dataset_dir, svg_index, args.size, args.dry_run)
        total_fixed += fixed
        total_found += found
        all_missing.extend(missing)
        print(f"{label}: {fixed}/{found} PNGs {'would be ' if args.dry_run else ''}fixed")
 
    print(f"\nDone. {total_fixed}/{total_found} total PNGs {'would be ' if args.dry_run else ''}fixed.")
 
    if all_missing:
        print(f"\n{len(all_missing)} PNG(s) had NO matching SVG found in raw assets:")
        for p in all_missing:
            print(f"  - {p}")
 
 
if __name__ == "__main__":
    main()