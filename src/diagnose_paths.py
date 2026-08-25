#!/usr/bin/env python3
"""
Diagnostic: compare relative paths under raw-assets vs train/validation
to figure out why SVG<->PNG matching is failing.

Usage:
    python3 diagnose_paths.py --raw /path/to/raw_assets --train /path/to/train --validation /path/to/validation
"""

import argparse
import os


def list_relpaths(root_dir, exts):
    paths = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(exts):
                rel = os.path.relpath(os.path.join(dirpath, fname), root_dir)
                paths.append(rel)
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    args = parser.parse_args()

    raw_paths = list_relpaths(args.raw, (".svg", ".svgz"))
    train_paths = list_relpaths(args.train, (".png",))
    val_paths = list_relpaths(args.validation, (".png",))

    print(f"raw_assets: {len(raw_paths)} svg files")
    print(f"train: {len(train_paths)} png files")
    print(f"validation: {len(val_paths)} png files\n")

    print("=== First 10 raw_assets relative paths ===")
    for p in raw_paths[:10]:
        print(f"  {p}")

    print("\n=== First 10 train relative paths ===")
    for p in train_paths[:10]:
        print(f"  {p}")

    print("\n=== First 10 validation relative paths ===")
    for p in val_paths[:10]:
        print(f"  {p}")

    # Try to find any near-matches by basename only (ignoring folders/case/ext)
    raw_basenames = {os.path.splitext(os.path.basename(p))[0].lower(): p for p in raw_paths}
    train_basenames = {os.path.splitext(os.path.basename(p))[0].lower(): p for p in train_paths}

    exact_basename_matches = set(raw_basenames.keys()) & set(train_basenames.keys())
    print(f"\n=== Basename-only match check (ignoring folder + extension + case) ===")
    print(f"{len(exact_basename_matches)} train PNG(s) share a basename with a raw SVG")
    if exact_basename_matches:
        sample = list(exact_basename_matches)[:5]
        for b in sample:
            print(f"  basename '{b}':")
            print(f"    raw:   {raw_basenames[b]}")
            print(f"    train: {train_basenames[b]}")


if __name__ == "__main__":
    main()