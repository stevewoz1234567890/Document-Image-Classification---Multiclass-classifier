#!/usr/bin/env python3
"""
Walk a reorganized RVL-CDIP tree (e.g. train/*/ and test/*/) and count
(height, width) occurrences. Optionally plot shapes that appear at least
--min-count times (similar to the notebook histogram step).

Uses Pillow to read only image headers where possible (fast for large trees).
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

from PIL import Image


def shape_for_file(path: Path) -> tuple[int, int] | None:
    """Return (height, width) for the first page of the image."""
    try:
        with Image.open(path) as im:
            w, h = im.size
            return (h, w)
    except OSError:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root folder to walk (e.g. Keras-style rvl-cdip with train/ and test/).",
    )
    p.add_argument(
        "--ext",
        default=".tif",
        help="File extension to include (default: .tif).",
    )
    p.add_argument(
        "--min-count",
        type=int,
        default=3000,
        help="When plotting, only include shapes with at least this many images.",
    )
    p.add_argument(
        "--exclude-underscore-path",
        action="store_true",
        help="Skip files whose full path contains '_' (matches some notebook filters).",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print stats; do not show matplotlib figure.",
    )
    p.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="Save figure to this path instead of showing interactively.",
    )
    args = p.parse_args()

    root = args.root.resolve()
    ext = args.ext.lower()
    if not ext.startswith("."):
        ext = "." + ext

    counts: Counter[tuple[int, int]] = Counter()
    min_h = min_w = 10**9
    max_h = max_w = 0

    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if not name.lower().endswith(ext):
                continue
            fp = Path(dirpath) / name
            if args.exclude_underscore_path and "_" in str(fp):
                continue
            sh = shape_for_file(fp)
            if sh is None:
                continue
            h, w = sh
            counts[sh] += 1
            min_h, max_h = min(min_h, h), max(max_h, h)
            min_w, max_w = min(min_w, w), max(max_w, w)

    n_files = sum(counts.values())
    n_shapes = len(counts)
    print(f"Images counted: {n_files}")
    print(f"Distinct (H, W) shapes: {n_shapes}")
    print(f"Global height range: [{min_h}, {max_h}], width range: [{min_w}, {max_w}]")
    top = counts.most_common(15)
    print("Top 15 shapes by count:")
    for (h, w), c in top:
        print(f"  ({h}, {w}): {c}")

    filtered = {f"{h}_{w}": c for (h, w), c in counts.items() if c >= args.min_count}
    if not filtered:
        print(f"No shapes with count >= {args.min_count}; skip plot.")
        return

    if args.no_plot and args.save_plot is None:
        return

    import matplotlib.pyplot as plt

    items = sorted(filtered.items(), key=lambda kv: kv[1], reverse=True)
    x_labels = [kv[0] for kv in items]
    y_vals = [kv[1] for kv in items]

    fig, ax = plt.subplots(figsize=(12, 6))
    x_idx = range(len(x_labels))
    ax.bar(x_idx, y_vals)
    ax.set_xticks(list(x_idx))
    ax.set_xticklabels(x_labels, rotation=75, ha="right")
    ax.set_xlabel("height_width (pixels)")
    ax.set_ylabel("number of images")
    ax.set_title(f"Shapes with at least {args.min_count} images")
    fig.tight_layout()

    if args.save_plot:
        plt.savefig(args.save_plot, dpi=150)
        print(f"Saved plot to {args.save_plot}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
