#!/usr/bin/env python3
"""
Randomly sample up to N TIFFs per leaf directory, resize, and save as PNG.

Mirrors the notebook `generate_dataset` idea with fixes:
  - sample file *paths* (no fragile index arithmetic)
  - `skimage.resize` output_shape is (height, width) — rows then columns
  - `source_path` / `target_path` parameters instead of hard-coded volumes
  - skip macOS resource forks `._*`
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize


def _is_tif_candidate(path: Path) -> bool:
    s = path.suffix.lower()
    return s in (".tif", ".tiff") and not path.name.startswith("._")


def _resize_gray(img: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    if img.ndim == 3:
        img = img[..., 0] if img.shape[2] >= 1 else np.mean(img, axis=2)
    img = img.astype(np.float64)
    out = resize(
        img,
        (out_height, out_width),
        order=1,
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def generate_dataset(
    out_width: int,
    out_height: int,
    target_path: Path,
    source_path: Path,
    desired_size_per_category: int,
    *,
    seed: int | None = None,
) -> None:
    """
    Walk `source_path`, and for each directory that contains .tif files, pick up to
    `desired_size_per_category` files uniformly at random, resize to
    (out_height, out_width), write PNG under `target_path` preserving relative layout.

    `out_width` / `out_height` are the output image width and height in pixels.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    source_path = source_path.resolve()
    target_path = target_path.resolve()

    category = 0
    for root, _dirs, files in os.walk(source_path):
        root_p = Path(root)
        tif_files = [f for f in files if _is_tif_candidate(root_p / f)]
        if not tif_files:
            continue

        k = min(desired_size_per_category, len(tif_files))
        chosen = random.sample(tif_files, k)
        print(
            f"Directory {root_p} has {len(tif_files)} .tif files; "
            f"choosing {k} at random."
        )

        for fname in chosen:
            src = root_p / fname
            try:
                img = imread(src)
            except OSError as e:
                print(f"  skip (read error): {src} ({e})", file=sys.stderr)
                continue
            if img.size == 0:
                continue

            try:
                out_img = _resize_gray(img, out_height, out_width)
            except Exception as e:
                print(f"  skip (resize error): {src} ({e})", file=sys.stderr)
                continue

            rel = src.relative_to(source_path)
            dst = target_path / rel.with_suffix(".png")
            dst.parent.mkdir(parents=True, exist_ok=True)
            imsave(dst, out_img, check_contrast=False)

        print(f"  done category bucket {category} ({k} images written)")
        category += 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, required=True, help="Output width (pixels).")
    p.add_argument("--height", type=int, required=True, help="Output height (pixels).")
    p.add_argument("--source", type=Path, required=True, help="Keras-style TIFF tree root.")
    p.add_argument("--target", type=Path, required=True, help="Output root for PNGs.")
    p.add_argument(
        "--per-class",
        type=int,
        required=True,
        help="Max images to sample per leaf directory (class folder).",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility.")
    args = p.parse_args()

    generate_dataset(
        args.width,
        args.height,
        args.target,
        args.source,
        args.per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
