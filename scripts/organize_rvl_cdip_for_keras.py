#!/usr/bin/env python3
"""
Organize extracted RVL-CDIP images into a Keras-friendly tree:

  <target>/train/{0..15}/<image files...>
  <target>/test/{0..15}/<image files...>

Reads official label files (path + class id per line). By default copies files;
use --move only if you intend to relocate images out of the original images/ tree.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_label_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line:
        return None
    parts = line.rsplit(None, 1)
    if len(parts) != 2:
        return None
    rel_path, category = parts[0], parts[1].strip()
    return rel_path, category


def process_label_file(
    label_file: Path,
    split_name: str,
    images_dir: Path,
    target_root: Path,
    *,
    move: bool = False,
    dry_run: bool = False,
) -> int:
    lines = label_file.read_text(encoding="utf-8", errors="replace").splitlines()
    n = 0
    for line in lines:
        parsed = parse_label_line(line)
        if parsed is None:
            continue
        rel_path, category = parsed
        image_name = Path(rel_path).name
        src = images_dir / rel_path
        dest_dir = target_root / split_name / category
        dest = dest_dir / image_name

        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            if not src.is_file():
                raise FileNotFoundError(f"Missing source image: {src}")
            if dest.exists():
                continue
            if move:
                shutil.move(str(src), str(dest))
            else:
                shutil.copy2(src, dest)
        n += 1

    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Extracted dataset root (contains images/ and labels/).",
    )
    p.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Output root (will contain train/ and test/ with class subfolders 0-15).",
    )
    p.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copy (destructive on the original tree).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse labels only; do not create dirs or touch files.",
    )
    args = p.parse_args()

    source = args.source.resolve()
    target = args.target.resolve()
    images_dir = source / "images"
    labels_dir = source / "labels"

    for name in ("images", "labels"):
        if not (source / name).is_dir():
            raise SystemExit(f"Expected directory missing: {source / name}")

    pairs = [
        (labels_dir / "train.txt", "train"),
        (labels_dir / "test.txt", "test"),
    ]
    for lf, split in pairs:
        if not lf.is_file():
            raise SystemExit(f"Label file not found: {lf}")
        count = process_label_file(
            lf,
            split,
            images_dir,
            target,
            move=args.move,
            dry_run=args.dry_run,
        )
        print(f"{lf.name} -> {split}/ : {count} entries processed")


if __name__ == "__main__":
    main()
