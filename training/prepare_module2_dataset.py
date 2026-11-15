"""Prepare the ``Qwerty0193/genimage-processed`` layout for Module 2 training.

The Hugging Face ZIP extracts to::

    images/
        train/
            ai/<generator>/*.jpg
            nature/<generator>/*.jpg
        val/...
        test/...

``training/train_ai_detector.py`` expects one flat real folder and one
generator-subfolder per AI source::

    prepared/
        real/*.jpg
        ai/<generator>/*.jpg

This script builds that layout with hardlinks (falling back to copies when the
filesystem refuses).  It never downloads anything: point ``--source`` at an
already extracted ``images`` directory.

Typical use::

    python training/prepare_module2_dataset.py \
        --source datasets/module2_genimage/images \
        --output datasets/module2_genimage/prepared

    # Small deterministic subset for a pipeline check:
    python training/prepare_module2_dataset.py \
        --source datasets/module2_genimage/images \
        --output datasets/module2_genimage/smoke --smoke
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _place(source: Path, destination: Path, *, copy: bool) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare(
    source: Path,
    output: Path,
    *,
    copy: bool = False,
    smoke: bool = False,
    seed: int = 42,
) -> dict:
    real_found: list[Path] = []
    ai_found: dict[str, list[Path]] = {}

    for split in SPLITS:
        split_dir = source / split
        if not split_dir.is_dir():
            continue
        real_found.extend(_images(split_dir / "nature"))
        ai_root = split_dir / "ai"
        if not ai_root.is_dir():
            continue
        for generator_dir in sorted(path for path in ai_root.iterdir() if path.is_dir()):
            ai_found.setdefault(generator_dir.name, []).extend(_images(generator_dir))
        # AI images stored directly under ai/ are still usable, but the folder
        # name is the only generator label available.
        loose = sorted(
            path
            for path in ai_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if loose:
            ai_found.setdefault("unknown", []).extend(loose)

    if not real_found:
        raise FileNotFoundError(
            f"No real images found. Expected {source}/<split>/nature/... "
            "with split in train/val/test."
        )
    if not ai_found:
        raise FileNotFoundError(
            f"No AI images found. Expected {source}/<split>/ai/<generator>/..."
        )

    rng = random.Random(seed)
    if smoke:
        real_found = rng.sample(real_found, min(200, len(real_found)))
        ai_found = {
            generator: rng.sample(paths, min(100, len(paths)))
            for generator, paths in ai_found.items()
        }

    real_dir = output / "real"
    real_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    real_written = 0
    for path in real_found:
        name = path.name
        if name in used_names:
            name = f"{path.parent.name}__{name}"
        used_names.add(name)
        _place(path, real_dir / name, copy=copy)
        real_written += 1

    ai_written: dict[str, int] = {}
    for generator, paths in sorted(ai_found.items()):
        target = output / "ai" / generator
        target.mkdir(parents=True, exist_ok=True)
        for path in paths:
            _place(path, target / path.name, copy=copy)
        ai_written[generator] = len(paths)

    return {
        "source": str(source),
        "output": str(output),
        "mode": "smoke" if smoke else "full",
        "real": real_written,
        "ai": ai_written,
        "ai_total": sum(ai_written.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of hardlinking (use on filesystems without hardlinks).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Keep 200 real images and 100 per generator for a pipeline check.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = prepare(
        args.source, args.output, copy=args.copy, smoke=args.smoke, seed=args.seed
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
