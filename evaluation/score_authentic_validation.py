"""Score authentic validation images with MVSS-Net for image-level calibration.

Module 1 localises manipulation with a pixel probability map.  The image-level
decision uses the maximum pixel probability, but that score must be calibrated
against *authentic* validation images as well as tampered ones.  No authentic
validation score cache existed, so this script produces one.

It is resumable: existing per-image ``.npy`` files are reused.  The sample order
is deterministic (shuffled with ``--seed``), so the same subset is selected on
every run.

Usage:
    python evaluation/score_authentic_validation.py --limit 400
    python evaluation/score_authentic_validation.py            # all 1124
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import settings
from utils.image_utils import load_image_rgb


def cache_name(filename: str) -> str:
    return hashlib.sha1(str(filename).encode("utf-8")).hexdigest() + ".npy"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation", choices=["validation"])
    parser.add_argument("--limit", type=int, default=400, help="0 = every authentic validation image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=settings.evaluation_dir / "image_level_score_cache" / "authentic_validation",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=settings.evaluation_dir / "authentic_validation_scores.csv",
    )
    args = parser.parse_args()

    if args.split != "validation":
        parser.error("Image-level calibration is restricted to the validation split.")

    split_frame = pd.read_csv(settings.split_csv)
    authentic = split_frame[
        (split_frame["split"] == args.split) & (split_frame["class_name"] == "Authentic")
    ]["filename"].astype(str).tolist()
    if not authentic:
        print("No authentic validation images found.", file=sys.stderr)
        return 2

    ordered = list(authentic)
    random.Random(args.seed).shuffle(ordered)
    if args.limit:
        ordered = ordered[: args.limit]

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: dict[str, dict] = {}
    if args.csv.exists():
        try:
            for row in pd.read_csv(args.csv).itertuples():
                existing_rows[str(row.filename)] = {
                    "filename": str(row.filename),
                    "max_probability": float(row.max_probability),
                    "mean_probability": float(row.mean_probability),
                    "elapsed_seconds": float(getattr(row, "elapsed_seconds", 0.0)),
                }
        except Exception:
            existing_rows = {}

    todo = [name for name in ordered if not (args.cache_dir / cache_name(name)).exists()]
    print(
        f"Authentic validation pool: {len(authentic)} | sample requested: {len(ordered)} | "
        f"cached: {len(ordered) - len(todo)} | to run: {len(todo)}",
        flush=True,
    )
    if not todo:
        print("Nothing to do; all requested scores are cached.", flush=True)
        return 0

    from modules.image_manipulation.model import get_default_predictor

    predictor = get_default_predictor()
    started_all = time.perf_counter()
    rows = list(existing_rows.values())

    for position, filename in enumerate(ordered, start=1):
        target = args.cache_dir / cache_name(filename)
        if target.exists():
            if filename not in existing_rows:
                scores = np.load(target).astype(np.float32)
                rows.append(
                    {
                        "filename": filename,
                        "max_probability": float(scores.max()),
                        "mean_probability": float(scores.mean()),
                        "elapsed_seconds": 0.0,
                    }
                )
            continue

        image_path = settings.dataset_dir / "Au" / filename
        if not image_path.exists():
            print(f"  missing: {filename}", flush=True)
            continue

        started = time.perf_counter()
        try:
            image = load_image_rgb(image_path)
            scores, _meta = predictor.predict_probability(image)
        except Exception as exc:
            print(f"  failed: {filename}: {exc}", flush=True)
            continue
        elapsed = time.perf_counter() - started
        np.save(target, scores.astype(np.float32))
        rows.append(
            {
                "filename": filename,
                "max_probability": float(scores.max()),
                "mean_probability": float(scores.mean()),
                "elapsed_seconds": round(elapsed, 3),
            }
        )
        print(
            f"  {position}/{len(ordered)} {filename} max={float(scores.max()):.4f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )

        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(args.csv, index=False)

    frame = pd.DataFrame(rows).drop_duplicates(subset=["filename"], keep="last")
    frame.to_csv(args.csv, index=False)

    elapsed_total = time.perf_counter() - started_all
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split,
        "seed": args.seed,
        "requested": len(ordered),
        "scored": int(len(frame)),
        "elapsed_seconds": round(elapsed_total, 1),
        "cache_dir": str(args.cache_dir.relative_to(settings.project_root)).replace("\\", "/"),
        "csv": args.csv.name,
        "score_definition": "maximum pixel probability from the official single-scale MVSS-Net inference",
    }
    (args.csv.with_suffix(".manifest.json")).write_text(
        __import__("json").dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Scored {len(frame)} authentic validation images in {elapsed_total / 60:.1f} min.")
    print(f"Saved: {args.csv}")
    print(f"Saved: {args.cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
