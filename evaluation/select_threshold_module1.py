"""Select the MVSS-Net localization threshold on the validation split only.

This script refuses to run on the test split. It reuses the prediction maps
already cached by the repository's validation experiments when available, so
the selection can be reproduced without another multi-hour CPU run.

Usage:
    python evaluation/select_threshold_module1.py
    python evaluation/select_threshold_module1.py --split validation --step 0.05
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import settings
from utils.metrics import SegmentationAccumulator


THRESHOLDS = [round(0.05 * index, 2) for index in range(1, 20)]  # 0.05 .. 0.95


def _cache_name(filename: str) -> str:
    return hashlib.sha1(str(filename).encode("utf-8")).hexdigest() + ".npy"


def _score_cache_dir(split: str) -> Path:
    if split == "validation":
        return settings.project_root / "results" / "validation_score_cache"
    return settings.project_root / "results" / "final_test" / "score_cache"


def _load_pairs() -> dict[str, Path]:
    pairs = pd.read_csv(settings.pairs_csv)
    mapping: dict[str, Path] = {}
    for row in pairs.itertuples():
        image_path = Path(str(row.image_path).replace("\\", "/"))
        mask_path = Path(str(row.mask_path).replace("\\", "/"))
        mapping[image_path.name] = settings.data_root / mask_path
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation", choices=["validation"])
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0, help="0 = all cached images")
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.evaluation_dir / "validation_threshold_selection.json",
    )
    args = parser.parse_args()

    if args.split != "validation":
        parser.error("Threshold selection is restricted to the validation split.")

    split_frame = pd.read_csv(settings.split_csv)
    tampered = split_frame[
        (split_frame["split"] == args.split) & (split_frame["class_name"] == "Tampered")
    ].copy()
    if tampered.empty:
        print("No tampered validation images found in the split CSV.", file=sys.stderr)
        return 2

    pairs = _load_pairs()
    cache_dir = _score_cache_dir(args.split)
    if not cache_dir.exists():
        print(
            f"No cached validation scores at {cache_dir}.\n"
            "Run the existing CPU evaluation first, e.g. "
            "python evaluate_final_test.py, then rerun this script.",
            file=sys.stderr,
        )
        return 2

    samples: list[tuple[np.ndarray, np.ndarray, str]] = []
    skipped: list[dict] = []
    for filename in tampered["filename"].astype(str):
        mask_path = pairs.get(filename)
        if mask_path is None or not mask_path.exists():
            skipped.append({"filename": filename, "reason": "mask not found"})
            continue
        cache_path = cache_dir / _cache_name(filename)
        if not cache_path.exists():
            skipped.append({"filename": filename, "reason": "score cache missing"})
            continue
        scores = np.load(cache_path).astype(np.float32)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            skipped.append({"filename": filename, "reason": "mask unreadable"})
            continue
        if scores.shape != mask.shape:
            scores = cv2.resize(scores, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        samples.append((scores, mask > 127, filename))
        if args.limit and len(samples) >= args.limit:
            break

    if not samples:
        print("No usable cached validation samples were found.", file=sys.stderr)
        return 2

    thresholds = sorted(
        {round(value, 2) for value in np.arange(0.05, 0.951, max(0.01, args.step))}
    )
    rows = []
    for threshold in thresholds:
        accumulator = SegmentationAccumulator()
        for scores, ground_truth, _filename in samples:
            accumulator.add(scores >= threshold, ground_truth)
        summary = accumulator.summary()
        rows.append({"threshold": float(threshold), **summary})

    frame = pd.DataFrame(rows)
    best = frame.sort_values(
        by=["mean_image_iou", "mean_image_f1"], ascending=False
    ).iloc[0]

    settings.evaluation_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    frame.to_csv(csv_path, index=False)

    artifact = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split,
        "samples_used": len(samples),
        "samples_skipped": len(skipped),
        "selection_rule": "maximum mean image IoU, tie-break by mean image F1",
        "selected_threshold": float(best["threshold"]),
        "selected_metrics": {
            key: (float(best[key]) if isinstance(best[key], (float, np.floating)) else int(best[key]))
            for key in ["images", "global_iou", "mean_image_iou", "micro_precision", "micro_recall", "micro_f1", "mean_image_f1"]
            if key in best
        },
        "score_source": str(cache_dir.relative_to(settings.project_root)).replace("\\", "/"),
        "dataset": "CASIA2.0 revised (validation split)",
        "note": (
            "Scores were cached by the repository's existing validation pipeline "
            "(BGR input, 512x512 resize, ImageNet normalisation). Selection used "
            "ground-truth CASIA masks and never the test split."
        ),
        "skipped": skipped[:20],
        "table": rows,
    }
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(frame[["threshold", "mean_image_iou", "mean_image_f1", "global_iou"]].to_string(index=False))
    print(f"\nSelected threshold: {artifact['selected_threshold']:.2f}")
    print(f"Saved: {args.output}")
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
