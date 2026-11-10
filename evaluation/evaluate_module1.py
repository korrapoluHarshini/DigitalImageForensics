"""Evaluate Module 1 (MVSS-Net) on the CASIA2.0 test or validation split.

The threshold is never selected here.  It must come from the validation-only
selector (``select_threshold_module1.py``); the default is read from
``utils.config.settings.manipulation_threshold``.

Prediction maps cached by the repository's earlier experiments are reused when
present, so pixel-level metrics can be reproduced without re-running the model.
An authentic-image sample is used for the false-positive-rate estimate; the
sample size is reported explicitly.

Usage:
    python evaluation/evaluate_module1.py --split test
    python evaluation/evaluate_module1.py --split test --authentic-limit 20
    python evaluation/evaluate_module1.py --split validation --authentic-limit 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.image_manipulation import detect_image_manipulation
from utils.config import settings
from utils.image_utils import load_image_rgb
from utils.metrics import SegmentationAccumulator, image_level_metrics, roc_auc


def _cache_name(filename: str) -> str:
    return hashlib.sha1(str(filename).encode("utf-8")).hexdigest() + ".npy"


def _existing_cache_dirs(split: str) -> list[Path]:
    if split == "validation":
        return [
            settings.project_root / "results" / "validation_score_cache",
            settings.evaluation_dir / "score_cache",
        ]
    return [
        settings.project_root / "results" / "final_test" / "score_cache",
        settings.evaluation_dir / "score_cache",
    ]


def _read_cached_scores(filename: str, split: str) -> np.ndarray | None:
    cache_name = _cache_name(filename)
    for directory in _existing_cache_dirs(split):
        path = directory / cache_name
        if path.exists():
            return np.load(path).astype(np.float32)
    return None


def _write_cached_scores(filename: str, scores: np.ndarray) -> None:
    cache_dir = settings.evaluation_dir / "score_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / _cache_name(filename), scores.astype(np.float32))


def _load_pairs() -> dict[str, Path]:
    pairs = pd.read_csv(settings.pairs_csv)
    mapping: dict[str, Path] = {}
    for row in pairs.itertuples():
        image_path = Path(str(row.image_path).replace("\\", "/"))
        mask_path = Path(str(row.mask_path).replace("\\", "/"))
        mapping[image_path.name] = settings.data_root / mask_path
    return mapping


def _predict(path: Path, threshold: float, predictor) -> tuple[np.ndarray, float]:
    image = load_image_rgb(path)
    result = detect_image_manipulation(image, threshold=threshold, predictor=predictor)
    if result.status != "ok" or result.probability_map is None:
        raise RuntimeError(result.error_message or "inference failed")
    return result.probability_map, float(result.image_score)


def _validation_threshold_provenance() -> dict:
    path = settings.evaluation_dir / "validation_threshold_selection.json"
    if not path.exists():
        return {
            "source": "missing",
            "note": "Run evaluation/select_threshold_module1.py to reproduce threshold selection.",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"source": "unreadable"}
    return {
        "source": path.name,
        "split": data.get("split"),
        "selected_threshold": data.get("selected_threshold"),
        "samples_used": data.get("samples_used"),
        "selection_rule": data.get("selection_rule"),
        "generated_utc": data.get("generated_utc"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--threshold", type=float, default=settings.manipulation_threshold)
    parser.add_argument("--limit", type=int, default=0, help="0 = all tampered images")
    parser.add_argument(
        "--authentic-limit",
        type=int,
        default=20,
        help="Authentic images sampled for the FPR estimate (0 = skip). CPU inference is slow.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Only use cached maps; fail if any required map is missing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to results/evaluation/module1_<split>_metrics.json",
    )
    args = parser.parse_args()

    output = args.output or settings.evaluation_dir / f"module1_{args.split}_metrics.json"
    settings.evaluation_dir.mkdir(parents=True, exist_ok=True)

    split_frame = pd.read_csv(settings.split_csv)
    pairs = _load_pairs()

    tampered = split_frame[
        (split_frame["split"] == args.split) & (split_frame["class_name"] == "Tampered")
    ].copy()
    authentic = split_frame[
        (split_frame["split"] == args.split) & (split_frame["class_name"] == "Authentic")
    ].copy()
    if tampered.empty:
        print("No tampered images found for this split.", file=sys.stderr)
        return 2

    predictor = None
    if not args.no_inference:
        from modules.image_manipulation.model import get_default_predictor

        predictor = get_default_predictor()

    # ---------------------------------------------------------------
    # Pixel-level localization on tampered images with masks
    # ---------------------------------------------------------------
    accumulator = SegmentationAccumulator()
    per_image: list[dict] = []
    skipped: list[dict] = []
    cached_count = 0
    inferred_count = 0

    for count, filename in enumerate(tampered["filename"].astype(str), start=1):
        if args.limit and count > args.limit:
            break
        mask_path = pairs.get(filename)
        if mask_path is None or not mask_path.exists():
            skipped.append({"filename": filename, "reason": "ground-truth mask missing"})
            continue
        ground_truth = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if ground_truth is None:
            skipped.append({"filename": filename, "reason": "mask unreadable"})
            continue

        scores = _read_cached_scores(filename, args.split)
        if scores is not None:
            cached_count += 1
        else:
            if args.no_inference or predictor is None:
                skipped.append({"filename": filename, "reason": "prediction not cached"})
                continue
            image_path = settings.dataset_dir / "Tp" / filename
            if not image_path.exists():
                skipped.append({"filename": filename, "reason": "image missing"})
                continue
            scores, _ = _predict(image_path, args.threshold, predictor)
            _write_cached_scores(filename, scores)
            inferred_count += 1

        if scores.shape != ground_truth.shape:
            scores = cv2.resize(
                scores,
                (ground_truth.shape[1], ground_truth.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        counts = accumulator.add(scores >= args.threshold, ground_truth > 127)
        per_image.append(
            {
                "filename": filename,
                "threshold": args.threshold,
                "max_probability": float(scores.max()),
                "IoU": counts.iou,
                "Precision": counts.precision,
                "Recall": counts.recall,
                "F1": counts.f1,
            }
        )
        if count % 100 == 0:
            print(f"  processed {count} tampered images...")

    if accumulator.images == 0:
        print("No tampered images could be evaluated.", file=sys.stderr)
        return 2

    localization = accumulator.summary()

    # ---------------------------------------------------------------
    # Authentic sample for the false-positive rate
    # ---------------------------------------------------------------
    authentic_limit = max(0, args.authentic_limit)
    sampled_authentic: list[str] = []
    authentic_scores: list[float] = []
    authentic_inferred = 0
    if authentic_limit:
        pool = list(authentic["filename"].astype(str))
        random.Random(args.seed).shuffle(pool)
        sampled_authentic = pool[:authentic_limit]
        for filename in sampled_authentic:
            scores = _read_cached_scores(filename, args.split)
            if scores is None:
                if args.no_inference or predictor is None:
                    continue
                image_path = settings.dataset_dir / "Au" / filename
                if not image_path.exists():
                    continue
                scores, _ = _predict(image_path, args.threshold, predictor)
                _write_cached_scores(filename, scores)
                authentic_inferred += 1
            authentic_scores.append(float(scores.max()))
            print(f"  authentic sample: {len(authentic_scores)}/{len(sampled_authentic)}")

    tampered_scores = [row["max_probability"] for row in per_image]
    image_level: dict = {
        "tampered_images": len(tampered_scores),
        "tampered_recall_at_threshold": float(
            np.mean(np.asarray(tampered_scores) >= args.threshold)
        )
        if tampered_scores
        else None,
        "authentic_sample_requested": authentic_limit,
        "authentic_sample_evaluated": len(authentic_scores),
        "authentic_sample_only": len(authentic_scores) < len(authentic),
    }
    if authentic_scores:
        all_scores = tampered_scores + authentic_scores
        all_labels = [1] * len(tampered_scores) + [0] * len(authentic_scores)
        image_level.update(image_level_metrics(all_scores, all_labels, args.threshold))
        image_level["roc_auc"] = float(roc_auc(np.asarray(all_scores), np.asarray(all_labels)))
        image_level["note"] = (
            "Image-level scores are maximum pixel probabilities. Authentic FPR is an "
            "estimate from the reported sample, not the full authentic split."
        )
    else:
        image_level["note"] = (
            "Authentic sample skipped (--authentic-limit 0); no false-positive rate is reported."
        )

    # ---------------------------------------------------------------
    # Save artifacts
    # ---------------------------------------------------------------
    per_image_path = output.with_name(output.stem + "_per_image.csv")
    pd.DataFrame(per_image).to_csv(per_image_path, index=False)

    artifact = {
        "artifact": output.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "module": "MVSS-Net image manipulation localization",
        "dataset": "CASIA2.0 revised",
        "split": args.split,
        "threshold": float(args.threshold),
        "threshold_provenance": _validation_threshold_provenance(),
        "localization": localization,
        "image_level": image_level,
        "samples": {
            "tampered_evaluated": len(per_image),
            "tampered_total_in_split": int(len(tampered)),
            "authentic_total_in_split": int(len(authentic)),
            "cached_prediction_maps": cached_count,
            "inferred_prediction_maps": inferred_count,
            "authentic_inferred": authentic_inferred,
            "skipped": len(skipped),
        },
        "skipped": skipped[:25],
        "provenance": (
            "Localization threshold selected on the validation split only. Pixel metrics "
            "are computed against CASIA2.0 ground-truth masks. Cached maps were produced by "
            "the repository's existing BGR/512x512/ImageNet-normalisation pipeline."
        ),
        "per_image_csv": per_image_path.name,
    }
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\n=== MODULE 1 EVALUATION ===")
    print(f"Split: {args.split} | threshold: {args.threshold:.2f}")
    print(
        f"Tampered images: {localization['images']} | cached maps: {cached_count} | inferred: {inferred_count}"
    )
    print(
        f"Global IoU: {localization['global_iou']:.4f} | "
        f"Mean image IoU: {localization['mean_image_iou']:.4f} | "
        f"Micro F1: {localization['micro_f1']:.4f} | "
        f"Mean image F1: {localization['mean_image_f1']:.4f}"
    )
    if authentic_scores:
        print(
            f"Authentic sample: {len(authentic_scores)} | "
            f"FPR: {image_level.get('false_positive_rate'):.4f} | "
            f"Tampered recall: {image_level.get('recall'):.4f} | "
            f"AUC: {image_level.get('roc_auc'):.4f}"
        )
    else:
        print("Authentic FPR: not evaluated (no authentic sample).")
    print(f"\nSaved: {output}")
    print(f"Saved: {per_image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
