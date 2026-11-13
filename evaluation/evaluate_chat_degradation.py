"""Degradation stress test for Module 3 reference comparison.

The base synthetic evaluation (``evaluate_chat_reference.py``) compares a clean
reference render with a clean edited render.  Real screenshots are re-encoded by
messaging apps and may be resized, so this script repeats the *same* evaluation
after applying controlled degradations to the suspect, to the reference, or to
both.  Ground-truth text changes and masks stay fixed; only the image bytes
change.

This is still a **synthetic** test.  It measures robustness to compression and
rescaling of rendered screenshots, not performance on real captured chats.

Usage::

    python evaluation/evaluate_chat_degradation.py \
        --dataset D:\\ImageForensics_Datasets\\chat_synthetic_v1

Output: ``results/evaluation/module3_degradation_metrics.json``.
"""

from __future__ import annotations

import argparse
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

from evaluation.evaluate_chat_reference import (  # noqa: E402
    _region_matches_box,
    _text_change_matches,
)
from modules.chat_tampering.reference import STATUS_OK, compare_chat_screenshot  # noqa: E402
from utils.config import settings  # noqa: E402
from utils.image_utils import load_image_rgb  # noqa: E402


CONDITIONS = [
    {"name": "clean", "reference": None, "suspect": None},
    {"name": "suspect_jpeg85", "reference": None, "suspect": {"jpeg": 85}},
    {"name": "suspect_jpeg60", "reference": None, "suspect": {"jpeg": 60}},
    {"name": "suspect_jpeg40", "reference": None, "suspect": {"jpeg": 40}},
    {"name": "suspect_rescale50", "reference": None, "suspect": {"rescale": 0.5}},
    {"name": "both_jpeg60", "reference": {"jpeg": 60}, "suspect": {"jpeg": 60}},
]


def _jpeg(rgb: np.ndarray, quality: int) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _rescale(rgb: np.ndarray, scale: float) -> np.ndarray:
    height, width = rgb.shape[:2]
    small = cv2.resize(
        rgb,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def _degrade(rgb: np.ndarray, spec: dict | None) -> np.ndarray:
    if not spec:
        return rgb
    if "jpeg" in spec:
        return _jpeg(rgb, int(spec["jpeg"]))
    if "rescale" in spec:
        return _rescale(rgb, float(spec["rescale"]))
    raise ValueError(f"Unknown degradation: {spec}")


def _evaluate_condition(annotations: dict, dataset: Path, condition: dict, args) -> dict:
    true_positives = false_positives = false_negatives = 0
    text_expected = text_detected = text_matched = 0
    region_expected = region_detected = region_matched = 0
    authentic_pairs = authentic_with_difference = 0
    mismatch_count = 0
    rows: list[dict] = []

    for entry in annotations.get("conversations", []):
        suspect = load_image_rgb(dataset / entry["manipulated_image"])
        reference = load_image_rgb(dataset / entry["authentic_image"])
        mask = cv2.imread(str(dataset / entry["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        suspect = _degrade(suspect, condition["suspect"])
        reference = _degrade(reference, condition["reference"])

        comparison = compare_chat_screenshot(
            suspect,
            reference,
            pixel_threshold=args.pixel_threshold,
            min_region_area=args.min_region_area,
        )

        predicted = (comparison.change_mask > 0).astype(np.uint8)
        ground_truth = (mask > 127).astype(np.uint8)
        if predicted.shape != ground_truth.shape:
            predicted = cv2.resize(
                predicted,
                (ground_truth.shape[1], ground_truth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        tp = int(np.logical_and(predicted, ground_truth).sum())
        fp = int(np.logical_and(predicted, 1 - ground_truth).sum())
        fn = int(np.logical_and(1 - predicted, ground_truth).sum())
        true_positives += tp
        false_positives += fp
        false_negatives += fn

        matched_regions = 0
        for change in entry.get("changes", []):
            region_expected += 1
            if change.get("bbox") and any(
                _region_matches_box(region, change["bbox"]) for region in comparison.regions
            ):
                region_matched += 1
                matched_regions += 1
        region_detected += len(comparison.regions)

        used: set[int] = set()
        matched_text = 0
        for expected_change in entry.get("changes", []):
            text_expected += 1
            if _text_change_matches(expected_change, comparison.text_changes, used) is not None:
                text_matched += 1
                matched_text += 1
        text_detected += len(comparison.text_changes)

        authentic_pair = compare_chat_screenshot(
            reference,
            reference,
            pixel_threshold=args.pixel_threshold,
            min_region_area=args.min_region_area,
        )
        authentic_pairs += 1
        if (
            authentic_pair.regions
            or authentic_pair.text_changes
            or authentic_pair.changed_pixels > 0
        ):
            authentic_with_difference += 1
        if comparison.status != STATUS_OK:
            mismatch_count += 1

        rows.append(
            {
                "conversation": entry["conversation_id"],
                "expected_changes": len(entry.get("changes", [])),
                "detected_text_changes": len(comparison.text_changes),
                "matched_text_changes": matched_text,
                "detected_regions": len(comparison.regions),
                "matched_regions": matched_regions,
                "status": comparison.status,
            }
        )

    precision = true_positives / float(true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / float(true_positives + false_negatives) if true_positives + false_negatives else 0.0
    iou = true_positives / float(true_positives + false_positives + false_negatives) if true_positives + false_positives + false_negatives else 0.0
    text_precision = text_matched / float(text_detected) if text_detected else 0.0
    text_recall = text_matched / float(text_expected) if text_expected else 0.0
    region_recall = region_matched / float(region_expected) if region_expected else 0.0

    return {
        "condition": condition["name"],
        "reference_degradation": condition["reference"],
        "suspect_degradation": condition["suspect"],
        "conversations": len(rows),
        "pixel": {
            "iou": round(iou, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "true_positive_pixels": true_positives,
            "false_positive_pixels": false_positives,
            "false_negative_pixels": false_negatives,
        },
        "regions": {
            "expected": region_expected,
            "detected": region_detected,
            "matched": region_matched,
            "recall": round(region_recall, 6),
        },
        "text_changes": {
            "expected": text_expected,
            "detected": text_detected,
            "matched": text_matched,
            "precision": round(text_precision, 6),
            "recall": round(text_recall, 6),
        },
        "authentic_pairs": {
            "evaluated": authentic_pairs,
            "with_any_difference": authentic_with_difference,
            "false_positive_rate": round(authentic_with_difference / authentic_pairs, 6)
            if authentic_pairs
            else None,
        },
        "reference_mismatch_rate": round(mismatch_count / len(rows), 6) if rows else None,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pixel-threshold", type=float, default=25.0)
    parser.add_argument("--min-region-area", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.evaluation_dir / "module3_degradation_metrics.json",
    )
    args = parser.parse_args()

    annotations_path = args.dataset / "annotations.json"
    if not annotations_path.exists():
        print(f"annotations.json not found in {args.dataset}", file=sys.stderr)
        return 2
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    if args.limit:
        annotations = {
            **annotations,
            "conversations": annotations.get("conversations", [])[: args.limit],
        }
    if not annotations.get("conversations"):
        print("No conversations to evaluate.", file=sys.stderr)
        return 2

    started_rows = []
    for condition in CONDITIONS:
        print(f"\n--- condition: {condition['name']} ---")
        result = _evaluate_condition(annotations, args.dataset, condition, args)
        started_rows.append(result)
        print(
            f"  text recall {result['text_changes']['recall']:.4f} | "
            f"region recall {result['regions']['recall']:.4f} | "
            f"pixel IoU {result['pixel']['iou']:.4f} | "
            f"authentic FPR {result['authentic_pairs']['false_positive_rate']}"
        )

    clean = next(row for row in started_rows if row["condition"] == "clean")
    summary = {
        "artifact": "module3_degradation_metrics.json",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "module": "Chat screenshot reference comparison (degradation stress test)",
        "dataset": annotations.get("dataset"),
        "dataset_license": annotations.get("license"),
        "dataset_provenance": annotations.get("privacy"),
        "protocol": (
            "The same synthetic authentic/manipulated pairs are re-compared after controlled "
            "JPEG compression or rescaling. Only the suspect, or both images, are degraded; "
            "ground-truth text changes and masks are unchanged. Thresholds are fixed."
        ),
        "pixel_threshold": float(args.pixel_threshold),
        "min_region_area": int(args.min_region_area),
        "conditions": started_rows,
        "relative_text_recall_retention": {
            row["condition"]: round(
                row["text_changes"]["recall"] / clean["text_changes"]["recall"], 6
            )
            if clean["text_changes"]["recall"]
            else None
            for row in started_rows
        },
        "limitations": [
            "Synthetic renders only: real app re-rendering, anti-aliasing and unknown "
            "resampling are not represented.",
            "Ground-truth masks cover whole edited text boxes; pixel IoU is lowered by "
            "construction even on clean inputs.",
            "The stress test applies degradations to the reference as well, which is unusual "
            "unless the only available reference is itself a forwarded screenshot.",
            "No result here is evidence about real captured conversations.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    per_condition = args.output.with_name(args.output.stem + "_per_condition.csv")
    pd.DataFrame(
        [
            {
                "condition": row["condition"],
                "text_recall": row["text_changes"]["recall"],
                "text_precision": row["text_changes"]["precision"],
                "region_recall": row["regions"]["recall"],
                "pixel_iou": row["pixel"]["iou"],
                "authentic_fpr": row["authentic_pairs"]["false_positive_rate"],
                "reference_mismatch_rate": row["reference_mismatch_rate"],
            }
            for row in started_rows
        ]
    ).to_csv(per_condition, index=False)

    print(f"\nSaved: {args.output}")
    print(f"Saved: {per_condition}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
