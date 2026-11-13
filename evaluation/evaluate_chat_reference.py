"""Evaluate Module 3 *with a trusted reference screenshot*.

This script measures the reference-comparison pipeline (alignment, pixel-change
regions and OCR text differences) on a dataset with ground-truth masks and an
annotation file.  It is independent of Module 1's MVSS-Net: no forgery model is
used here, so the metrics isolate the value of having an original reference.

Expected layout (produced by ``evaluation/build_synthetic_chat_dataset.py``)::

    annotations.json
    authentic/<conversation>.png
    manipulated/<conversation>_edited.png
    masks/<conversation>_edited.png
    ocr_refs/<conversation>_edited.txt

Usage::

    python evaluation/evaluate_chat_reference.py \
        --dataset D:\\ImageForensics_Datasets\\chat_synthetic_v1

Without ``annotations.json`` the script exits with instructions and reports no
metrics — it never invents ground truth.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.chat_tampering.reference import STATUS_OK, compare_chat_screenshot
from utils.config import settings
from utils.image_utils import load_image_rgb


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _similarity(first: str, second: str) -> float:
    return float(
        difflib.SequenceMatcher(a=_normalise(first).lower(), b=_normalise(second).lower()).ratio()
    )


def _box_iou(first, second) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area_first = max(1, (first[2] - first[0]) * (first[3] - first[1]))
    area_second = max(1, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / float(area_first + area_second - intersection)


def _region_matches_box(region, box, padding: int = 4) -> bool:
    x1 = max(0, box[0] - padding)
    y1 = max(0, box[1] - padding)
    x2 = box[2] + padding
    y2 = box[3] + padding
    if _box_iou(region.bbox, (x1, y1, x2, y2)) >= 0.10:
        return True
    center_x = (region.bbox[0] + region.bbox[2]) / 2.0
    center_y = (region.bbox[1] + region.bbox[3]) / 2.0
    return x1 <= center_x <= x2 and y1 <= center_y <= y2


def _empty_kind_counts() -> dict:
    return {}


def _update_counts(counts: dict, kind: str, matched: bool) -> None:
    entry = counts.setdefault(kind, {"expected": 0, "matched": 0})
    entry["expected"] += 1
    entry["matched"] += int(bool(matched))


def _text_change_matches(expected: dict, detected_changes: list, used: set[int]) -> int | None:
    best_id = None
    best_score = 0.0
    for change in detected_changes:
        if change.change_id in used:
            continue
        if change.kind != expected["kind"]:
            continue
        reference_score = _similarity(expected["original"], change.reference_text)
        suspect_score = _similarity(expected["edited"], change.suspect_text)
        score = max(reference_score, suspect_score)
        expected_box = expected.get("bbox")
        if expected_box:
            for box in (change.reference_bbox, change.suspect_bbox):
                if box and _box_iou(box, expected_box) >= 0.05:
                    score = max(score, 0.70)
        if score > best_score:
            best_score = score
            best_id = change.change_id
    if best_id is None or best_score < 0.45:
        return None
    used.add(best_id)
    return best_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pixel-threshold", type=float, default=25.0)
    parser.add_argument("--min-region-area", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.evaluation_dir / "module3_reference_metrics.json",
    )
    args = parser.parse_args()

    annotations_path = args.dataset / "annotations.json"
    if not annotations_path.exists():
        print(
            "Reference-based evaluation needs annotations.json with authentic/manipulated "
            "pairs, ground-truth masks and expected text changes.\n"
            f"  looked for: {annotations_path}\n\n"
            "Generate the controlled synthetic set with:\n"
            "  python evaluation/build_synthetic_chat_dataset.py --conversations 8\n"
            "No metrics are reported without ground truth.",
            file=sys.stderr,
        )
        return 2

    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    conversations = annotations.get("conversations", [])
    if args.limit:
        conversations = conversations[: args.limit]
    if not conversations:
        print("annotations.json contains no conversations.", file=sys.stderr)
        return 2

    true_positives = false_positives = false_negatives = 0
    region_expected = region_detected = region_matched = 0
    kind_counts: dict = _empty_kind_counts()
    text_expected = text_detected = text_matched = 0
    rows: list[dict] = []
    skipped: list[dict] = []

    authentic_pairs = authentic_with_difference = 0
    mismatch_count = 0

    for position, entry in enumerate(conversations, start=1):
        suspicious_path = args.dataset / entry["manipulated_image"]
        reference_path = args.dataset / entry["authentic_image"]
        mask_path = args.dataset / entry["mask"]

        try:
            suspect = load_image_rgb(suspicious_path)
            reference = load_image_rgb(reference_path)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"conversation": entry.get("conversation_id"), "reason": str(exc)})
            continue
        if mask is None:
            skipped.append(
                {"conversation": entry.get("conversation_id"), "reason": "mask unreadable"}
            )
            continue

        comparison = compare_chat_screenshot(
            suspect,
            reference,
            pixel_threshold=args.pixel_threshold,
            min_region_area=args.min_region_area,
        )

        # ---- pixel-level: predicted change mask vs ground-truth edit mask ----
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
        pixel_iou = tp / float(tp + fp + fn) if (tp + fp + fn) else 0.0
        pixel_precision = tp / float(tp + fp) if (tp + fp) else 0.0
        pixel_recall = tp / float(tp + fn) if (tp + fn) else 0.0

        # ---- region-level: each ground-truth edit box vs predicted regions ----
        matched_regions = 0
        for change in entry.get("changes", []):
            region_expected += 1
            if change.get("bbox") and any(
                _region_matches_box(region, change["bbox"]) for region in comparison.regions
            ):
                region_matched += 1
                matched_regions += 1
        region_detected += len(comparison.regions)

        # ---- text-level: expected changes vs detected OCR differences ----
        used: set[int] = set()
        matched_text = 0
        for expected_change in entry.get("changes", []):
            text_expected += 1
            match_id = _text_change_matches(expected_change, comparison.text_changes, used)
            matched = match_id is not None
            _update_counts(kind_counts, expected_change["kind"], matched)
            if matched:
                matched_text += 1
                text_matched += 1
        text_detected += len(comparison.text_changes)

        # ---- false positives on the authentic pair itself ----
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
                "profile": entry.get("profile"),
                "status": comparison.status,
                "finding": comparison.finding,
                "alignment_correlation": (
                    round(comparison.alignment.correlation, 4)
                    if comparison.alignment
                    else None
                ),
                "expected_changes": len(entry.get("changes", [])),
                "detected_text_changes": len(comparison.text_changes),
                "matched_text_changes": matched_text,
                "expected_regions": len(entry.get("changes", [])),
                "detected_regions": len(comparison.regions),
                "matched_regions": matched_regions,
                "pixel_iou": round(pixel_iou, 6),
                "pixel_precision": round(pixel_precision, 6),
                "pixel_recall": round(pixel_recall, 6),
                "changed_pixel_ratio": round(comparison.changed_pixel_ratio, 6),
                "authentic_pair_false_positive": bool(
                    authentic_pair.regions
                    or authentic_pair.text_changes
                    or authentic_pair.changed_pixels > 0
                ),
            }
        )
        print(f"  {position}/{len(conversations)}: {entry['conversation_id']}")

    if not rows:
        print("No conversation could be evaluated.", file=sys.stderr)
        return 2

    global_iou = (
        true_positives / float(true_positives + false_positives + false_negatives)
        if (true_positives + false_positives + false_negatives)
        else 0.0
    )
    global_precision = (
        true_positives / float(true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    global_recall = (
        true_positives / float(true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )
    global_f1 = (
        2 * global_precision * global_recall / (global_precision + global_recall)
        if (global_precision + global_recall)
        else 0.0
    )

    text_precision = text_matched / float(text_detected) if text_detected else 0.0
    text_recall = text_matched / float(text_expected) if text_expected else 0.0
    region_precision = region_matched / float(region_detected) if region_detected else 0.0
    region_recall = region_matched / float(region_expected) if region_expected else 0.0

    artifact = {
        "artifact": args.output.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "module": "Chat screenshot reference comparison",
        "dataset": annotations.get("dataset"),
        "dataset_license": annotations.get("license"),
        "dataset_provenance": annotations.get("privacy"),
        "protocol": (
            "Synthetic controlled pairs. The authentic render is used as the reference for "
            "its edited copy. All comparisons use fixed, pre-declared thresholds; no test "
            "image influenced threshold selection."
        ),
        "pixel_threshold": float(args.pixel_threshold),
        "min_region_area": int(args.min_region_area),
        "samples": {
            "conversations": len(rows),
            "conversations_available": len(conversations),
            "skipped": len(skipped),
            "text_changes_expected": text_expected,
            "text_changes_detected": text_detected,
            "regions_expected": region_expected,
            "regions_detected": region_detected,
        },
        "pixel_localization": {
            "global_iou": round(global_iou, 6),
            "precision": round(global_precision, 6),
            "recall": round(global_recall, 6),
            "f1": round(global_f1, 6),
            "true_positive_pixels": true_positives,
            "false_positive_pixels": false_positives,
            "false_negative_pixels": false_negatives,
        },
        "region_localization": {
            "matched": region_matched,
            "precision": round(region_precision, 6),
            "recall": round(region_recall, 6),
            "note": "A region matches when IoU >= 0.10 or its centre falls inside the edited box.",
        },
        "text_change_detection": {
            "matched": text_matched,
            "precision": round(text_precision, 6),
            "recall": round(text_recall, 6),
            "per_kind": {
                kind: {
                    **counts,
                    "recall": round(counts["matched"] / counts["expected"], 4)
                    if counts["expected"]
                    else None,
                }
                for kind, counts in sorted(kind_counts.items())
            },
        },
        "authentic_pairs": {
            "evaluated": authentic_pairs,
            "with_any_difference": authentic_with_difference,
            "false_positive_rate": round(authentic_with_difference / authentic_pairs, 6)
            if authentic_pairs
            else None,
            "note": "Identical reference/suspect pairs should yield no differences.",
        },
        "reference_mismatch_rate": round(mismatch_count / len(rows), 6),
        "skipped": skipped,
        "limitations": [
            "Synthetic screenshots are rendered, not captured from a real messaging app; "
            "real app re-rendering and compression noise are not represented.",
            "Ground-truth masks mark whole edited text boxes while pixel predictions only "
            "mark pixels that actually changed, which lowers pixel IoU by construction.",
            "OCR differences depend on the installed OCR backend and its recognition errors.",
            "A reference comparison cannot establish which screenshot is the original.",
        ],
    }

    settings.evaluation_dir.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    per_conversation = args.output.with_name(args.output.stem + "_per_conversation.csv")
    pd.DataFrame(rows).to_csv(per_conversation, index=False)

    print("\n=== MODULE 3 REFERENCE COMPARISON ===")
    print(
        f"Conversations: {len(rows)} | pixel IoU: {global_iou:.4f} | "
        f"region recall: {region_recall:.4f} | text recall: {text_recall:.4f}"
    )
    print(
        f"Authentic pairs with any difference: {authentic_with_difference}/{authentic_pairs}"
    )
    print(f"\nSaved: {args.output}")
    print(f"Saved: {per_conversation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
