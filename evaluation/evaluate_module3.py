"""Evaluate Module 3 (chat screenshot screening + OCR).

A labelled chat-screenshot dataset is required.  The script reports visual
localization metrics where ground-truth masks exist, an authentic-image
false-positive estimate, and OCR quality when reference transcripts are
provided.  Without the dataset it prints the expected layout and exits — no
metrics are fabricated.

Expected layout::

    manipulated/  img1.png, img2.png, ...
    masks/        img1.png, img2.png, ...           (same stems)
    authentic/    real1.png, ...                    (optional, for FPR)
    ocr_refs/     img1.txt, img2.txt, ...           (optional)

Usage:
    python evaluation/evaluate_module3.py --manipulated-dir ... --masks-dir ...
    python evaluation/evaluate_module3.py --manipulated-dir ... --masks-dir ... \
        --authentic-dir ... --ocr-references ... --detailed
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
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

from modules.chat_tampering import detect_chat_tampering
from utils.config import settings
from utils.image_utils import load_image_rgb
from utils.metrics import SegmentationAccumulator, image_level_metrics


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _collect(directory: Path) -> list[Path]:
    if not directory or not directory.exists():
        return []
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _find_mask(masks_dir: Path, image_path: Path) -> Path | None:
    for suffix in (".png", ".PNG", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        candidate = masks_dir / f"{image_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _ocr_similarity(reference: str, hypothesis: str) -> dict:
    ref = _normalise(reference)
    hyp = _normalise(hypothesis)
    matcher = difflib.SequenceMatcher(a=ref, b=hyp)
    return {
        "exact_match": bool(ref and ref == hyp),
        "character_similarity": float(matcher.ratio()),
        "reference_characters": len(ref),
        "hypothesis_characters": len(hyp),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manipulated-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--authentic-dir", type=Path, default=None)
    parser.add_argument("--ocr-references", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=settings.manipulation_threshold)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--authentic-limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detailed", action="store_true", help="Use overlapping-patch inference.")
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.evaluation_dir / "module3_metrics.json",
    )
    args = parser.parse_args()

    manipulated = _collect(args.manipulated_dir)
    if not manipulated or not args.masks_dir.exists():
        print(
            "Module 3 evaluation requires a labelled screenshot dataset.\n"
            "  manipulated images: "
            f"{len(manipulated)}\n"
            f"  masks directory exists: {args.masks_dir.exists()}\n\n"
            "Expected layout:\n"
            "  manipulated/<image>\n  masks/<same stem>.png\n"
            "  authentic/<image> (optional, FPR)\n"
            "  ocr_refs/<same stem>.txt (optional, OCR quality)\n\n"
            "No metrics are reported without ground-truth masks.",
            file=sys.stderr,
        )
        return 2

    if args.limit:
        manipulated = manipulated[: args.limit]

    accumulator = SegmentationAccumulator()
    per_image: list[dict] = []
    ocr_rows: list[dict] = []
    skipped: list[dict] = []

    for position, image_path in enumerate(manipulated, start=1):
        mask_path = _find_mask(args.masks_dir, image_path)
        if mask_path is None:
            skipped.append({"file": image_path.name, "reason": "mask missing"})
            continue
        ground_truth = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if ground_truth is None:
            skipped.append({"file": image_path.name, "reason": "mask unreadable"})
            continue
        try:
            image = load_image_rgb(image_path)
        except Exception as exc:
            skipped.append({"file": image_path.name, "reason": f"image unreadable: {exc}"})
            continue

        result = detect_chat_tampering(
            image,
            threshold=args.threshold,
            detailed=args.detailed,
            run_ocr=not args.no_ocr,
        )
        if result.mask_visual is None or result.manipulation is None:
            skipped.append({"file": image_path.name, "reason": result.error_message or "visual failure"})
            continue

        binary_mask = result.manipulation.binary_mask > 0
        if binary_mask.shape != ground_truth.shape:
            binary_mask = cv2.resize(
                binary_mask.astype(np.uint8),
                (ground_truth.shape[1], ground_truth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        counts = accumulator.add(binary_mask, ground_truth > 127)
        per_image.append(
            {
                "file": image_path.name,
                "threshold": args.threshold,
                "max_score": result.visual_score,
                "IoU": counts.iou,
                "Precision": counts.precision,
                "Recall": counts.recall,
                "F1": counts.f1,
            }
        )

        if args.ocr_references and result.ocr is not None and result.ocr.available:
            reference_path = args.ocr_references / f"{image_path.stem}.txt"
            if reference_path.exists():
                score = _ocr_similarity(
                    reference_path.read_text(encoding="utf-8", errors="ignore"),
                    result.ocr.full_text,
                )
                score["file"] = image_path.name
                ocr_rows.append(score)

        print(f"  {position}/{len(manipulated)}: {image_path.name}")

    if accumulator.images == 0:
        print("No screenshot could be evaluated.", file=sys.stderr)
        return 2

    localization = accumulator.summary()
    manipulated_scores = [row["max_score"] for row in per_image]

    authentic_scores: list[float] = []
    authentic_images = _collect(args.authentic_dir) if args.authentic_dir else []
    if authentic_images and args.authentic_limit:
        random.Random(args.seed).shuffle(authentic_images)
        for image_path in authentic_images[: args.authentic_limit]:
            try:
                image = load_image_rgb(image_path)
            except Exception:
                continue
            result = detect_chat_tampering(
                image,
                threshold=args.threshold,
                detailed=args.detailed,
                run_ocr=False,
            )
            if result.manipulation is not None:
                authentic_scores.append(result.visual_score)

    image_level: dict = {
        "manipulated_images": len(manipulated_scores),
        "manipulated_recall_at_threshold": float(
            np.mean(np.asarray(manipulated_scores) >= args.threshold)
        ),
        "authentic_sample_evaluated": len(authentic_scores),
        "authentic_sample_only": len(authentic_scores) < len(authentic_images),
    }
    if authentic_scores:
        scores = manipulated_scores + authentic_scores
        labels = [1] * len(manipulated_scores) + [0] * len(authentic_scores)
        image_level.update(image_level_metrics(scores, labels, args.threshold))
        image_level["note"] = (
            "Authentic FPR is an estimate from the reported sample. Screenshot tampering "
            "ground truth is independent of the CASIA training data."
        )
    else:
        image_level["note"] = "No authentic screenshots supplied; FPR not reported."

    ocr_quality = None
    if ocr_rows:
        ocr_frame = pd.DataFrame(ocr_rows)
        ocr_quality = {
            "screenshots_with_reference": int(len(ocr_frame)),
            "mean_character_similarity": float(ocr_frame["character_similarity"].mean()),
            "exact_match_rate": float(ocr_frame["exact_match"].mean()),
        }

    artifact = {
        "artifact": args.output.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "module": "Chat screenshot visual screening + OCR",
        "threshold": float(args.threshold),
        "threshold_provenance": "Same validation-selected MVSS-Net threshold as Module 1.",
        "localization": localization,
        "image_level": image_level,
        "ocr_quality": ocr_quality,
        "samples": {
            "manipulated_images": len(manipulated),
            "evaluated": accumulator.images,
            "skipped": len(skipped),
            "authentic_images_available": len(authentic_images),
        },
        "skipped": skipped[:25],
        "limitations": [
            "MVSS-Net is trained on CASIA photographs; chat screenshots are out of domain.",
            "OCR quality is reported only against supplied reference transcripts.",
            "Visual screening cannot verify message authenticity without trusted source records.",
        ],
    }

    settings.evaluation_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = args.output.with_name(args.output.stem + "_per_image.csv")
    pd.DataFrame(per_image).to_csv(per_image_path, index=False)
    if ocr_rows:
        pd.DataFrame(ocr_rows).to_csv(
            args.output.with_name(args.output.stem + "_ocr.csv"), index=False
        )
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\n=== MODULE 3 EVALUATION ===")
    print(
        f"Images: {localization['images']} | Global IoU: {localization['global_iou']:.4f} | "
        f"Mean image IoU: {localization['mean_image_iou']:.4f} | Mean F1: {localization['mean_image_f1']:.4f}"
    )
    if authentic_scores:
        print(
            f"Authentic sample: {len(authentic_scores)} | "
            f"FPR: {image_level.get('false_positive_rate'):.4f}"
        )
    if ocr_quality:
        print(
            f"OCR: mean character similarity {ocr_quality['mean_character_similarity']:.4f} | "
            f"exact match {ocr_quality['exact_match_rate']:.4f}"
        )
    print(f"\nSaved: {args.output}")
    print(f"Saved: {per_image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
