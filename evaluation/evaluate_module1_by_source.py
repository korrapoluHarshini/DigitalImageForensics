"""Evaluate Module 1 image/pixel metrics separately for each source group.

Predictions are supplied by a CSV that must use maximum pixel probability as the
image score.  Raw thresholded masks and post-filtered highlighted masks are
reported separately.  If a probability map is supplied, the raw mask is derived
at the localization threshold and the filtered mask can be reproduced with the
deployed component filter.

Usage:
    .\\mvss_venv\\Scripts\\python.exe evaluation/evaluate_module1_by_source.py \\
        --manifest results/manifests/casia2_manifest.csv \\
        --predictions results/manifests/casia2_predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_manifest.evaluation import evaluate_per_source, write_evaluation_json
from dataset_manifest.io import load_manifest
from utils.config import settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=settings.data_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "evaluation" / "module1_by_source.json",
    )
    parser.add_argument("--image-threshold", type=float, default=settings.image_level_threshold)
    parser.add_argument(
        "--localization-threshold", type=float, default=settings.manipulation_threshold
    )
    parser.add_argument(
        "--uncertainty-margin", type=float, default=settings.image_uncertainty_margin
    )
    parser.add_argument(
        "--no-derive-filtered",
        action="store_true",
        help="Only use explicitly supplied filtered masks.",
    )
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    payload = evaluate_per_source(
        records,
        args.predictions,
        data_root=args.data_root,
        image_threshold=args.image_threshold,
        localization_threshold=args.localization_threshold,
        uncertainty_margin=args.uncertainty_margin,
        derive_filtered=not args.no_derive_filtered,
        manifest_path=args.manifest,
        predictions_path=args.predictions,
    )
    write_evaluation_json(payload, args.output)

    print("\n=== MODULE 1 BY SOURCE ===")
    print(
        f"Image threshold: {args.image_threshold:.2f} | "
        f"Localization threshold: {args.localization_threshold:.2f} | "
        f"Uncertainty: +/-{args.uncertainty_margin:.2f}"
    )
    for name, group in payload["source_groups"].items():
        image = group["image_level"]
        coverage = group["sample_coverage"]
        print(
            f"{name:20s} status={group['status']:13s} "
            f"evaluated={coverage['evaluated_samples']:4d} "
            f"accuracy={_fmt(image.get('accuracy'))} "
            f"fpr={_fmt(image.get('false_positive_rate'))} "
            f"auc={_fmt(image.get('roc_auc'))} "
            f"uncertain={_fmt(image.get('uncertain_rate'))}"
        )
    print(f"\nSaved: {args.output}")
    return 0


def _fmt(value) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
