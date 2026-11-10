"""Compare candidate image-level scoring rules on validation data only.

Module 1 currently derives its image-level Authentic/Tampered score from the
**maximum** MVSS-Net pixel probability.  A single unusual pixel could in
principle dominate that score, so this script compares it against alternatives
on the validation split:

* ``max_pixel``               - current method (one score per image)
* ``mean_top_k``              - mean of the ``k`` highest pixel probabilities
* ``area_fraction_raw``       - fraction of pixels at/above the localization
                                threshold (0.40)
* ``area_fraction_retained``  - same, after the application's minimum-area
                                component filter

Only validation probability maps are read.  The test split is never touched.
The script does **not** modify the configured thresholds: the deployed
threshold (0.42) and uncertainty margin (0.13) remain unchanged.  AUC is the
primary comparison because it is threshold-independent; operating points are
also reported at the same ``--max-fpr`` constraint used by the threshold
selector.

Usage:
    python evaluation/compare_image_level_scores.py
    python evaluation/compare_image_level_scores.py --bootstrap 2000
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


TOP_K_VALUES = (1, 5, 10, 25, 50, 100, 250, 500, 1000)
MAX_REGIONS = 25
MIN_REGION_AREA_RATIO = 0.0002


def cache_name(filename: str) -> str:
    return hashlib.sha1(str(filename).encode("utf-8")).hexdigest() + ".npy"


def _load_maps(
    filenames: list[str], cache_dir: Path, label: str
) -> tuple[list[np.ndarray], list[str], list[str]]:
    maps: list[np.ndarray] = []
    kept: list[str] = []
    missing: list[str] = []
    for filename in filenames:
        path = cache_dir / cache_name(filename)
        if not path.exists():
            missing.append(filename)
            continue
        try:
            array = np.load(path)
        except Exception:
            missing.append(filename)
            continue
        maps.append(np.asarray(array, dtype=np.float32))
        kept.append(filename)
    print(
        f"{label}: loaded {len(maps)}/{len(filenames)} probability maps "
        f"({len(missing)} missing)",
        flush=True,
    )
    return maps, kept, missing


def _retained_area_fraction(
    probability_map: np.ndarray,
    pixel_threshold: float,
    min_region_area_ratio: float = MIN_REGION_AREA_RATIO,
    max_regions: int = MAX_REGIONS,
) -> float:
    """Fraction of pixels inside components that pass the app's area filter."""
    total_pixels = int(probability_map.size)
    if total_pixels == 0:
        return 0.0
    mask = (probability_map >= pixel_threshold).astype(np.uint8)
    if not mask.any():
        return 0.0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    min_pixels = max(16, int(round(total_pixels * max(0.0, min_region_area_ratio))))
    areas = [
        int(stats[label][cv2.CC_STAT_AREA])
        for label in range(1, count)
        if int(stats[label][cv2.CC_STAT_AREA]) >= min_pixels
    ]
    areas.sort(reverse=True)
    retained = sum(areas[: max(1, max_regions)])
    return retained / float(total_pixels)


def _score_maps(
    maps: list[np.ndarray], pixel_threshold: float
) -> pd.DataFrame:
    rows: list[dict] = []
    for probability_map in maps:
        flat = np.ravel(probability_map)
        size = int(flat.size)
        if size == 0:
            continue
        top_count = min(size, max(TOP_K_VALUES))
        order = np.sort(
            np.partition(flat, size - top_count)[size - top_count :]
        )[::-1]
        row = {"max_pixel": float(flat.max())}
        for k in TOP_K_VALUES:
            effective = min(k, size)
            row[f"mean_top_{k}"] = float(order[:effective].mean())
        row["area_fraction_raw"] = float(
            np.count_nonzero(probability_map >= pixel_threshold) / float(size)
        )
        row["area_fraction_retained"] = _retained_area_fraction(
            probability_map, pixel_threshold
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Rank-based AUC (Mann-Whitney U), tolerant of tied scores."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start
        while (
            end + 1 < len(sorted_scores)
            and sorted_scores[end + 1] == sorted_scores[start]
        ):
            end += 1
        average_rank = (start + end + 2) / 2.0  # 1-based average rank
        ranks[order[start : end + 1]] = average_rank
        start = end + 1
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _confusion(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predicted = np.asarray(scores) >= float(threshold)
    labels = np.asarray(labels, dtype=bool)
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    tn = int(np.sum(~predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    positives = tp + fn
    negatives = tn + fp
    recall = tp / positives if positives else 0.0
    fpr = fp / negatives if negatives else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    balanced_accuracy = (recall + (1.0 - fpr)) / 2.0
    return {
        "threshold": round(float(threshold), 6),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "recall": round(recall, 6),
        "false_positive_rate": round(fpr, 6),
        "precision": round(precision, 6),
        "f1": round(f1, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "accuracy": round((tp + tn) / max(1, len(labels)), 6),
    }


def _candidate_thresholds(scores: np.ndarray, points: int = 201) -> np.ndarray:
    """Distribution-adaptive thresholds so every score scale is covered."""
    scores = np.asarray(scores, dtype=np.float64)
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, points))
    unique = np.unique(np.round(quantiles, 10))
    if unique.size == 0:
        return np.asarray([0.5])
    # Evaluated with >=, so include a value just above the smallest score.
    return np.unique(np.concatenate([[np.nextafter(unique[0], np.inf)], unique]))


def _select_operating_point(
    scores: np.ndarray, labels: np.ndarray, max_fpr: float
) -> dict:
    table = pd.DataFrame(
        [_confusion(scores, labels, threshold) for threshold in _candidate_thresholds(scores)]
    )
    feasible = table[table["false_positive_rate"] <= max_fpr]
    if not feasible.empty:
        ordered = feasible.sort_values(
            by=["recall", "balanced_accuracy", "threshold"],
            ascending=[False, False, False],
        )
        row = ordered.iloc[0]
        rule = f"recall maximised subject to FPR <= {max_fpr:.0%}"
    else:
        ordered = table.sort_values(
            by=["false_positive_rate", "recall", "balanced_accuracy"],
            ascending=[True, False, False],
        )
        row = ordered.iloc[0]
        rule = f"no threshold met FPR <= {max_fpr:.0%}; lowest FPR selected"
    return {**row.to_dict(), "selection_rule": rule}


def _bootstrap_auc_difference(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    label_a: str,
    label_b: str,
    rounds: int,
    seed: int = 42,
) -> dict:
    """Percentile CI for AUC(a) - AUC(b) using a pooled bootstrap."""
    if rounds <= 0:
        return {"rounds": 0}
    labels = np.asarray(labels, dtype=bool)
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(rounds):
        indices = rng.integers(0, len(labels), len(labels))
        sample_labels = labels[indices]
        if sample_labels.all() or not sample_labels.any():
            continue
        auc_a = _roc_auc(sample_labels, scores_a[indices])
        auc_b = _roc_auc(sample_labels, scores_b[indices])
        if auc_a is None or auc_b is None:
            continue
        differences.append(auc_a - auc_b)
    if not differences:
        return {"rounds": 0}
    values = np.asarray(differences, dtype=np.float64)
    return {
        "rounds": int(values.size),
        "seed": seed,
        "mean_difference": round(float(values.mean()), 6),
        "ci95_low": round(float(np.quantile(values, 0.025)), 6),
        "ci95_high": round(float(np.quantile(values, 0.975)), 6),
        "probability_b_better": round(float((values < 0).mean()), 6),
        "comparison": f"AUC({label_a}) - AUC({label_b})",
    }


METHOD_DESCRIPTIONS = {
    "max_pixel": (
        "Maximum pixel probability (current). Sensitive to a single high pixel, "
        "but it is the score the localization model already produces."
    ),
}
for _k in TOP_K_VALUES:
    METHOD_DESCRIPTIONS[f"mean_top_{_k}"] = (
        f"Mean of the {_k} highest pixel probabilities. Reduces single-pixel "
        "sensitivity but dilutes small forgeries."
    )
METHOD_DESCRIPTIONS.update(
    {
        "area_fraction_raw": (
            "Fraction of pixels at/above the 0.40 localization threshold, before "
            "component filtering."
        ),
        "area_fraction_retained": (
            "Fraction of pixels inside components that pass the app's minimum-area "
            "filter at the 0.40 localization threshold."
        ),
    }
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="validation",
        choices=["validation"],
        help="Comparison is restricted to the validation split.",
    )
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument("--pixel-threshold", type=float, default=settings.manipulation_threshold)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--tampered-cache",
        type=Path,
        default=settings.project_root / "results" / "validation_score_cache",
    )
    parser.add_argument(
        "--authentic-cache",
        type=Path,
        default=settings.evaluation_dir / "image_level_score_cache" / "authentic_validation",
    )
    parser.add_argument(
        "--authentic-csv",
        type=Path,
        default=settings.evaluation_dir / "authentic_validation_scores.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.evaluation_dir / "module1_image_level_score_comparison.json",
    )
    args = parser.parse_args()

    if args.split != "validation":
        parser.error("Score comparison is restricted to the validation split.")

    split_frame = pd.read_csv(settings.split_csv)
    tampered_names = split_frame[
        (split_frame["split"] == args.split) & (split_frame["class_name"] == "Tampered")
    ]["filename"].astype(str).tolist()
    if not tampered_names:
        print("No tampered validation images found.", file=sys.stderr)
        return 2
    if not args.authentic_csv.exists():
        print(
            f"Missing authentic score list: {args.authentic_csv}. Run "
            "evaluation/score_authentic_validation.py first.",
            file=sys.stderr,
        )
        return 2
    authentic_names = (
        pd.read_csv(args.authentic_csv)["filename"].astype(str).tolist()
    )

    tampered_maps, tampered_kept, tampered_missing = _load_maps(
        tampered_names, args.tampered_cache, "Tampered validation"
    )
    authentic_maps, authentic_kept, authentic_missing = _load_maps(
        authentic_names, args.authentic_cache, "Authentic validation"
    )
    if not tampered_maps or not authentic_maps:
        print("Need both tampered and authentic validation maps.", file=sys.stderr)
        return 2

    tampered_scores = _score_maps(tampered_maps, args.pixel_threshold)
    authentic_scores = _score_maps(authentic_maps, args.pixel_threshold)
    scores = pd.concat([tampered_scores, authentic_scores], ignore_index=True)
    labels = np.concatenate(
        [
            np.ones(len(tampered_scores), dtype=bool),
            np.zeros(len(authentic_scores), dtype=bool),
        ]
    )

    rows: list[dict] = []
    for method in scores.columns:
        values = scores[method].to_numpy(dtype=np.float64)
        auc = _roc_auc(labels, values)
        operating = _select_operating_point(values, labels, args.max_fpr)
        row = {
            "method": method,
            "description": METHOD_DESCRIPTIONS.get(method, ""),
            "auc": None if auc is None else round(auc, 6),
            "operating_threshold": operating["threshold"],
            "operating_recall": operating["recall"],
            "operating_false_positive_rate": operating["false_positive_rate"],
            "operating_precision": operating["precision"],
            "operating_f1": operating["f1"],
            "operating_balanced_accuracy": operating["balanced_accuracy"],
            "operating_selection_rule": operating["selection_rule"],
        }
        if method == "max_pixel":
            current = _confusion(values, labels, 0.42)
            row.update(
                {
                    "current_threshold": 0.42,
                    "current_recall": current["recall"],
                    "current_false_positive_rate": current["false_positive_rate"],
                    "current_precision": current["precision"],
                    "current_f1": current["f1"],
                    "current_balanced_accuracy": current["balanced_accuracy"],
                }
            )
        rows.append(row)

    table = pd.DataFrame(rows)
    table = table.sort_values(
        by=["auc", "operating_balanced_accuracy"], ascending=[False, False]
    ).reset_index(drop=True)

    best_row = table.iloc[0]
    best_method = str(best_row["method"])
    max_row = table[table["method"] == "max_pixel"].iloc[0]
    auc_gap = None
    if best_method != "max_pixel" and best_row["auc"] is not None:
        auc_gap = float(best_row["auc"]) - float(max_row["auc"])

    bootstrap = {"rounds": 0}
    if best_method != "max_pixel" and args.bootstrap > 0:
        bootstrap = _bootstrap_auc_difference(
            labels,
            scores[best_method].to_numpy(dtype=np.float64),
            scores["max_pixel"].to_numpy(dtype=np.float64),
            best_method,
            "max_pixel",
            args.bootstrap,
        )

    # A change is only suggested if the alternative is better by a margin that
    # the bootstrap CI does not straddle zero, and its operating point is also
    # better on balanced accuracy.  Otherwise the current method stays.
    recommendation_changed = (
        best_method != "max_pixel"
        and auc_gap is not None
        and auc_gap >= 0.005
        and bootstrap.get("ci95_low", -1.0) > 0.0
        and float(best_row["operating_balanced_accuracy"])
        > float(max_row["operating_balanced_accuracy"])
    )
    recommendation = {
        "decision": (
            f"Consider replacing max_pixel with {best_method} after a new "
            "validation-only threshold selection."
            if recommendation_changed
            else "Keep max_pixel; no alternative shows a clear, validated advantage."
        ),
        "keep_current_method": not recommendation_changed,
        "current_method": "max_pixel",
        "best_alternative_by_auc": (
            None if best_method == "max_pixel" else best_method
        ),
        "auc_gap_over_current": None if auc_gap is None else round(auc_gap, 6),
        "requires_new_threshold_selection": bool(recommendation_changed),
        "note": (
            "This script is diagnostic only. The configured image threshold "
            "(0.42) and uncertainty margin (0.13) were selected for max_pixel "
            "and are not changed here. Any switch would require a new "
            "validation-only threshold selection and preserved test reporting."
        ),
    }

    source_threshold = settings.evaluation_dir / "module1_image_level_threshold.json"
    reference = {}
    if source_threshold.exists():
        try:
            artifact = json.loads(source_threshold.read_text(encoding="utf-8"))
            reference = {
                "selected_threshold": artifact.get("selected_threshold"),
                "selected_metrics": artifact.get("selected_metrics"),
                "uncertainty": artifact.get("uncertainty"),
                "samples": artifact.get("samples"),
                "generated_utc": artifact.get("generated_utc"),
            }
        except (OSError, json.JSONDecodeError):
            reference = {}

    artifact = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split,
        "dataset": "CASIA2.0 revised (validation split only)",
        "pixel_localization_threshold": args.pixel_threshold,
        "max_false_positive_rate": args.max_fpr,
        "samples": {
            "tampered_scored": int(len(tampered_scores)),
            "tampered_available": int(len(tampered_names)),
            "tampered_missing": len(tampered_missing),
            "authentic_scored": int(len(authentic_scores)),
            "authentic_available": int(len(authentic_names)),
            "authentic_missing": len(authentic_missing),
        },
        "score_scales": {
            "max_pixel": "0-1 pixel probability",
            "mean_top_k": "0-1 pixel probability",
            "area_fraction_raw": "0-1 fraction of image pixels",
            "area_fraction_retained": "0-1 fraction of image pixels",
        },
        "primary_metric": "AUC (threshold-independent ranking quality)",
        "operating_point_rule": (
            f"Per-method threshold selected on the same validation data by "
            f"maximising recall subject to FPR <= {args.max_fpr:.0%}; AUC is "
            "reported separately so the comparison is not driven by one cutoff."
        ),
        "current_provenance": reference,
        "methods": json.loads(table.to_json(orient="records")),
        "bootstrap_auc_difference": bootstrap,
        "recommendation": recommendation,
        "missing_examples": {
            "tampered": tampered_missing[:10],
            "authentic": authentic_missing[:10],
        },
        "test_split_used": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    table.to_csv(csv_path, index=False)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    display = table[
        [
            "method",
            "auc",
            "operating_threshold",
            "operating_recall",
            "operating_false_positive_rate",
            "operating_balanced_accuracy",
        ]
    ]
    print("\nValidation-only image-level score comparison")
    print(display.to_string(index=False))
    print("\nCurrent max_pixel operating point at threshold 0.42:")
    print(
        table[table["method"] == "max_pixel"][
            [
                "current_threshold",
                "current_recall",
                "current_false_positive_rate",
                "current_precision",
                "current_f1",
                "current_balanced_accuracy",
            ]
        ].to_string(index=False)
    )
    if bootstrap.get("rounds"):
        print(
            f"\nBootstrap {bootstrap['comparison']}: "
            f"{bootstrap['mean_difference']:+.4f} "
            f"(95% CI {bootstrap['ci95_low']:+.4f} to {bootstrap['ci95_high']:+.4f})"
        )
    print(f"\nRecommendation: {recommendation['decision']}")
    print(f"Saved: {args.output}")
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
