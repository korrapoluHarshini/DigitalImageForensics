"""Select the Module 1 image-level Authentic/Tampered threshold on validation only.

The image-level score is the maximum pixel probability produced by the existing
MVSS-Net localization map.  Thresholding that score gives an image-level
decision; this script chooses the threshold from validation data only, using
both authentic and tampered validation images.

Selection rule
--------------
1. Sweep thresholds and compute recall, false-positive rate (FPR), F1 and
   balanced accuracy.
2. Keep thresholds whose validation FPR is at most ``--max-fpr`` (default 5%).
   This directly controls the false-positive problem of the raw maximum-pixel
   rule.
3. Among those, choose the threshold with the highest recall (tie-break:
   balanced accuracy, then the higher threshold).
4. If no threshold meets the FPR constraint, choose the lowest FPR, then the
   highest recall.

The uncertainty band is selected on the same validation data: the smallest
half-width that keeps at least ``--min-coverage`` of validation images outside
the band while minimising confidently misclassified images.  The test split is
never read by this script.

Usage:
    python evaluation/select_image_threshold_module1.py
    python evaluation/select_image_threshold_module1.py --max-fpr 0.05 --min-coverage 0.90
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import settings


def _cache_name(filename: str) -> str:
    return hashlib.sha1(str(filename).encode("utf-8")).hexdigest() + ".npy"


def _load_scores(
    filenames: list[str],
    cache_dir: Path,
    csv_path: Path | None,
) -> tuple[np.ndarray, list[str]]:
    """Return maximum probability per filename and a list of missing names."""
    csv_lookup: dict[str, float] = {}
    if csv_path is not None and csv_path.exists():
        try:
            frame = pd.read_csv(csv_path)
            csv_lookup = {
                str(row.filename): float(row.max_probability)
                for row in frame.itertuples()
            }
        except Exception:
            csv_lookup = {}

    values: list[float] = []
    missing: list[str] = []
    for filename in filenames:
        cache_path = cache_dir / _cache_name(filename)
        if cache_path.exists():
            try:
                values.append(float(np.load(cache_path).max()))
                continue
            except Exception:
                pass
        if filename in csv_lookup:
            values.append(csv_lookup[filename])
            continue
        missing.append(filename)
    return np.asarray(values, dtype=np.float64), missing


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
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        average_rank = (start + end + 2) / 2.0  # 1-based average rank
        ranks[order[start : end + 1]] = average_rank
        start = end + 1
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _confusion(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    tn = int(np.sum(~predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    positives = tp + fn
    negatives = tn + fp
    recall = tp / positives if positives else 0.0
    fpr = fp / negatives if negatives else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    balanced_accuracy = (recall + (1.0 - fpr)) / 2.0
    accuracy = (tp + tn) / max(1, len(labels))
    return {
        "threshold": float(threshold),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "recall": round(recall, 6),
        "false_positive_rate": round(fpr, 6),
        "precision": round(precision, 6),
        "f1": round(f1, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "accuracy": round(accuracy, 6),
    }


def _select_threshold(table: pd.DataFrame, max_fpr: float) -> tuple[pd.Series, str]:
    feasible = table[table["false_positive_rate"] <= max_fpr]
    if not feasible.empty:
        ordered = feasible.sort_values(
            by=["recall", "balanced_accuracy", "threshold"],
            ascending=[False, False, False],
        )
        return ordered.iloc[0], f"recall maximised subject to FPR <= {max_fpr:.0%}"
    ordered = table.sort_values(
        by=["false_positive_rate", "recall", "balanced_accuracy"],
        ascending=[True, False, False],
    )
    return (
        ordered.iloc[0],
        f"no threshold met FPR <= {max_fpr:.0%}; lowest FPR selected",
    )


def _select_margin(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    min_coverage: float,
) -> tuple[dict, list[dict]]:
    """Choose the smallest uncertainty band meeting the coverage constraint."""
    rows: list[dict] = []
    total = len(labels)
    for step in range(0, 21):
        margin = round(0.01 * step, 2)
        inside = np.abs(scores - threshold) <= margin
        coverage = float((~inside).mean()) if total else 0.0
        confident = ~inside
        confident_errors = int(np.sum(confident & ((scores >= threshold) != labels)))
        confident_correct = int(np.sum(confident & ((scores >= threshold) == labels)))
        selective_accuracy = confident_correct / max(1, confident_correct + confident_errors)
        selective_balanced_accuracy = None
        confident_positives = int(np.sum(confident & labels))
        confident_negatives = int(np.sum(confident & ~labels))
        if confident_positives and confident_negatives:
            tp = int(np.sum(confident & labels & (scores >= threshold)))
            fp = int(np.sum(confident & ~labels & (scores >= threshold)))
            selective_balanced_accuracy = 0.5 * (
                tp / confident_positives
                + (1.0 - fp / confident_negatives)
            )
        rows.append(
            {
                "margin": margin,
                "coverage": round(coverage, 6),
                "abstained": int(inside.sum()),
                "confident_correct": confident_correct,
                "confident_errors": confident_errors,
                "selective_accuracy": round(selective_accuracy, 6),
                "selective_balanced_accuracy": (
                    None
                    if selective_balanced_accuracy is None
                    else round(selective_balanced_accuracy, 6)
                ),
            }
        )

    eligible = [row for row in rows if row["coverage"] >= min_coverage]
    if eligible:
        best = min(
            eligible,
            key=lambda row: (
                row["confident_errors"],
                -row["coverage"],
                row["margin"],
            ),
        )
        reason = (
            f"fewest confident errors with coverage >= {min_coverage:.0%}; "
            "smallest margin breaks ties"
        )
    else:
        best = min(rows, key=lambda row: (row["confident_errors"], row["margin"]))
        reason = "no margin met the coverage constraint; fewest confident errors selected"
    return {**best, "selection_rule": reason}, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation", choices=["validation"])
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument("--min-coverage", type=float, default=0.90)
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
    parser.add_argument("--min-authentic", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.evaluation_dir / "module1_image_level_threshold.json",
    )
    args = parser.parse_args()

    if args.split != "validation":
        parser.error("Threshold selection is restricted to the validation split.")

    split_frame = pd.read_csv(settings.split_csv)
    tampered_names = split_frame[
        (split_frame["split"] == args.split) & (split_frame["class_name"] == "Tampered")
    ]["filename"].astype(str).tolist()
    authentic_names = split_frame[
        (split_frame["split"] == args.split) & (split_frame["class_name"] == "Authentic")
    ]["filename"].astype(str).tolist()
    if not tampered_names or not authentic_names:
        print("Validation split does not contain both classes.", file=sys.stderr)
        return 2

    tampered_scores, tampered_missing = _load_scores(
        tampered_names, args.tampered_cache, None
    )
    authentic_scores, authentic_missing = _load_scores(
        authentic_names, args.authentic_cache, args.authentic_csv
    )

    print(
        f"Tampered validation scores: {len(tampered_scores)}/{len(tampered_names)}",
        flush=True,
    )
    print(
        f"Authentic validation scores: {len(authentic_scores)}/{len(authentic_names)}",
        flush=True,
    )
    if len(authentic_scores) < args.min_authentic:
        print(
            f"Only {len(authentic_scores)} authentic validation scores are available; "
            f"at least {args.min_authentic} are required before selecting a threshold. "
            "Run evaluation/score_authentic_validation.py first.",
            file=sys.stderr,
        )
        return 2
    if not len(tampered_scores):
        print("No tampered validation scores available.", file=sys.stderr)
        return 2

    scores = np.concatenate([tampered_scores, authentic_scores])
    labels = np.concatenate(
        [np.ones(len(tampered_scores), dtype=bool), np.zeros(len(authentic_scores), dtype=bool)]
    )

    scoring_manifest: dict = {}
    manifest_path = args.authentic_csv.with_suffix(".manifest.json")
    if manifest_path.exists():
        try:
            scoring_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            scoring_manifest = {}

    thresholds = sorted(
        {round(float(value), 3) for value in np.arange(0.05, 0.991, max(0.005, args.step))}
    )
    table = pd.DataFrame(
        [_confusion(scores, labels, threshold) for threshold in thresholds]
    )
    best_row, threshold_rule = _select_threshold(table, args.max_fpr)
    threshold = float(best_row["threshold"])

    margin_summary, margin_rows = _select_margin(
        scores, labels, threshold, args.min_coverage
    )
    margin_table = pd.DataFrame(margin_rows)

    auc = _roc_auc(labels, scores)
    float_keys = {
        "threshold",
        "recall",
        "false_positive_rate",
        "precision",
        "f1",
        "balanced_accuracy",
        "accuracy",
    }
    selected = {
        key: (float(value) if key in float_keys else int(value))
        for key, value in best_row.items()
    }

    artifact = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split,
        "dataset": "CASIA2.0 revised (validation split only)",
        "score_definition": (
            "maximum pixel probability from the official single-scale MVSS-Net "
            "inference; already used for localization, not a separate classifier"
        ),
        "samples": {
            "tampered_scored": int(len(tampered_scores)),
            "tampered_available": int(len(tampered_names)),
            "authentic_scored": int(len(authentic_scores)),
            "authentic_available": int(len(authentic_names)),
            "tampered_missing": len(tampered_missing),
            "authentic_missing": len(authentic_missing),
            "authentic_scoring_requested": scoring_manifest.get("requested"),
            "authentic_scoring_seed": scoring_manifest.get("seed"),
            "authentic_scoring_failed": (
                None
                if scoring_manifest.get("requested") is None
                else int(scoring_manifest["requested"]) - int(scoring_manifest.get("scored", len(authentic_scores)))
            ),
        },
        "auc": None if auc is None else round(auc, 6),
        "selection_rule": threshold_rule,
        "selection_constraints": {"max_false_positive_rate": args.max_fpr},
        "selected_threshold": threshold,
        "selected_metrics": selected,
        "uncertainty": {
            **margin_summary,
            "constraint": {"min_coverage": args.min_coverage},
        },
        "sources": {
            "tampered_cache": str(args.tampered_cache.relative_to(settings.project_root)).replace("\\", "/"),
            "authentic_cache": str(args.authentic_cache.relative_to(settings.project_root)).replace("\\", "/"),
            "authentic_csv": args.authentic_csv.name,
        },
        "note": (
            "The test split was not read. Authentic validation images were scored as a "
            "deterministic 400-image sample (seed 42) from the 1124-image validation "
            "authentic pool; 2 .bmp files were rejected by the application's upload "
            "validator. The image-level score is not a calibrated probability."
        ),
        "authentic_scoring_manifest": scoring_manifest,
        "missing_examples": {
            "tampered": tampered_missing[:10],
            "authentic": authentic_missing[:10],
        },
        "threshold_table": json.loads(table.to_json(orient="records")),
        "uncertainty_table": json.loads(margin_table.to_json(orient="records")),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_name("module1_image_level_threshold.csv")
    uncertainty_csv = args.output.with_name("module1_image_level_uncertainty.csv")
    table.to_csv(csv_path, index=False)
    margin_table.to_csv(uncertainty_csv, index=False)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(
        table[
            ["threshold", "recall", "false_positive_rate", "precision", "f1", "balanced_accuracy"]
        ].to_string(index=False)
    )
    print(f"\nAUC: {auc:.4f}")
    print(f"Selected image-level threshold: {threshold:.2f} ({threshold_rule})")
    print(
        f"  recall={selected['recall']:.4f}  FPR={selected['false_positive_rate']:.4f}  "
        f"F1={selected['f1']:.4f}  balanced accuracy={selected['balanced_accuracy']:.4f}"
    )
    print(
        f"Selected uncertainty margin: +/-{margin_summary['margin']:.2f} "
        f"(coverage {margin_summary['coverage']:.2%}, "
        f"{margin_summary['confident_errors']} confident errors)"
    )
    print(f"Saved: {args.output}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {uncertainty_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
