"""Run Module 1 on external images without tuning any threshold.

This is a *generalization / external* check, kept deliberately separate from the
validation-based calibration:

* it never changes the localization threshold (0.40) or the image-level
  threshold (0.42) / uncertainty margin (0.13);
* it never computes training or threshold-selection statistics;
* it reports each image individually, and a group summary only for orientation.

The manifest is a CSV with at least a ``path`` column. Optional columns:

* ``group``  - free-text grouping label (default ``external``)
* ``label``  - ``tampered``, ``authentic`` or ``unknown`` (default ``unknown``)

External phone photographs should normally use ``label=unknown``: a photograph
coming directly from a phone is not proof that it is authentic, so the app must
not be scored as if it were a confirmed negative.  Only use ``authentic`` or
``tampered`` when the label is independently known (e.g. a documented edit).

Usage:
    python evaluation/external_image_test.py --manifest external_images.csv
    python evaluation/external_image_test.py --manifest external_images.csv --max-images 8
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.image_manipulation import detect_image_manipulation
from utils.config import settings
from utils.image_utils import load_image_rgb


VALID_LABELS = {"tampered", "authentic", "unknown"}


def _read_manifest(path: Path, max_images: int | None) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The manifest is empty.")
    if "path" not in rows[0]:
        raise ValueError("The manifest needs a 'path' column.")

    entries: list[dict] = []
    for row in rows:
        raw_path = (row.get("path") or "").strip()
        if not raw_path:
            continue
        image_path = Path(raw_path).expanduser()
        if not image_path.is_absolute():
            image_path = (path.parent / image_path).resolve()
        label = (row.get("label") or "unknown").strip().lower()
        if label not in VALID_LABELS:
            raise ValueError(
                f"Invalid label '{label}' for {image_path.name}; "
                f"expected one of {sorted(VALID_LABELS)}."
            )
        entries.append(
            {
                "path": image_path,
                "group": (row.get("group") or "external").strip() or "external",
                "label": label,
                "note": (row.get("note") or "").strip(),
            }
        )
        if max_images is not None and len(entries) >= max_images:
            break
    return entries


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.evaluation_dir,
    )
    parser.add_argument("--pixel-threshold", type=float, default=settings.manipulation_threshold)
    parser.add_argument("--image-threshold", type=float, default=settings.image_level_threshold)
    parser.add_argument(
        "--uncertainty-margin", type=float, default=settings.image_uncertainty_margin
    )
    args = parser.parse_args()

    entries = _read_manifest(args.manifest, args.max_images)
    if not entries:
        print("No valid manifest entries.", file=sys.stderr)
        return 2

    rows: list[dict] = []
    failures: list[dict] = []
    for index, entry in enumerate(entries, start=1):
        image_path: Path = entry["path"]
        print(
            f"[{index}/{len(entries)}] {entry['group']} · {image_path.name}",
            flush=True,
        )
        try:
            rgb = load_image_rgb(image_path)
        except Exception as exc:
            failures.append({"source": image_path.name, "error": str(exc)})
            print(f"  skipped: {exc}", flush=True)
            continue
        try:
            result = detect_image_manipulation(
                rgb,
                threshold=args.pixel_threshold,
                image_threshold=args.image_threshold,
                image_uncertainty_margin=args.uncertainty_margin,
            )
        except Exception as exc:  # defensive: one bad image must not stop the run
            failures.append({"source": image_path.name, "error": str(exc)})
            print(f"  failed: {exc}", flush=True)
            continue
        if result.status != "ok":
            failures.append(
                {"source": image_path.name, "error": result.error_message or result.status}
            )
            print(f"  failed: {result.error_message}", flush=True)
            continue

        row = {
            "source": image_path.name,
            "group": entry["group"],
            "manual_label": entry["label"],
            "note": entry["note"],
            "image_size": f"{result.image_size[0]}x{result.image_size[1]}",
            "image_decision": result.image_decision,
            "image_decision_label": result.image_decision_label,
            "localization_label": result.localization_label,
            "max_pixel_probability": round(result.image_score, 6),
            "image_threshold": round(result.image_threshold, 4),
            "uncertainty_margin": round(result.image_uncertainty_margin, 4),
            "image_decision_margin": round(result.image_decision_margin, 6),
            "localization_threshold": round(result.localization_threshold, 4),
            "raw_suspicious_pixels": result.raw_suspicious_pixels,
            "raw_suspicious_area_percent": round(
                result.raw_suspicious_area_ratio * 100, 8
            ),
            "highlighted_pixels": result.processed_suspicious_pixels,
            "highlighted_area_percent": round(
                result.processed_suspicious_area_ratio * 100, 8
            ),
            "raw_component_count": result.raw_component_count,
            "retained_component_count": len(result.regions),
            "explanation": result.decision_explanation,
            "warnings": " | ".join(result.warnings),
        }
        rows.append(row)
        print(
            f"  {result.image_decision_label} · max={result.image_score:.4f} · "
            f"raw={result.raw_suspicious_area_ratio * 100:.6f}% · "
            f"highlighted={result.processed_suspicious_area_ratio * 100:.6f}%",
            flush=True,
        )

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    summary = []
    for group, group_rows in sorted(groups.items()):
        decisions = Counter(row["image_decision"] for row in group_rows)
        summary.append(
            {
                "group": group,
                "images": len(group_rows),
                "authentic": decisions.get("authentic", 0),
                "tampered": decisions.get("tampered", 0),
                "uncertain": decisions.get("uncertain", 0),
                "mean_max_pixel_probability": _mean(
                    [row["max_pixel_probability"] for row in group_rows]
                ),
                "mean_raw_area_percent": _mean(
                    [row["raw_suspicious_area_percent"] for row in group_rows]
                ),
                "mean_highlighted_area_percent": _mean(
                    [row["highlighted_area_percent"] for row in group_rows]
                ),
            }
        )

    labelled = [row for row in rows if row["manual_label"] in {"authentic", "tampered"}]
    labelled_summary = {
        "images_with_known_label": len(labelled),
        "note": (
            "Descriptive only. The external set is small and was not used to "
            "select or tune any threshold; do not report it as a test-split metric."
        ),
    }
    if labelled:
        correct = sum(
            1
            for row in labelled
            if (row["manual_label"] == "tampered")
            == (row["image_decision"] == "tampered")
        )
        labelled_summary["descriptive_agreement"] = round(correct / len(labelled), 6)

    artifact = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "external_generalization_test",
        "test_split_used_for_tuning": False,
        "thresholds": {
            "pixel_localization": args.pixel_threshold,
            "image_level": args.image_threshold,
            "uncertainty_margin": args.uncertainty_margin,
            "note": "Fixed from validation data; not adjusted on these images.",
        },
        "manifest": args.manifest.name,
        "images_attempted": len(entries),
        "images_scored": len(rows),
        "failures": failures,
        "summary_by_group": summary,
        "labelled_summary": labelled_summary,
        "images": rows,
        "caveats": [
            "A phone photograph is not automatically authentic; unlabelled "
            "external images are reported as unknown, not as true negatives.",
            "Known AI-generated images are outside the manipulation dataset and "
            "are not a validated class for MVSS-Net; results are descriptive.",
            "This set is too small for accuracy/F1 claims and was not used to "
            "change the validation-selected thresholds.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "external_generalization_test.json"
    csv_path = args.output_dir / "external_generalization_test.csv"
    json_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("\nSummary by group")
    for item in summary:
        print(
            f"  {item['group']}: n={item['images']} "
            f"(A={item['authentic']}, T={item['tampered']}, U={item['uncertain']}) "
            f"mean max={item['mean_max_pixel_probability']:.4f}"
        )
    print(f"\nScored {len(rows)}/{len(entries)} image(s).")
    print(f"Saved: {json_path}")
    if rows:
        print(f"Saved: {csv_path}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
