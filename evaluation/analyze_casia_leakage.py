"""Quantify CASIA cross-split leakage and write a leakage-free experimental split.

Two artefacts are produced, without touching the historical split files:

``results/evaluation/casia2_leakage_impact.json``
    Pixel-level test metrics recomputed from the per-image evaluation artefact
    on (a) the full historical test set and (b) the subset whose duplicate
    component appears in only one split.  The per-image TP/FP/FN counts are
    recovered algebraically from the stored IoU/precision/recall and the
    ground-truth mask area; no model inference is run.

``results/manifests/casia2_groupaware_split.csv``
    A separate experimental split in which every duplicate component
    (exact/decoded/pHash/dHash) is kept wholly inside one split.  It is offered
    for future retraining; the historical split remains the reported baseline.

Method notes
------------
Given ground-truth positive pixels ``G`` and stored recall ``r``, precision
``p`` and IoU ``IoU``::

    TP = r * G
    FN = G - TP
    FP = TP / IoU - TP - FN          (when TP > 0)
       = TP * (1 - p) / p            (cross-check when p > 0)

When ``TP == 0`` the stored metrics cannot distinguish a true negative from an
all-false-positive mask, so ``FP`` is recorded as 0 for that image and the
ambiguity is reported in the artefact.
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

from dataset_manifest.io import load_manifest
from dataset_manifest.schema import resolve_manifest_path
from utils.config import settings


HASH_MODALITIES = ("exact_groups", "decoded_groups", "phash_groups", "dhash_groups")


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, node: str) -> str:
        self.parent.setdefault(node, node)
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            self.parent[node], node = root, self.parent[node]
        return root

    def union(self, first: str, second: str) -> None:
        root_first, root_second = self.find(first), self.find(second)
        if root_first != root_second:
            self.parent[root_second] = root_first


def _load_audit(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_components(records: list[dict], audit: dict) -> tuple[dict[str, str], dict[str, dict]]:
    """Union duplicate groups into components; return sample->component and metadata."""

    union = UnionFind()
    for record in records:
        union.find(record["sample_id"])

    modality_by_sample: dict[str, set[str]] = {}
    for modality in HASH_MODALITIES:
        for group in audit.get(modality, []):
            sample_ids = [sample for sample in group.get("sample_ids", []) if sample]
            for sample in sample_ids:
                modality_by_sample.setdefault(sample, set()).add(modality)
            for sample in sample_ids[1:]:
                union.union(sample_ids[0], sample)

    components: dict[str, dict] = {}
    for record in records:
        root = union.find(record["sample_id"])
        entry = components.setdefault(
            root,
            {"component_id": root, "sample_ids": [], "splits": set(), "classes": set(), "modalities": set()},
        )
        entry["sample_ids"].append(record["sample_id"])
        entry["splits"].add(record["split"])
        entry["classes"].add(record["class_name"])
        entry["modalities"].update(modality_by_sample.get(record["sample_id"], set()))

    sample_to_component = {
        sample: entry["component_id"]
        for entry in components.values()
        for sample in entry["sample_ids"]
    }
    return sample_to_component, components


def _mask_area(mask_path: str) -> int:
    path = resolve_manifest_path(mask_path, settings.data_root)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Mask could not be decoded: {path}")
    return int((mask > 0).sum())


def _recover_counts(gt_pixels: int, iou: float, precision: float, recall: float) -> dict:
    tp = float(recall) * gt_pixels
    fn = gt_pixels - tp
    if tp > 0 and iou > 0:
        fp = tp / iou - tp - fn
    elif tp > 0 and precision > 0:
        fp = tp * (1.0 - precision) / precision
    else:
        fp = 0.0
    precision_check = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {
        "tp": max(0.0, tp),
        "fp": max(0.0, fp),
        "fn": max(0.0, fn),
        "precision_check": precision_check,
        "ambiguous": bool(tp == 0),
    }


def _aggregate(rows: pd.DataFrame) -> dict:
    tp = float(rows["tp"].sum())
    fp = float(rows["fp"].sum())
    fn = float(rows["fn"].sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {
        "images": int(len(rows)),
        "gt_pixels": int(rows["gt_pixels"].sum()),
        "tp_pixels": int(round(tp)),
        "fp_pixels": int(round(fp)),
        "fn_pixels": int(round(fn)),
        "micro_precision": round(precision, 6),
        "micro_recall": round(recall, 6),
        "micro_f1": round(f1, 6),
        "micro_iou": round(iou, 6),
        "mean_image_iou": round(float(rows["IoU"].mean()), 6) if len(rows) else 0.0,
        "mean_image_f1": round(float(rows["F1"].mean()), 6) if len(rows) else 0.0,
        "images_with_ambiguous_fp": int(rows["ambiguous"].sum()),
    }


def _write_groupaware_split(
    records: list[dict],
    sample_to_component: dict[str, str],
    components: dict[str, dict],
    output: Path,
    seed: int,
) -> dict:
    # Components with mixed classes are assigned as a whole; class ratios are
    # therefore approximate, but no duplicate can cross a split.
    component_class: dict[str, str] = {}
    for component_id, entry in components.items():
        component_class[component_id] = (
            "Tampered" if "Tampered" in entry["classes"] else "Authentic"
        )

    assignment: dict[str, str] = {}
    for class_name, ratios in (("Authentic", (0.70, 0.85)), ("Tampered", (0.70, 0.85))):
        class_components = [
            component_id
            for component_id, value in component_class.items()
            if value == class_name
        ]
        class_components.sort(
            key=lambda component_id: hashlib.sha256(
                f"{seed}:{component_id}".encode("utf-8")
            ).hexdigest()
        )
        total_rows = sum(len(components[component_id]["sample_ids"]) for component_id in class_components)
        cumulative = 0
        for component_id in class_components:
            size = len(components[component_id]["sample_ids"])
            fraction = (cumulative + size / 2.0) / max(1, total_rows)
            if fraction < ratios[0]:
                split = "train"
            elif fraction < ratios[1]:
                split = "validation"
            else:
                split = "test"
            assignment[component_id] = split
            cumulative += size

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "sample_id": record["sample_id"],
            "group_id": record["group_id"],
            "duplicate_component": sample_to_component[record["sample_id"]],
            "class_name": record["class_name"],
            "mask_available": record["mask_available"],
            "original_split": record["split"],
            "experimental_split": assignment[sample_to_component[record["sample_id"]]],
        }
        for record in records
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)

    by_split = frame["experimental_split"].value_counts().to_dict()
    by_class = (
        frame.groupby(["experimental_split", "class_name"]).size().unstack(fill_value=0).to_dict()
    )
    component_splits = frame.groupby("duplicate_component")["experimental_split"].nunique()
    summary = {
        "artifact": output.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "rows": int(len(frame)),
        "components": int(frame["duplicate_component"].nunique()),
        "split_counts": {key: int(value) for key, value in by_split.items()},
        "class_counts_per_split": {
            split: {class_name: int(count) for class_name, count in values.items()}
            for split, values in by_class.items()
        },
        "components_spanning_multiple_experimental_splits": int((component_splits > 1).sum()),
        "note": (
            "Experimental split only. The historical CASIA split files were not modified. "
            "Duplicate components come from exact, decoded, pHash and dHash groups in "
            "results/manifests/casia2_duplicate_audit.json."
        ),
    }
    output.with_name(output.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "results" / "manifests" / "casia2_manifest.csv",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=PROJECT_ROOT / "results" / "manifests" / "casia2_duplicate_audit.json",
    )
    parser.add_argument(
        "--per-image",
        type=Path,
        default=PROJECT_ROOT / "results" / "evaluation" / "module1_test_metrics_per_image.csv",
    )
    parser.add_argument(
        "--image-level",
        type=Path,
        default=PROJECT_ROOT / "results" / "evaluation" / "module1_image_level_score_comparison.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.evaluation_dir / "casia2_leakage_impact.json",
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        default=PROJECT_ROOT / "results" / "manifests" / "casia2_groupaware_split.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    audit = _load_audit(args.audit)
    sample_to_component, components = _build_components(records, audit)

    leaked_components = {
        component_id
        for component_id, entry in components.items()
        if len(entry["splits"]) > 1
    }
    leaked_test_samples = {
        sample
        for sample, component_id in sample_to_component.items()
        if component_id in leaked_components
    }

    manifest_by_filename = {}
    for record in records:
        manifest_by_filename[Path(record["image_path"]).name] = record

    per_image = pd.read_csv(args.per_image)
    rows = []
    for _, row in per_image.iterrows():
        record = manifest_by_filename.get(row["filename"])
        if record is None:
            raise ValueError(f"Image {row['filename']} is not present in the manifest")
        gt_pixels = _mask_area(record["mask_path"])
        counts = _recover_counts(
            gt_pixels,
            float(row["IoU"]),
            float(row["Precision"]),
            float(row["Recall"]),
        )
        rows.append(
            {
                "sample_id": record["sample_id"],
                "filename": row["filename"],
                "original_split": record["split"],
                "component_id": sample_to_component[record["sample_id"]],
                "leaked_component": sample_to_component[record["sample_id"]] in leaked_components,
                "gt_pixels": gt_pixels,
                "IoU": float(row["IoU"]),
                "F1": float(row["F1"]),
                "max_probability": float(row["max_probability"]),
                **counts,
            }
        )

    frame = pd.DataFrame(rows)
    full = _aggregate(frame)
    clean = _aggregate(frame[~frame["leaked_component"]])
    leaked = _aggregate(frame[frame["leaked_component"]])

    threshold = 0.42
    tampered_recall_full = float((frame["max_probability"] >= threshold).mean())
    clean_frame = frame[~frame["leaked_component"]]
    tampered_recall_clean = (
        float((clean_frame["max_probability"] >= threshold).mean()) if len(clean_frame) else None
    )

    image_level_reference = None
    if args.image_level.exists():
        comparison = pd.read_csv(args.image_level)
        max_pixel_row = comparison[comparison["method"] == "max_pixel"]
        if not max_pixel_row.empty:
            row = max_pixel_row.iloc[0]
            image_level_reference = {
                "method": "max_pixel",
                "split": "validation",
                "auc": float(row["auc"]),
                "operating_threshold": float(row["operating_threshold"]),
                "operating_recall": float(row["operating_recall"]),
                "operating_false_positive_rate": float(row["operating_false_positive_rate"]),
                "current_threshold": float(row["current_threshold"]),
                "current_recall": float(row["current_recall"]),
                "current_false_positive_rate": float(row["current_false_positive_rate"]),
                "note": (
                    "These are the published *validation* image-level metrics from the "
                    "method comparison; the test split was not rescored there."
                ),
            }

    test_image_level_report = None
    recovery_check = None
    test_report_path = args.per_image.with_name("module1_test_metrics.json")
    if test_report_path.exists():
        test_artifact = json.loads(test_report_path.read_text(encoding="utf-8"))
        test_image_level_report = {
            "threshold": test_artifact.get("threshold"),
            "image_level": test_artifact.get("image_level"),
            "samples": test_artifact.get("samples"),
            "note": (
                "Existing test report at the 0.40 localization threshold. Authentic FPR is "
                "an estimate from 12 sampled authentic test images, not the full split."
            ),
        }
        localization = test_artifact.get("localization", {})
        recovery_check = {
            "reported_global_iou": localization.get("global_iou"),
            "recovered_global_iou": full["micro_iou"],
            "global_iou_absolute_difference": round(
                abs(float(localization.get("global_iou", 0.0)) - full["micro_iou"]), 6
            ),
            "reported_micro_precision": localization.get("micro_precision"),
            "recovered_micro_precision": full["micro_precision"],
            "reported_micro_recall": localization.get("micro_recall"),
            "recovered_micro_recall": full["micro_recall"],
            "note": "Small differences are expected from rounded per-image metrics.",
        }

    artifact = {
        "artifact": args.output.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "module": "Module 1 - CASIA2.0 cross-split leakage impact",
        "manifest": str(args.manifest),
        "duplicate_audit": str(args.audit),
        "per_image_metrics": str(args.per_image),
        "recovery_method": (
            "TP = recall * GT pixels; FN = GT - TP; FP = TP/IoU - TP - FN (fallback "
            "TP*(1-precision)/precision). GT area is measured from each test mask. "
            "Images with TP=0 have ambiguous FP and are counted explicitly."
        ),
        "leakage_definition": (
            "A duplicate component is leaked when exact/decoded/pHash/dHash evidence links "
            "its samples across more than one historical split."
        ),
        "test_images": int(len(frame)),
        "leaked_test_images": int(frame["leaked_component"].sum()),
        "leaked_test_fraction": round(float(frame["leaked_component"].mean()), 6),
        "clean_test_images": int(len(clean_frame)),
        "clean_test_fraction": round(float(len(clean_frame)) / max(1, len(frame)), 6),
        "leaked_components": len(leaked_components),
        "leaked_test_sample_ids_total_in_audit": len(leaked_test_samples),
        "metrics": {
            "full_test_set": full,
            "clean_subset_no_cross_split_duplicates": clean,
            "leaked_subset": leaked,
            "delta_clean_minus_full": {
                "micro_iou": round(clean["micro_iou"] - full["micro_iou"], 6),
                "mean_image_iou": round(clean["mean_image_iou"] - full["mean_image_iou"], 6),
                "micro_f1": round(clean["micro_f1"] - full["micro_f1"], 6),
            },
        },
        "image_level_tampered_recall_at_0.42": {
            "full_test_set": round(tampered_recall_full, 6),
            "clean_subset": round(tampered_recall_clean, 6)
            if tampered_recall_clean is not None
            else None,
            "note": "No authentic test max-probabilities are available for a clean FPR/AUC.",
        },
        "validation_image_level_comparison": image_level_reference,
        "test_image_level_report": test_image_level_report,
        "recovery_check": recovery_check,
        "duplicate_summary_from_audit": audit.get("summary", {}),
        "limitations": [
            "Counts are recovered from rounded per-image metrics; small rounding differences "
            "are possible.",
            "The clean subset is small and biased towards images that happen to have no "
            "detected duplicate; it is not a replacement for retraining on the leakage-free "
            "experimental split.",
            "Ground-truth masks are binary region annotations, not exact object boundaries.",
            "The clean subset only removes duplicates that cross historical splits according "
            "to the audit hashes; undetected near-duplicates may remain.",
            "A leakage-free experimental split is provided, but retraining is required before "
            "its metrics can be reported.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    split_summary = _write_groupaware_split(
        records, sample_to_component, components, args.split_output, args.seed
    )

    print("=== CASIA2.0 LEAKAGE IMPACT ===")
    print(
        f"Test images: {artifact['test_images']} | in cross-split components: "
        f"{artifact['leaked_test_images']} ({artifact['leaked_test_fraction'] * 100:.2f}%) | "
        f"clean subset: {artifact['clean_test_images']}"
    )
    print(
        f"Full micro IoU: {full['micro_iou']:.4f} | clean micro IoU: {clean['micro_iou']:.4f} "
        f"| mean image IoU: {full['mean_image_iou']:.4f} -> {clean['mean_image_iou']:.4f}"
    )
    print(
        f"Image-level recall @0.42: {tampered_recall_full:.4f} -> "
        f"{tampered_recall_clean if tampered_recall_clean is None else round(tampered_recall_clean, 4)}"
    )
    print(f"\nSaved: {args.output}")
    print(f"Saved: {args.split_output}")
    print(
        f"Experimental split: {split_summary['split_counts']} | components spanning splits: "
        f"{split_summary['components_spanning_multiple_experimental_splits']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
