"""Per-source image- and pixel-level evaluation from a manifest.

The evaluator is additive and does not import or modify the model checkpoint.
Predictions can be supplied as a CSV with maximum pixel probabilities and
optional raw/filtered masks.  A probability-map path can also be supplied; the
image score is then recomputed as ``max(probability_map)`` so the maximum-pixel
scoring rule is enforced.

Required prediction columns
---------------------------
sample_id
image_score                 maximum pixel probability in [0, 1]
probability_map_path        optional .npy/image probability map
raw_mask_path               optional binary mask thresholded at 0.40
filtered_mask_path          optional post-processed highlighted mask
score_definition            optional; must say max_pixel_probability if present

Raw thresholded masks and post-filtered highlighted masks are accumulated and
reported separately.  Groups without real labelled predictions are reported as
``not_available``; no metric is invented for them.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from utils.metrics import roc_auc
from utils.config import settings

from .schema import normalise_record, resolve_manifest_path, validate_manifest


BASE_GROUPS: tuple[str, ...] = (
    "dslr_mirrorless",
    "android",
    "iphone",
    "other_mobile",
    "unknown",
)

SOURCE_REPORTS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "dslr_mirrorless",
        ("dslr_mirrorless",),
        "DSLR and mirrorless cameras",
    ),
    ("android", ("android",), "Android smartphones"),
    ("iphone", ("iphone",), "iPhones"),
    ("other_mobile", ("other_mobile",), "Other mobile devices"),
    (
        "combined_camera",
        ("dslr_mirrorless",),
        "Combined DSLR/mirrorless camera group",
    ),
    (
        "combined_mobile",
        ("android", "iphone", "other_mobile"),
        "Combined mobile-device group",
    ),
    (
        "all_sources",
        ("dslr_mirrorless", "android", "iphone", "other_mobile", "unknown"),
        "All sources combined",
    ),
)


class EvaluationInputError(ValueError):
    """Raised when predictions or masks cannot be evaluated unambiguously."""


@dataclass
class PredictionRecord:
    sample_id: str
    image_score: float
    probability_map_path: Path | None = None
    raw_mask_path: Path | None = None
    filtered_mask_path: Path | None = None
    score_definition: str = "max_pixel_probability"

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "image_score": float(self.image_score),
            "probability_map_path": str(self.probability_map_path or ""),
            "raw_mask_path": str(self.raw_mask_path or ""),
            "filtered_mask_path": str(self.filtered_mask_path or ""),
            "score_definition": self.score_definition,
        }


def _resolve(value: Any, root: Path | None) -> Path | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return (root / path) if root is not None else path


def _read_probability_map(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path).astype(np.float32)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise EvaluationInputError(f"Probability map could not be decoded: {path}")
        array = image.astype(np.float32)
        if array.max(initial=0) > 1.0:
            array /= 255.0
    if array.ndim == 3:
        array = array[..., 0]
    if not np.isfinite(array).all():
        raise EvaluationInputError(f"Probability map contains non-finite values: {path}")
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def load_predictions(path: str | Path, *, prediction_root: str | Path | None = None) -> list[PredictionRecord]:
    """Load and validate a predictions CSV."""

    path = Path(path)
    root = Path(prediction_root) if prediction_root is not None else path.parent
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise EvaluationInputError(f"Predictions file has no header: {path}")
        if "sample_id" not in reader.fieldnames:
            raise EvaluationInputError("Predictions CSV requires a sample_id column")
        rows = [dict(row) for row in reader]

    predictions: list[PredictionRecord] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise EvaluationInputError(f"Predictions row {index} has an empty sample_id")
        if sample_id in seen:
            raise EvaluationInputError(f"Duplicate prediction for sample_id {sample_id!r}")
        seen.add(sample_id)

        definition = str(row.get("score_definition", "")).strip()
        if definition and definition != "max_pixel_probability":
            raise EvaluationInputError(
                f"sample {sample_id}: image_score must be max_pixel_probability, got {definition!r}"
            )

        map_path = _resolve(row.get("probability_map_path") or row.get("probability_map"), root)
        probability_map: np.ndarray | None = None
        if map_path is not None:
            if not map_path.exists():
                raise EvaluationInputError(f"sample {sample_id}: probability map missing: {map_path}")
            probability_map = _read_probability_map(map_path)
            computed_score = float(probability_map.max(initial=0.0))
            supplied_raw = str(row.get("image_score", "")).strip()
            if supplied_raw:
                supplied_score = float(supplied_raw)
                if not np.isclose(supplied_score, computed_score, atol=1e-6):
                    raise EvaluationInputError(
                        f"sample {sample_id}: image_score {supplied_score} does not equal "
                        f"max(probability_map)={computed_score}"
                    )
            image_score = computed_score
        else:
            raw_score = str(row.get("image_score", "")).strip()
            if not raw_score:
                raise EvaluationInputError(
                    f"sample {sample_id}: image_score or probability_map_path is required"
                )
            image_score = float(raw_score)

        if not np.isfinite(image_score) or not 0.0 <= image_score <= 1.0:
            raise EvaluationInputError(
                f"sample {sample_id}: image_score must be finite and in [0, 1]"
            )

        raw_mask_path = _resolve(row.get("raw_mask_path") or row.get("raw_mask"), root)
        filtered_mask_path = _resolve(
            row.get("filtered_mask_path") or row.get("filtered_mask"), root
        )
        for label, mask_path in (("raw", raw_mask_path), ("filtered", filtered_mask_path)):
            if mask_path is not None and not mask_path.exists():
                raise EvaluationInputError(
                    f"sample {sample_id}: {label} mask missing: {mask_path}"
                )

        predictions.append(
            PredictionRecord(
                sample_id=sample_id,
                image_score=float(image_score),
                probability_map_path=map_path,
                raw_mask_path=raw_mask_path,
                filtered_mask_path=filtered_mask_path,
                score_definition="max_pixel_probability",
            )
        )

    return predictions


def _read_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise EvaluationInputError(f"Mask could not be decoded: {path}")
    return (mask > 0).astype(np.uint8)


def _read_manifest_mask(path: str | Path | None, data_root: Path | None) -> np.ndarray:
    resolved = resolve_manifest_path(path, data_root)
    if resolved is None or not resolved.exists():
        raise EvaluationInputError(f"Ground-truth mask missing or empty: {path!r}")
    return _read_binary_mask(resolved)


def _align_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    return (
        cv2.resize(
            np.asarray(mask, dtype=np.uint8),
            (int(shape[1]), int(shape[0])),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    ).astype(np.uint8)


def _postprocess_mask(raw_mask: np.ndarray, probability_map: np.ndarray | None) -> np.ndarray:
    """Reproduce the deployed connected-component highlight filter.

    Imported lazily so the evaluator remains usable without loading the model.
    """

    from modules.image_manipulation.inference import _extract_regions

    if probability_map is None:
        probability_map = raw_mask.astype(np.float32)
    _, cleaned = _extract_regions(
        probability_map,
        raw_mask.astype(np.uint8) * 255,
        min_region_area_ratio=0.0002,
        max_regions=25,
    )
    return (cleaned > 0).astype(np.uint8)


@dataclass
class PixelAccumulator:
    images: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    ious: list[float] = field(default_factory=list)
    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    f1: list[float] = field(default_factory=list)
    predicted_area: list[float] = field(default_factory=list)
    ground_truth_area: list[float] = field(default_factory=list)

    def add(self, prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
        pred = np.asarray(prediction).astype(bool)
        gt = np.asarray(ground_truth).astype(bool)
        if pred.shape != gt.shape:
            raise EvaluationInputError(
                f"Prediction shape {pred.shape} does not match mask shape {gt.shape}"
            )
        tp = int(np.logical_and(pred, gt).sum())
        fp = int(np.logical_and(pred, ~gt).sum())
        fn = int(np.logical_and(~pred, gt).sum())
        tn = int(np.logical_and(~pred, ~gt).sum())
        eps = 1e-9
        iou = tp / (tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        total = max(1, int(gt.size))
        self.images += 1
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.ious.append(float(iou))
        self.precision.append(float(precision))
        self.recall.append(float(recall))
        self.f1.append(float(f1))
        self.predicted_area.append(float(pred.sum()) / total)
        self.ground_truth_area.append(float(gt.sum()) / total)
        return {"iou": float(iou), "precision": float(precision), "recall": float(recall), "f1": float(f1)}

    def summary(self) -> dict[str, Any]:
        if self.images == 0:
            return {"status": "not_available", "images": 0}
        eps = 1e-9
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        return {
            "status": "available",
            "images": int(self.images),
            "tp": int(self.tp),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "tn": int(self.tn),
            "global_iou": float(self.tp / (self.tp + self.fp + self.fn + eps)),
            "mean_image_iou": float(np.mean(self.ious)),
            "pixel_precision": float(precision),
            "pixel_recall": float(recall),
            "pixel_f1": float(f1),
            "mean_image_precision": float(np.mean(self.precision)),
            "mean_image_recall": float(np.mean(self.recall)),
            "mean_image_f1": float(np.mean(self.f1)),
            "mean_predicted_area_fraction": float(np.mean(self.predicted_area)),
            "mean_ground_truth_area_fraction": float(np.mean(self.ground_truth_area)),
        }


def _image_metrics(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    threshold: float,
    uncertainty_margin: float,
) -> dict[str, Any]:
    if not scores:
        return {"status": "not_available", "images": 0, "reason": "no labelled predictions"}
    scores_array = np.asarray(scores, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions = (scores_array >= threshold).astype(np.int64)
    tp = int(((predictions == 1) & (labels_array == 1)).sum())
    fp = int(((predictions == 1) & (labels_array == 0)).sum())
    fn = int(((predictions == 0) & (labels_array == 1)).sum())
    tn = int(((predictions == 0) & (labels_array == 0)).sum())
    n = int(len(labels_array))
    authentic = int((labels_array == 0).sum())
    tampered = int((labels_array == 1).sum())

    def ratio(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else float(numerator) / float(denominator)

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = (
        None
        if precision is None or recall is None or (precision + recall) == 0
        else 2.0 * precision * recall / (precision + recall)
    )
    fpr = ratio(fp, fp + tn)
    tnr = ratio(tn, tn + fp)
    balanced = None if recall is None or tnr is None else (recall + tnr) / 2.0
    auc = roc_auc(scores_array, labels_array) if authentic and tampered else float("nan")
    uncertain = int((np.abs(scores_array - threshold) <= max(0.0, uncertainty_margin)).sum())

    status = "available" if authentic and tampered else "partial"
    return {
        "status": status,
        "images": n,
        "authentic_images": authentic,
        "tampered_images": tampered,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": float((tp + tn) / n),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "true_negative_rate": tnr,
        "balanced_accuracy": balanced,
        "roc_auc": None if np.isnan(auc) else float(auc),
        "uncertain_images": uncertain,
        "uncertain_rate": float(uncertain / n),
        "threshold": float(threshold),
        "uncertainty_margin": float(uncertainty_margin),
    }


def _prediction_raw_mask(
    prediction: PredictionRecord,
    *,
    localization_threshold: float,
    derive_filtered: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    probability_map: np.ndarray | None = None
    raw_mask: np.ndarray | None = None
    if prediction.probability_map_path is not None:
        probability_map = _read_probability_map(prediction.probability_map_path)
        raw_mask = (probability_map >= localization_threshold).astype(np.uint8)
    elif prediction.raw_mask_path is not None:
        raw_mask = _read_binary_mask(prediction.raw_mask_path)

    filtered_mask: np.ndarray | None = None
    if prediction.filtered_mask_path is not None:
        filtered_mask = _read_binary_mask(prediction.filtered_mask_path)
    elif derive_filtered and raw_mask is not None:
        filtered_mask = _postprocess_mask(raw_mask, probability_map)
    return probability_map, raw_mask, filtered_mask


def evaluate_per_source(
    records: Sequence[Mapping[str, Any]],
    predictions: str | Path | Sequence[PredictionRecord],
    *,
    data_root: str | Path | None = None,
    image_threshold: float = settings.image_level_threshold,
    localization_threshold: float = settings.manipulation_threshold,
    uncertainty_margin: float = settings.image_uncertainty_margin,
    derive_filtered: bool = True,
    manifest_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate every requested source group without inventing missing results."""

    root = Path(data_root).resolve() if data_root is not None else None
    normalised = [normalise_record(record) for record in records]

    validation = validate_manifest(normalised, data_root=root, strict=False, check_paths=False)
    if not validation.ok:
        validation.raise_for_errors()

    prediction_records = (
        load_predictions(predictions) if isinstance(predictions, (str, Path)) else list(predictions)
    )
    record_by_id: dict[str, dict[str, Any]] = {}
    for record in normalised:
        sample_id = str(record["sample_id"])
        if sample_id in record_by_id:
            raise EvaluationInputError(f"Duplicate manifest sample_id: {sample_id}")
        record_by_id[sample_id] = record
    prediction_by_id = {prediction.sample_id: prediction for prediction in prediction_records}
    unknown_predictions = sorted(set(prediction_by_id) - set(record_by_id))
    if unknown_predictions:
        raise EvaluationInputError(
            "Predictions reference sample IDs absent from the manifest: "
            + ", ".join(unknown_predictions[:25])
        )

    per_sample: list[dict[str, Any]] = []
    group_payloads: dict[str, Any] = {}
    unavailable: list[dict[str, str]] = []

    for report_name, members, description in SOURCE_REPORTS:
        group_records = [record for record in normalised if record["source_group"] in members]
        label_records = [
            record for record in group_records if record["class_name"] in {"Authentic", "Tampered"}
        ]
        with_predictions = [
            record for record in label_records if record["sample_id"] in prediction_by_id
        ]
        missing_predictions = [
            record for record in label_records if record["sample_id"] not in prediction_by_id
        ]
        scores = [prediction_by_id[record["sample_id"]].image_score for record in with_predictions]
        labels = [1 if record["class_name"] == "Tampered" else 0 for record in with_predictions]
        image_level = _image_metrics(
            scores,
            labels,
            threshold=image_threshold,
            uncertainty_margin=uncertainty_margin,
        )

        if not label_records:
            status = "not_available"
            reason = "no Authentic/Tampered samples in this source group"
        elif not with_predictions:
            status = "not_available"
            reason = "no predictions for the labelled samples in this source group"
        elif missing_predictions:
            status = "partial"
            reason = f"{len(missing_predictions)} labelled sample(s) have no prediction"
        elif image_level["status"] == "partial":
            status = "partial"
            reason = "only one class is present in the evaluated predictions"
        else:
            status = "available"
            reason = ""

        raw_accumulator = PixelAccumulator()
        filtered_accumulator = PixelAccumulator()
        pixel_missing_maps: list[str] = []
        raw_present = 0
        filtered_present = 0

        for record in label_records:
            if record["class_name"] != "Tampered":
                continue
            prediction = prediction_by_id.get(record["sample_id"])
            if prediction is None:
                continue
            _, raw_mask, filtered_mask = _prediction_raw_mask(
                prediction,
                localization_threshold=localization_threshold,
                derive_filtered=derive_filtered,
            )
            if raw_mask is None and filtered_mask is None:
                pixel_missing_maps.append(str(record["sample_id"]))
                continue
            try:
                ground_truth = _read_manifest_mask(record.get("mask_path"), root)
            except EvaluationInputError as exc:
                raise EvaluationInputError(
                    f"sample {record['sample_id']}: {exc}"
                ) from exc
            shape = ground_truth.shape
            sample_row: dict[str, Any] = {
                "report_group": report_name,
                "sample_id": record["sample_id"],
                "source_group": record["source_group"],
                "image_score": float(prediction.image_score),
                "ground_truth_area_fraction": float(ground_truth.mean()),
            }
            if raw_mask is not None:
                raw_present += 1
                aligned = _align_mask(raw_mask, shape)
                metrics = raw_accumulator.add(aligned, ground_truth)
                sample_row.update(
                    {
                        "raw_iou": metrics["iou"],
                        "raw_precision": metrics["precision"],
                        "raw_recall": metrics["recall"],
                        "raw_f1": metrics["f1"],
                        "raw_area_fraction": float(aligned.mean()),
                    }
                )
            if filtered_mask is not None:
                filtered_present += 1
                aligned = _align_mask(filtered_mask, shape)
                metrics = filtered_accumulator.add(aligned, ground_truth)
                sample_row.update(
                    {
                        "filtered_iou": metrics["iou"],
                        "filtered_precision": metrics["precision"],
                        "filtered_recall": metrics["recall"],
                        "filtered_f1": metrics["f1"],
                        "filtered_area_fraction": float(aligned.mean()),
                    }
                )
            per_sample.append(sample_row)

        tampered_in_group = sum(1 for record in label_records if record["class_name"] == "Tampered")
        pixel_level = {
            "status": (
                "available"
                if raw_accumulator.images or filtered_accumulator.images
                else "not_available"
            ),
            "tampered_samples_in_group": int(tampered_in_group),
            "tampered_samples_with_predictions": int(
                sum(
                    1
                    for record in label_records
                    if record["class_name"] == "Tampered" and record["sample_id"] in prediction_by_id
                )
            ),
            "raw_masks_present": int(raw_present),
            "filtered_masks_present": int(filtered_present),
            "raw_thresholded": raw_accumulator.summary(),
            "post_filtered_highlighted": filtered_accumulator.summary(),
            "samples_without_prediction_masks": pixel_missing_maps[:25],
            "notes": [
                "Raw thresholded masks use the localization threshold and are not post-filtered.",
                "Post-filtered highlighted masks are connected-component filtered and are reported separately.",
                "Pixel metrics use tampered images with valid ground-truth masks only.",
            ],
        }
        if not raw_accumulator.images and not filtered_accumulator.images:
            pixel_level["reason"] = (
                "no tampered sample with a raw or filtered prediction mask"
            )

        group_payload = {
            "status": status,
            "description": description,
            "source_groups": list(members),
            "reason": reason or None,
            "sample_coverage": {
                "manifest_labelled_samples": int(len(label_records)),
                "evaluated_samples": int(len(with_predictions)),
                "missing_predictions": int(len(missing_predictions)),
                "missing_prediction_sample_ids": [str(record["sample_id"]) for record in missing_predictions[:25]],
                "unknown_class_samples": int(
                    sum(1 for record in group_records if record["class_name"] == "Unknown")
                ),
            },
            "image_level": image_level,
            "pixel_level": pixel_level,
        }
        if status == "not_available":
            unavailable.append({"source_group": report_name, "reason": reason})
        group_payloads[report_name] = group_payload

    return {
        "artifact": "module1_by_source_evaluation",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "module": "MVSS-Net image manipulation localization",
        "manifest": str(manifest_path or ""),
        "predictions": str(predictions_path or ""),
        "parameters": {
            "image_threshold": float(image_threshold),
            "localization_threshold": float(localization_threshold),
            "uncertainty_margin": float(uncertainty_margin),
            "image_score": "maximum pixel probability",
            "derive_filtered": bool(derive_filtered),
        },
        "manifest_validation": validation.as_dict(),
        "prediction_rows": int(len(prediction_records)),
        "source_groups": group_payloads,
        "unavailable_source_groups": unavailable,
        "per_sample": per_sample,
        "definitions": {
            "image_level": (
                "Binary metrics threshold the maximum pixel probability at image_threshold. "
                "The uncertain rate counts scores within +/- uncertainty_margin and does not "
                "replace the binary accuracy."
            ),
            "pixel_level": (
                "Global IoU is micro total TP/(TP+FP+FN); mean image IoU is the macro mean. "
                "Raw thresholded and post-filtered highlighted masks are accumulated separately."
            ),
            "source_groups": {
                "dslr_mirrorless": "DSLR and mirrorless cameras",
                "android": "Android smartphones",
                "iphone": "iPhones",
                "other_mobile": "Other mobile devices",
                "combined_camera": "Derived from dslr_mirrorless",
                "combined_mobile": "Derived from android + iphone + other_mobile",
                "all_sources": "All canonical groups, including unknown; image metrics exclude Unknown class",
            },
        },
        "notes": [
            "No metrics are reported for groups without real labelled predictions.",
            "Unknown-class phone photographs are retained in the manifest but are not treated as authentic ground truth.",
            "This tool does not train, fine-tune, or alter the checkpoint.",
        ],
    }


def write_evaluation_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
