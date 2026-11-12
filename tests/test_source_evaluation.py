from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dataset_manifest.evaluation import (
    EvaluationInputError,
    PredictionRecord,
    evaluate_per_source,
    load_predictions,
)
from dataset_manifest.schema import normalise_record


def _image(path: Path, value: int = 40) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((20, 20, 3), dtype=np.uint8)
    array[..., 0] = value
    array[2:8, 3:9] = (210, 50, 40)
    Image.fromarray(array).save(path)
    return path


def _mask(path: Path, region: tuple[int, int, int, int], size: tuple[int, int] = (10, 10)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros(size, dtype=np.uint8)
    x, y, w, h = region
    array[y : y + h, x : x + w] = 255
    Image.fromarray(array).save(path)
    return path


def _record(
    sample_id: str,
    source_group: str,
    class_name: str,
    image: Path,
    *,
    score: float | None = None,
    mask: Path | None = None,
):
    del score
    return normalise_record(
        {
            "sample_id": sample_id,
            "dataset": "synthetic",
            "source_group": source_group,
            "split": "test",
            "class_name": class_name,
            "image_path": str(image),
            "mask_path": str(mask) if mask else "",
            "pair_id": f"pair-{sample_id}" if mask else "",
            "group_id": f"group-{sample_id}",
            "manipulation_type": "synthetic" if mask else "",
            "device_make": "",
            "device_model": "",
            "capture_type": "smartphone" if "mobile" in source_group or source_group in {"android", "iphone"} else "camera",
            "width": 20,
            "height": 20,
            "format": "png",
            "mask_available": bool(mask),
            "mask_type": "binary_png" if mask else "",
            "mask_quality": "synthetic" if mask else "",
            "license": "test-only",
            "provenance_url": "",
            "sha256": "",
            "phash": "",
            "notes": "synthetic fixture; metrics are not real-world evidence",
            "pixel_training_eligible": False,
        }
    )


def _fixture(tmp_path: Path):
    image = _image(tmp_path / "image.png")

    gt_b = _mask(tmp_path / "gt_b.png", (2, 2, 4, 4))
    raw_b = _mask(tmp_path / "raw_b.png", (1, 1, 6, 6))
    filtered_b = gt_b
    gt_d = _mask(tmp_path / "gt_d.png", (2, 2, 5, 5))
    raw_d = gt_d
    filtered_d = _mask(tmp_path / "filtered_d.png", (2, 2, 3, 3))

    records = [
        _record("a", "dslr_mirrorless", "Authentic", image),
        _record("b", "dslr_mirrorless", "Tampered", image, mask=gt_b),
        _record("c", "android", "Authentic", image),
        _record("d", "android", "Tampered", image, mask=gt_d),
        _record("e", "iphone", "Authentic", image),
        _record("f", "iphone", "Tampered", image, mask=gt_d),
        _record("g", "other_mobile", "Authentic", image),
        _record("h", "unknown", "Unknown", image),
    ]
    predictions = [
        PredictionRecord("a", 0.10),
        PredictionRecord("b", 0.90, raw_mask_path=raw_b, filtered_mask_path=filtered_b),
        PredictionRecord("c", 0.41),
        PredictionRecord("d", 0.80, raw_mask_path=raw_d, filtered_mask_path=filtered_d),
        PredictionRecord("e", 0.60),
        PredictionRecord("f", 0.20),
        PredictionRecord("g", 0.05),
        PredictionRecord("h", 0.99),
    ]
    return records, predictions, gt_b, raw_b, filtered_b, gt_d, raw_d, filtered_d


def test_image_level_metrics_per_source_and_combined(tmp_path):
    records, predictions, *_ = _fixture(tmp_path)
    payload = evaluate_per_source(records, predictions, derive_filtered=False)
    groups = payload["source_groups"]

    assert groups["dslr_mirrorless"]["status"] == "available"
    camera = groups["dslr_mirrorless"]["image_level"]
    assert camera["accuracy"] == pytest.approx(1.0)
    assert camera["false_positive_rate"] == pytest.approx(0.0)
    assert camera["roc_auc"] == pytest.approx(1.0)
    assert camera["uncertain_rate"] == pytest.approx(0.0)

    android = groups["android"]["image_level"]
    assert android["accuracy"] == pytest.approx(1.0)
    assert android["uncertain_rate"] == pytest.approx(0.5)

    iphone = groups["iphone"]["image_level"]
    assert iphone["accuracy"] == pytest.approx(0.0)
    assert iphone["false_positive_rate"] == pytest.approx(1.0)
    assert iphone["recall"] == pytest.approx(0.0)
    assert iphone["balanced_accuracy"] == pytest.approx(0.0)

    other = groups["other_mobile"]["image_level"]
    assert other["status"] == "partial"
    assert other["recall"] is None
    assert other["roc_auc"] is None

    combined_mobile = groups["combined_mobile"]["image_level"]
    assert combined_mobile["images"] == 5
    assert combined_mobile["accuracy"] == pytest.approx(3 / 5)
    assert combined_mobile["false_positive_rate"] == pytest.approx(1 / 3)
    assert combined_mobile["roc_auc"] == pytest.approx(2 / 3)
    assert combined_mobile["uncertain_rate"] == pytest.approx(1 / 5)

    all_sources = groups["all_sources"]["image_level"]
    assert all_sources["images"] == 7
    assert all_sources["accuracy"] == pytest.approx(5 / 7)
    assert all_sources["precision"] == pytest.approx(2 / 3)
    assert all_sources["recall"] == pytest.approx(2 / 3)
    assert all_sources["false_positive_rate"] == pytest.approx(1 / 4)
    assert all_sources["roc_auc"] == pytest.approx(10 / 12)
    assert all_sources["uncertain_rate"] == pytest.approx(1 / 7)
    assert groups["all_sources"]["sample_coverage"]["unknown_class_samples"] == 1


def test_pixel_metrics_keep_raw_and_filtered_separate(tmp_path):
    records, predictions, *_ = _fixture(tmp_path)
    payload = evaluate_per_source(records, predictions, derive_filtered=False)
    camera = payload["source_groups"]["dslr_mirrorless"]["pixel_level"]
    android = payload["source_groups"]["android"]["pixel_level"]
    mobile = payload["source_groups"]["combined_mobile"]["pixel_level"]

    assert camera["raw_thresholded"]["status"] == "available"
    assert camera["post_filtered_highlighted"]["status"] == "available"
    assert camera["raw_thresholded"]["global_iou"] == pytest.approx(16 / 36)
    assert camera["post_filtered_highlighted"]["global_iou"] == pytest.approx(1.0)
    assert camera["raw_thresholded"]["pixel_precision"] == pytest.approx(16 / 36)
    assert camera["raw_thresholded"]["pixel_recall"] == pytest.approx(1.0)
    assert camera["raw_thresholded"]["pixel_f1"] == pytest.approx(2 * (16 / 36) / (1 + 16 / 36))

    assert android["raw_thresholded"]["global_iou"] == pytest.approx(1.0)
    assert android["post_filtered_highlighted"]["global_iou"] == pytest.approx(9 / 25)

    assert mobile["raw_thresholded"]["images"] == 1
    assert mobile["post_filtered_highlighted"]["images"] == 1
    assert mobile["raw_thresholded"]["global_iou"] == pytest.approx(1.0)
    assert mobile["post_filtered_highlighted"]["global_iou"] == pytest.approx(9 / 25)

    all_sources = payload["source_groups"]["all_sources"]["pixel_level"]
    assert all_sources["raw_thresholded"]["images"] == 2
    assert all_sources["post_filtered_highlighted"]["images"] == 2
    assert all_sources["raw_thresholded"]["global_iou"] == pytest.approx(41 / 61)
    assert all_sources["post_filtered_highlighted"]["global_iou"] == pytest.approx(25 / 41)
    assert all_sources["raw_thresholded"]["mean_image_iou"] == pytest.approx((16 / 36 + 1.0) / 2)
    assert all_sources["post_filtered_highlighted"]["mean_image_iou"] == pytest.approx((1.0 + 9 / 25) / 2)
    assert camera["raw_thresholded"]["mean_predicted_area_fraction"] != pytest.approx(
        camera["post_filtered_highlighted"]["mean_predicted_area_fraction"]
    )


def test_missing_predictions_make_group_partial_and_are_listed(tmp_path):
    records, predictions, *_ = _fixture(tmp_path)
    filtered = [prediction for prediction in predictions if prediction.sample_id != "f"]
    payload = evaluate_per_source(records, filtered, derive_filtered=False)
    iphone = payload["source_groups"]["iphone"]
    assert iphone["status"] == "partial"
    assert iphone["sample_coverage"]["missing_predictions"] == 1
    assert iphone["sample_coverage"]["missing_prediction_sample_ids"] == ["f"]
    assert iphone["image_level"]["recall"] is None


def test_unavailable_source_groups_are_explicit(tmp_path):
    image = _image(tmp_path / "image.png")
    records = [
        _record("c", "android", "Authentic", image),
        _record("d", "android", "Tampered", image, mask=_mask(tmp_path / "gt.png", (2, 2, 4, 4))),
    ]
    predictions = [
        PredictionRecord("c", 0.10),
        PredictionRecord("d", 0.90),
    ]
    payload = evaluate_per_source(records, predictions, derive_filtered=False)
    assert payload["source_groups"]["iphone"]["status"] == "not_available"
    assert payload["source_groups"]["iphone"]["image_level"]["status"] == "not_available"
    assert "iphone" in {
        entry["source_group"] for entry in payload["unavailable_source_groups"]
    }


def test_tampered_prediction_without_ground_truth_mask_fails_loudly(tmp_path):
    image = _image(tmp_path / "image.png")
    raw = _mask(tmp_path / "raw.png", (1, 1, 4, 4))
    record = _record("t", "android", "Tampered", image)
    record["mask_available"] = False
    record["mask_path"] = ""
    with pytest.raises(EvaluationInputError, match="Ground-truth mask"):
        evaluate_per_source(
            [record],
            [PredictionRecord("t", 0.9, raw_mask_path=raw)],
            derive_filtered=False,
        )


def test_probability_map_score_must_be_max_pixel(tmp_path):
    map_path = tmp_path / "probability.npy"
    np.save(map_path, np.array([[0.1, 0.2], [0.3, 0.9]], dtype=np.float32))
    predictions_csv = tmp_path / "predictions.csv"
    with predictions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_score", "probability_map_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "s1",
                "image_score": "0.9",
                "probability_map_path": str(map_path),
            }
        )
    loaded = load_predictions(predictions_csv)
    assert loaded[0].image_score == pytest.approx(0.9)

    with predictions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_score", "probability_map_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "s1",
                "image_score": "0.8",
                "probability_map_path": str(map_path),
            }
        )
    with pytest.raises(EvaluationInputError, match="does not equal"):
        load_predictions(predictions_csv)


def test_score_definition_is_validated(tmp_path):
    predictions_csv = tmp_path / "predictions.csv"
    with predictions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_score", "score_definition"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "s1",
                "image_score": "0.5",
                "score_definition": "classifier_probability",
            }
        )
    with pytest.raises(EvaluationInputError, match="max_pixel_probability"):
        load_predictions(predictions_csv)


def test_derived_filtered_mask_is_reported_separately_from_raw(tmp_path):
    image = _image(tmp_path / "image.png")
    gt = _mask(tmp_path / "gt.png", (10, 10, 20, 20), size=(100, 100))
    probability = np.zeros((100, 100), dtype=np.float32)
    probability[10:30, 10:30] = 0.9
    probability[80, 80] = 0.9  # isolated pixel dropped by the deployed filter
    map_path = tmp_path / "map.npy"
    np.save(map_path, probability)
    record = _record("t", "android", "Tampered", image, mask=gt)
    payload = evaluate_per_source(
        [record],
        [
            PredictionRecord(
                "t",
                float(probability.max()),
                probability_map_path=map_path,
            )
        ],
        localization_threshold=0.4,
        derive_filtered=True,
    )
    pixel = payload["source_groups"]["android"]["pixel_level"]
    assert pixel["raw_thresholded"]["status"] == "available"
    assert pixel["post_filtered_highlighted"]["status"] == "available"
    assert (
        pixel["raw_thresholded"]["mean_predicted_area_fraction"]
        > pixel["post_filtered_highlighted"]["mean_predicted_area_fraction"]
    )
    assert pixel["raw_thresholded"]["global_iou"] < pixel["post_filtered_highlighted"]["global_iou"]
