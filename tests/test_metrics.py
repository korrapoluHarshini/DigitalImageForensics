from __future__ import annotations

import numpy as np
import pytest

from utils.metrics import (
    PixelCounts,
    SegmentationAccumulator,
    classification_metrics,
    image_level_metrics,
    pixel_counts,
    roc_auc,
)


def test_pixel_counts_known_case():
    prediction = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    ground_truth = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    counts = pixel_counts(prediction, ground_truth)
    assert (counts.tp, counts.fp, counts.fn, counts.tn) == (1, 1, 1, 1)
    assert counts.iou == pytest.approx(1 / 3)
    assert counts.precision == pytest.approx(0.5)
    assert counts.recall == pytest.approx(0.5)
    assert counts.f1 == pytest.approx(0.5)


def test_pixel_counts_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        pixel_counts(np.zeros((4, 4)), np.zeros((4, 5)))


def test_accumulator_micro_and_macro():
    accumulator = SegmentationAccumulator()
    accumulator.add(np.array([[1, 1], [0, 0]]), np.array([[1, 0], [1, 0]]))
    accumulator.add(np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 1]]))
    summary = accumulator.summary()
    assert summary["images"] == 2
    assert summary["mean_image_iou"] == pytest.approx((1 / 3 + 1.0) / 2)
    # Micro: TP=5, FP=1, FN=1 -> IoU = 5/7
    assert summary["global_iou"] == pytest.approx(5 / 7)
    assert summary["micro_precision"] == pytest.approx(5 / 6)
    assert summary["micro_recall"] == pytest.approx(5 / 6)


def test_segmentation_summary_empty():
    assert SegmentationAccumulator().summary() == {"images": 0}


def test_image_level_metrics():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    metrics = image_level_metrics(scores, labels, threshold=0.5)
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert metrics["false_positive_rate"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_image_level_false_positive():
    metrics = image_level_metrics([0.9, 0.6], [1, 0], threshold=0.5)
    assert metrics["fp"] == 1
    assert metrics["false_positive_rate"] == pytest.approx(1.0)


def test_roc_auc_perfect_and_inverted():
    assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == pytest.approx(1.0)
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([1, 1, 0, 0])) == pytest.approx(0.0)
    assert np.isnan(roc_auc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_classification_metrics():
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    metrics = classification_metrics(labels, predictions, 2, ["real", "ai_generated"])
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["per_class"]["ai_generated"]["recall"] == pytest.approx(1.0)


def test_pixel_counts_dict():
    counts = PixelCounts(tp=1, fp=1, fn=1, tn=1)
    payload = counts.as_dict()
    assert payload["tp"] == 1
    assert payload["iou"] == pytest.approx(1 / 3)
