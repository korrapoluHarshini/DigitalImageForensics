"""Metric helpers used by the evaluation scripts and tests.

Definitions are kept explicit so that reported numbers can be reproduced:

* ``global_iou``  - micro IoU: total TP / total union over all images.
* ``mean_iou``    - macro IoU: mean of per-image IoU.
* ``precision`` / ``recall`` / ``f1`` - available as micro and macro variants.
* image-level metrics treat an image as manipulated when its maximum pixel
  probability is greater than or equal to the configured threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


EPS = 1e-9


@dataclass
class PixelCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def iou(self) -> float:
        return self.tp / (self.tp + self.fp + self.fn + EPS)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + EPS)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + EPS)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2.0 * p * r / (p + r + EPS)

    def as_dict(self) -> dict:
        return {
            "tp": int(self.tp),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "tn": int(self.tn),
            "iou": float(self.iou),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
        }


def pixel_counts(prediction: np.ndarray, ground_truth: np.ndarray) -> PixelCounts:
    """Count TP/FP/FN/TN for two binary arrays of the same shape."""
    pred = np.asarray(prediction).astype(bool)
    gt = np.asarray(ground_truth).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} vs mask {gt.shape}")
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return PixelCounts(tp=tp, fp=fp, fn=fn, tn=tn)


@dataclass
class SegmentationAccumulator:
    """Accumulates pixel counts and per-image scores."""

    counts: PixelCounts = field(default_factory=PixelCounts)
    per_image_iou: list[float] = field(default_factory=list)
    per_image_precision: list[float] = field(default_factory=list)
    per_image_recall: list[float] = field(default_factory=list)
    per_image_f1: list[float] = field(default_factory=list)
    images: int = 0

    def add(self, prediction: np.ndarray, ground_truth: np.ndarray) -> PixelCounts:
        image_counts = pixel_counts(prediction, ground_truth)
        self.counts.tp += image_counts.tp
        self.counts.fp += image_counts.fp
        self.counts.fn += image_counts.fn
        self.counts.tn += image_counts.tn
        self.per_image_iou.append(image_counts.iou)
        self.per_image_precision.append(image_counts.precision)
        self.per_image_recall.append(image_counts.recall)
        self.per_image_f1.append(image_counts.f1)
        self.images += 1
        return image_counts

    def summary(self) -> dict:
        if self.images == 0:
            return {"images": 0}
        return {
            "images": int(self.images),
            "global_iou": float(self.counts.iou),
            "mean_image_iou": float(np.mean(self.per_image_iou)),
            "micro_precision": float(self.counts.precision),
            "micro_recall": float(self.counts.recall),
            "micro_f1": float(self.counts.f1),
            "mean_image_precision": float(np.mean(self.per_image_precision)),
            "mean_image_recall": float(np.mean(self.per_image_recall)),
            "mean_image_f1": float(np.mean(self.per_image_f1)),
            "std_image_iou": float(np.std(self.per_image_iou)),
        }


def image_level_metrics(
    scores: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
    threshold: float,
) -> dict:
    """Image-level detection metrics.

    ``labels``: 0 = authentic, 1 = manipulated.
    ``scores``: maximum pixel probability (higher means more suspicious).
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if scores_arr.shape != labels_arr.shape:
        raise ValueError("scores and labels must have the same length")
    predictions = (scores_arr >= threshold).astype(np.int64)

    tp = int(((predictions == 1) & (labels_arr == 1)).sum())
    fp = int(((predictions == 1) & (labels_arr == 0)).sum())
    fn = int(((predictions == 0) & (labels_arr == 1)).sum())
    tn = int(((predictions == 0) & (labels_arr == 0)).sum())

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    fpr = fp / (fp + tn + EPS)
    tnr = tn / (tn + fp + EPS)
    accuracy = (tp + tn) / max(1, len(labels_arr))

    return {
        "threshold": float(threshold),
        "images": int(len(labels_arr)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "true_negative_rate": float(tnr),
        "balanced_accuracy": float((recall + tnr) / 2.0),
    }


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC AUC (no sklearn dependency)."""
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    positives = labels_arr == 1
    negatives = ~positives
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores_arr, kind="mergesort")
    ranks = np.empty(len(scores_arr), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores_arr) + 1, dtype=np.float64)

    # Average ranks for ties.
    sorted_scores = scores_arr[order]
    start = 0
    for index in range(1, len(sorted_scores) + 1):
        if index == len(sorted_scores) or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                average = (start + 1 + index) / 2.0
                ranks[order[start:index]] = average
            start = index

    rank_sum = ranks[positives].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def confusion_matrix(predictions: np.ndarray, labels: np.ndarray, num_classes: int = 2) -> list[list[int]]:
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for label, prediction in zip(np.asarray(labels), np.asarray(predictions)):
        label = int(label)
        prediction = int(prediction)
        if 0 <= label < num_classes and 0 <= prediction < num_classes:
            matrix[label][prediction] += 1
    return matrix


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    class_names: list[str] | None = None,
) -> dict:
    """Accuracy, macro precision/recall/F1 and the confusion matrix."""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    names = class_names or [str(i) for i in range(num_classes)]
    matrix = confusion_matrix(predictions, labels, num_classes)

    per_class = {}
    precisions, recalls, f1s = [], [], []
    for index, name in enumerate(names):
        tp = matrix[index][index] if index < len(matrix) else 0
        fp = sum(matrix[row][index] for row in range(num_classes) if row != index)
        fn = sum(matrix[index][col] for col in range(num_classes) if col != index)
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        f1 = 2 * precision * recall / (precision + recall + EPS)
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(sum(matrix[index])),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": float(np.mean(predictions == labels)) if len(labels) else 0.0,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "class_names": names,
        "samples": int(len(labels)),
    }
