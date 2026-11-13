"""Independently evaluate a Module 1 MVSS-Net checkpoint on the group-aware split.

The group-aware split (``results/manifests/casia2_groupaware_split.csv``) confines
every cross-split duplicate component to a single split, so a checkpoint trained
on its ``train`` rows can be evaluated on its ``test`` rows without the leakage
that affects the historical CASIA split.

This script only evaluates.  It never selects a threshold, never trains and never
uses the test split to choose anything.  The pixel operating threshold is the
one already selected on the historical validation split
(``utils.config.settings.manipulation_threshold``); the image-level threshold is
``settings.image_level_threshold`` with the uncertainty band
``settings.image_uncertainty_margin``.  Both are reported with their provenance.

Preprocessing matches the Streamlit application exactly: images are read in BGR
order and normalised with ImageNet statistics via
``modules.image_manipulation.preprocessing.preprocess_bgr``.  Ground-truth masks
are resized with nearest-neighbour interpolation and binarised at 127.

Usage
-----
Evaluate a trained group-aware checkpoint on the leakage-free test split::

    python evaluation/evaluate_groupaware_checkpoint.py \
        --checkpoint experiments/module1/groupaware_v1/mvssnet_groupaware.pt \
        --split test \
        --output results/evaluation/module1_groupaware_test_metrics.json

Quick plumbing check on a few images (results are explicitly marked ``smoke``
and must not be quoted as evaluation results)::

    python evaluation/evaluate_groupaware_checkpoint.py --smoke

Outputs
-------
JSON summary (default ``results/evaluation/module1_groupaware_<split>_metrics.json``)
and optionally a per-image CSV with ``--per-image-csv``.  Probability maps can be
cached with ``--save-probabilities`` (float16 ``.npy`` per sample) to avoid
re-running inference.

Notes
-----
* Thresholded masks produced here are **raw** model threshold outputs.  The
  application's post-filtered highlight overlays are a separate visual stage.
* If fewer than all rows are evaluated (``--limit-*`` or ``--smoke``), the JSON
  records ``sampled`` counts and a warning; sampled metrics are not comparable to
  a full-split run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.image_manipulation.preprocessing import preprocess_bgr  # noqa: E402
from models.mvssnet import get_mvss  # noqa: E402
from training.train_mvss_groupaware import build_records  # noqa: E402
from utils.config import settings  # noqa: E402

DEFAULT_MANIFEST = PROJECT_ROOT / "results" / "manifests" / "casia2_manifest.csv"
DEFAULT_SPLIT = PROJECT_ROOT / "results" / "manifests" / "casia2_groupaware_split.csv"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _load_state_dict(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"No state dict found in {path}")
    if all(str(key).startswith("module.") for key in payload):
        payload = {str(key)[7:]: value for key, value in payload.items()}
    return payload


def _build_model(device: torch.device) -> torch.nn.Module:
    model = get_mvss(
        backbone="resnet50",
        pretrained_base=False,
        nclass=1,
        sobel=True,
        constrain=True,
        n_input=3,
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class EvaluationDataset(Dataset):
    def __init__(self, records, *, image_size: int):
        self.records = records
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        bgr = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Image could not be decoded: {record.image_path}")
        tensor = preprocess_bgr(bgr, size=self.image_size)

        if record.class_name == "Tampered" and record.mask_path is not None:
            mask = cv2.imread(str(record.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Mask could not be decoded: {record.mask_path}")
            mask = cv2.resize(
                mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
            )
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        label = 1.0 if record.class_name == "Tampered" else 0.0
        return (
            tensor[0],
            torch.from_numpy(mask).unsqueeze(0),
            torch.tensor([label], dtype=torch.float32),
            record.sample_id,
        )


def _select_records(records, *, split: str, limit_tampered: int, limit_authentic: int, seed: int):
    pool = [record for record in records if record.split == split]
    tampered = sorted(
        (record for record in pool if record.class_name == "Tampered"), key=lambda r: r.sample_id
    )
    authentic = sorted(
        (record for record in pool if record.class_name == "Authentic"), key=lambda r: r.sample_id
    )
    rng = random.Random(seed)
    if limit_tampered > 0 and len(tampered) > limit_tampered:
        tampered = sorted(rng.sample(tampered, limit_tampered), key=lambda r: r.sample_id)
    if limit_authentic > 0 and len(authentic) > limit_authentic:
        authentic = sorted(rng.sample(authentic, limit_authentic), key=lambda r: r.sample_id)
    return tampered, authentic, len(pool)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Rank AUC with average ranks for ties (Mann-Whitney U)."""
    positives = labels == 1
    negatives = labels == 0
    if not positives.any() or not negatives.any():
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    position = 0
    while position < len(sorted_scores):
        end = position
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[position]:
            end += 1
        average_rank = (position + end + 2) / 2.0  # 1-based average rank
        ranks[order[position : end + 1]] = average_rank
        position = end + 1
    positive_ranks = ranks[positives].sum()
    return float(
        (positive_ranks - positives.sum() * (positives.sum() + 1) / 2.0)
        / (positives.sum() * negatives.sum())
    )


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run(args, model_factory=None) -> dict:
    started = time.perf_counter()
    device = _resolve_device(args.device)
    torch.set_num_threads(max(1, args.torch_threads))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data_root = Path(args.data_root).resolve()
    records, excluded = build_records(Path(args.manifest), Path(args.split_csv), data_root)
    tampered, authentic, pool_size = _select_records(
        records,
        split=args.split,
        limit_tampered=args.limit_tampered,
        limit_authentic=args.limit_authentic,
        seed=args.seed,
    )
    if not tampered and not authentic:
        raise ValueError(f"No evaluable rows for split '{args.split}'.")

    selected = tampered + authentic
    dataset = EvaluationDataset(selected, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = (model_factory or _build_model)(device)
    model.load_state_dict(_load_state_dict(checkpoint_path), strict=True)
    model.eval()

    pixel_threshold = float(args.pixel_threshold)
    image_threshold = float(args.image_threshold)
    uncertainty = float(args.image_uncertainty_margin)

    totals = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
    authentic_positive = 0.0
    authentic_pixels = 0.0
    negative_pixels = 0.0
    ious: list[float] = []
    recalls: list[float] = []
    image_scores: list[float] = []
    image_labels: list[float] = []
    per_image: list[dict] = []
    probability_dir = Path(args.save_probabilities) if args.save_probabilities else None
    if probability_dir:
        probability_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for images, masks, labels, sample_ids in tqdm(loader, desc=f"{args.split} evaluation"):
            images = images.to(device)
            masks = masks.to(device)
            _, logits = model(images)
            probabilities = torch.sigmoid(logits)
            predicted = (probabilities >= pixel_threshold).float()

            totals["tp"] += float((predicted * masks).sum())
            totals["fp"] += float((predicted * (1 - masks)).sum())
            totals["fn"] += float(((1 - predicted) * masks).sum())

            flat_scores = F.adaptive_max_pool2d(probabilities, output_size=1).flatten(1)[:, 0]

            for index in range(images.size(0)):
                mask = masks[index]
                probability = probabilities[index]
                identification = str(sample_ids[index])
                label = float(labels[index, 0])
                score = float(flat_scores[index])
                predicted_pixels = float(predicted[index].sum())
                mask_pixels = float(mask.sum())
                intersection = float((predicted[index] * mask).sum())
                union = predicted_pixels + mask_pixels - intersection
                iou = _ratio(intersection, union) if mask_pixels > 0 else None

                if label == 1.0 and mask_pixels > 0:
                    ious.append(iou if iou is not None else 0.0)
                    recalls.append(_ratio(intersection, mask_pixels))
                if label == 0.0:
                    authentic_positive += predicted_pixels
                    authentic_pixels += float(mask.numel())
                negative_pixels += float((1 - mask).sum())

                if probability_dir:
                    np.save(
                        probability_dir / f"{identification}.npy",
                        probability[0].detach().cpu().numpy().astype(np.float16),
                    )

                decision = (
                    "Tampered"
                    if score >= image_threshold + uncertainty
                    else "Authentic"
                    if score <= image_threshold - uncertainty
                    else "Uncertain"
                )

                image_scores.append(score)
                image_labels.append(label)
                per_image.append(
                    {
                        "sample_id": identification,
                        "class_name": "Tampered" if label == 1.0 else "Authentic",
                        "split": args.split,
                        "image_score": round(score, 6),
                        "image_decision": decision,
                        "image_decision_binary": int(score >= image_threshold),
                        "pixel_iou": None if iou is None else round(iou, 6),
                        "pixel_recall": (
                            None
                            if label == 0.0 or mask_pixels <= 0
                            else round(_ratio(intersection, mask_pixels), 6)
                        ),
                        "mask_area_fraction": round(_ratio(mask_pixels, float(mask.numel())), 6),
                        "predicted_area_fraction": round(
                            _ratio(predicted_pixels, float(mask.numel())), 6
                        ),
                    }
                )

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    micro_precision = _ratio(tp, tp + fp)
    micro_recall = _ratio(tp, tp + fn)
    micro_f1 = _ratio(2 * micro_precision * micro_recall, micro_precision + micro_recall)
    micro_iou = _ratio(tp, tp + fp + fn)

    scores_array = np.asarray(image_scores, dtype=float)
    labels_array = np.asarray(image_labels, dtype=float)
    auc = _auc(scores_array, labels_array)
    tampered_mask = labels_array == 1
    authentic_mask = labels_array == 0
    image_recall = _ratio(
        float((scores_array[tampered_mask] >= image_threshold).sum()), float(tampered_mask.sum())
    )
    image_fpr = _ratio(
        float((scores_array[authentic_mask] >= image_threshold).sum()), float(authentic_mask.sum())
    )
    uncertain = int(
        (
            (scores_array > image_threshold - uncertainty)
            & (scores_array < image_threshold + uncertainty)
        ).sum()
    )

    sqrt_n = max(1.0, float(np.sqrt(len(tampered))))
    summary = {
        "artifact": "module1_groupaware_evaluation.json",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "smoke" if args.smoke else "full",
        "split": args.split,
        "device": str(device),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "inputs": {
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": _sha256(Path(args.manifest)),
            "split_csv": str(Path(args.split_csv).resolve()),
            "split_csv_sha256": _sha256(Path(args.split_csv)),
            "data_root": str(data_root),
        },
        "thresholds": {
            "pixel": pixel_threshold,
            "pixel_provenance": settings.manipulation_threshold_source,
            "image": image_threshold,
            "image_uncertainty_margin": uncertainty,
            "image_provenance": settings.image_level_threshold_source,
            "selection": "fixed before evaluation; the test split was not used to choose thresholds",
        },
        "samples": {
            "split_rows_available": pool_size,
            "evaluated": len(selected),
            "tampered": len(tampered),
            "authentic": len(authentic),
            "tampered_with_mask": sum(1 for record in tampered if record.mask_path is not None),
            "excluded_tampered_without_mask": len(
                [row for row in excluded if row["split"] == args.split]
            ),
            "sampled": bool(args.smoke or args.limit_tampered or args.limit_authentic),
        },
        "pixel_metrics_at_fixed_threshold": {
            "threshold": pixel_threshold,
            "micro_iou": round(micro_iou, 6),
            "micro_precision": round(micro_precision, 6),
            "micro_recall": round(micro_recall, 6),
            "micro_f1": round(micro_f1, 6),
            "mean_image_iou_tampered": round(float(np.mean(ious)), 6) if ious else None,
            "mean_image_recall_tampered": round(float(np.mean(recalls)), 6) if recalls else None,
            "iou_standard_error": round(float(np.std(ious)) / sqrt_n, 6) if ious else None,
            "authentic_pixel_false_positive_rate": round(
                _ratio(authentic_positive, authentic_pixels), 6
            ),
            "background_pixel_false_positive_rate": round(_ratio(fp, negative_pixels), 6),
            "true_positive_pixels": int(tp),
            "false_positive_pixels": int(fp),
            "false_negative_pixels": int(fn),
            "note": (
                "Raw thresholded masks only; the application's post-filtered "
                "highlights are a separate display stage."
            ),
        },
        "image_level_metrics_at_fixed_threshold": {
            "threshold": image_threshold,
            "uncertainty_margin": uncertainty,
            "auc_threshold_free": None if auc is None else round(auc, 6),
            "tampered_recall": round(image_recall, 6),
            "authentic_false_positive_rate": round(image_fpr, 6),
            "uncertain_count": uncertain,
            "uncertain_fraction": round(_ratio(uncertain, len(selected)), 6),
            "mean_score_tampered": (
                round(float(scores_array[tampered_mask].mean()), 6) if tampered_mask.any() else None
            ),
            "mean_score_authentic": (
                round(float(scores_array[authentic_mask].mean()), 6)
                if authentic_mask.any()
                else None
            ),
            "score_definition": "maximum pixel probability per image",
        },
        "per_image_csv": str(Path(args.per_image_csv).resolve()) if args.per_image_csv else None,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "warnings": [],
        "notes": [
            "Thresholds were fixed from historical validation-only selection; no test-set tuning occurred.",
            "The group-aware split confines every duplicate component to one split; the historical split was not modified.",
            "Pixel metrics use group-truth masks resized to the evaluation size with nearest-neighbour interpolation.",
        ],
    }

    if summary["samples"]["sampled"]:
        summary["warnings"].append(
            "Only a subset of the split was evaluated; these metrics are for pipeline "
            "verification and must not be quoted as full-split results."
        )

    if args.per_image_csv:
        per_image_path = Path(args.per_image_csv)
        per_image_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_image).to_csv(per_image_path, index=False)

    output_path = Path(args.output) if args.output else (
        settings.evaluation_dir / f"module1_groupaware_{args.split}_metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["output"] = str(output_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "ckpt" / "mvssnet_casia.pt")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--data-root", type=Path, default=settings.data_root)
    parser.add_argument("--split", default="test", choices=["test", "validation", "train"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--per-image-csv", type=Path, default=None)
    parser.add_argument("--save-probabilities", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--pixel-threshold", type=float, default=settings.manipulation_threshold)
    parser.add_argument("--image-threshold", type=float, default=settings.image_level_threshold)
    parser.add_argument(
        "--image-uncertainty-margin", type=float, default=settings.image_uncertainty_margin
    )
    parser.add_argument("--limit-tampered", type=int, default=0)
    parser.add_argument("--limit-authentic", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.smoke:
        args.limit_tampered = max(1, min(args.limit_tampered or 2, 2))
        args.limit_authentic = max(1, min(args.limit_authentic or 2, 2))
        args.num_workers = 0
        if args.output is None and args.split == "test":
            args.output = settings.evaluation_dir / "module1_groupaware_smoke_metrics.json"

    try:
        summary = run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 2

    pixel = summary["pixel_metrics_at_fixed_threshold"]
    image = summary["image_level_metrics_at_fixed_threshold"]
    print("\n=== MODULE 1 GROUP-AWARE EVALUATION ===")
    print(f"Mode: {summary['mode']} | split: {summary['split']} | device: {summary['device']}")
    print(f"Samples: {json.dumps(summary['samples'])}")
    print(
        f"Pixel @ {pixel['threshold']}: micro IoU {pixel['micro_iou']} | "
        f"mean image IoU {pixel['mean_image_iou_tampered']} | F1 {pixel['micro_f1']}"
    )
    print(
        f"Image @ {image['threshold']}: AUC {image['auc_threshold_free']} | "
        f"recall {image['tampered_recall']} | FPR {image['authentic_false_positive_rate']}"
    )
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}")
    print(f"JSON: {summary['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
