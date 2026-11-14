"""Retrain MVSS-Net on the leakage-free CASIA2.0 group-aware split.

This script is designed to run on a CUDA machine (for example Google Colab) and
to be resumed after a disconnect.  It never writes to the original checkpoint
(``ckpt/mvssnet_casia.pt``) and never modifies the historical CASIA splits or the
group-aware split artefact.

Inputs
------
``results/manifests/casia2_manifest.csv``
    Full CASIA2.0 manifest with image/mask paths.
``results/manifests/casia2_groupaware_split.csv``
    Experimental split produced by ``evaluation/analyze_casia_leakage.py``.
    Every duplicate component is confined to one split.

Outputs (inside ``--output-dir``, never the original checkpoint)
---------------------------------------------------------------
``last.pt``               resumable training state (model, optimizer, scaler)
``best.pt``               best-validation training state
``mvssnet_groupaware.pt``  plain ``state_dict`` for the Streamlit application
``training_log.csv``      per-epoch metrics
``run.json``              configuration, sample counts and final metrics
``split_used.csv``        exact rows used, for reproducibility

Example (Google Colab, GPU):::

    python training/train_mvss_groupaware.py \
        --manifest "/content/drive/MyDrive/mvss/results/manifests/casia2_manifest.csv" \
        --split-csv "/content/drive/MyDrive/mvss/results/manifests/casia2_groupaware_split.csv" \
        --data-root "/content/drive/MyDrive/mvss" \
        --output-dir "/content/drive/MyDrive/mvss/experiments/module1/groupaware_v1" \
        --device cuda --epochs 40 --batch-size 8

Resume after a disconnect with ``--resume .../last.pt``.  ``--smoke`` runs a
tiny end-to-end pass on a handful of images to verify the pipeline before a long
run.

Notes
-----
* The model is initialised from ``ckpt/mvssnet_casia.pt`` by default (transfer
  learning), but that file is only ever read.
* Loss weights mirror the repository's existing MVSS-Net training script:
  ``0.16 * seg + 0.80 * edge + 0.04 * cls``.
* Validation selection uses mean image IoU on validation **tampered** images at
  a fixed 0.5 operating threshold.  The test split is never used for model
  selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from casia_dataset import EdgeGenerator  # noqa: E402 - reuse exact edge parity
from models.mvssnet import get_mvss  # noqa: E402
from utils.config import settings  # noqa: E402


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
LOSS_WEIGHTS = {"seg": 0.16, "edge": 0.80, "cls": 0.04}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class Record:
    sample_id: str
    class_name: str
    image_path: Path
    mask_path: Path | None
    split: str


def _resolve(data_root: Path, relative: str) -> Path:
    path = Path(str(relative).replace("\\", "/"))
    return path if path.is_absolute() else (data_root / path)


def build_records(
    manifest_path: Path, split_csv: Path, data_root: Path
) -> tuple[list[Record], list[dict]]:
    manifest = pd.read_csv(manifest_path)
    split_frame = pd.read_csv(split_csv)
    required_manifest = {"sample_id", "class_name", "image_path", "mask_path"}
    required_split = {"sample_id", "experimental_split"}
    missing_manifest = required_manifest - set(manifest.columns)
    missing_split = required_split - set(split_frame.columns)
    if missing_manifest or missing_split:
        raise ValueError(
            f"Missing columns. manifest={sorted(missing_manifest)} "
            f"split={sorted(missing_split)}"
        )

    split_map = dict(zip(split_frame["sample_id"], split_frame["experimental_split"]))
    records: list[Record] = []
    skipped: list[dict] = []
    for row in manifest.itertuples(index=False):
        split = split_map.get(row.sample_id)
        if split is None:
            continue
        mask_path = None
        if isinstance(row.mask_path, str) and row.mask_path.strip():
            candidate = _resolve(data_root, row.mask_path)
            mask_path = candidate if candidate.exists() else None
        image_path = _resolve(data_root, row.image_path)
        if row.class_name == "Tampered" and mask_path is None:
            # 17 tampered CASIA2.0 rows have no usable mask.  They cannot
            # supervise pixel localisation, so they are excluded and counted
            # rather than silently treated as authentic.
            skipped.append(
                {
                    "sample_id": row.sample_id,
                    "split": split,
                    "reason": "tampered row without a readable ground-truth mask",
                    "mask_path": str(row.mask_path),
                }
            )
            continue
        records.append(
            Record(
                sample_id=row.sample_id,
                class_name=row.class_name,
                image_path=image_path,
                mask_path=mask_path,
                split=split,
            )
        )

    missing_images = [record.image_path for record in records if not record.image_path.exists()]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} image path(s) are missing under {data_root}. "
            f"First: {missing_images[0]}"
        )
    if skipped:
        print(
            f"Skipping {len(skipped)} tampered row(s) without a readable mask "
            f"(first: {skipped[0]['sample_id']})."
        )
    return records, skipped


class GroupAwareDataset(Dataset):
    def __init__(
        self,
        records: list[Record],
        *,
        image_size: int,
        augment: bool,
        limit: int = 0,
    ) -> None:
        self.records = records[:limit] if limit > 0 else records
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.edge_generator = EdgeGenerator(kernel_size=15)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        # Read in BGR order and keep it: the application and the official
        # evaluation pipeline normalise BGR channels with ImageNet statistics
        # (``modules.image_manipulation.preprocessing.preprocess_bgr``).  A
        # retrained checkpoint must see exactly the same channel order.
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Image could not be decoded: {record.image_path}")
        if record.class_name == "Tampered":
            mask = cv2.imread(str(record.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Mask could not be decoded: {record.mask_path}")
            label = 1.0
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            label = 0.0

        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(
            mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
        )
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if random.random() < 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()
            if random.random() < 0.5:
                k = random.randint(1, 3)
                image = np.rot90(image, k=k).copy()
                mask = np.rot90(mask, k=k).copy()

        image = image.astype(np.float32) / 255.0
        image = (image - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())

        mask_tensor = torch.from_numpy(mask.copy()).unsqueeze(0)
        edge = self.edge_generator(mask_tensor)
        edge = F.interpolate(
            edge,
            size=(self.image_size // 4, self.image_size // 4),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return image_tensor, mask_tensor, edge, torch.tensor([label], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------


def dice_loss(probabilities: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probabilities = probabilities.reshape(-1)
    target = target.reshape(-1)
    intersection = (probabilities * target).sum()
    return 1.0 - (2.0 * intersection + smooth) / (
        probabilities.sum() + target.sum() + smooth
    )


def compute_loss(edge_logits, mask_logits, masks, edges, labels):
    seg_bce = F.binary_cross_entropy_with_logits(mask_logits, masks)
    seg_dice = dice_loss(torch.sigmoid(mask_logits), masks)
    seg_loss = 0.5 * (seg_bce + seg_dice)

    edge_bce = F.binary_cross_entropy_with_logits(edge_logits, edges)
    edge_dice = dice_loss(torch.sigmoid(edge_logits), edges)
    edge_loss = 0.5 * (edge_bce + edge_dice)

    image_logits = F.adaptive_max_pool2d(mask_logits, output_size=1).flatten(1)
    cls_loss = F.binary_cross_entropy_with_logits(image_logits, labels)

    total = (
        LOSS_WEIGHTS["seg"] * seg_loss
        + LOSS_WEIGHTS["edge"] * edge_loss
        + LOSS_WEIGHTS["cls"] * cls_loss
    )
    return total, {
        "seg": float(seg_loss.detach()),
        "edge": float(edge_loss.detach()),
        "cls": float(cls_loss.detach()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    total = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    ious: list[float] = []
    image_scores: list[float] = []
    image_labels: list[float] = []
    loss_sum = 0.0

    for images, masks, edges, labels in tqdm(loader, desc="validation", leave=False):
        images, masks, edges, labels = (
            images.to(device),
            masks.to(device),
            edges.to(device),
            labels.to(device),
        )
        edge_logits, mask_logits = model(images)
        loss, _ = compute_loss(edge_logits, mask_logits, masks, edges, labels)
        loss_sum += float(loss) * images.size(0)

        probabilities = torch.sigmoid(mask_logits)
        predicted = (probabilities >= threshold).float()
        total["tp"] += float((predicted * masks).sum())
        total["fp"] += float((predicted * (1 - masks)).sum())
        total["fn"] += float(((1 - predicted) * masks).sum())

        for index in range(images.size(0)):
            mask_sum = float(masks[index].sum())
            if mask_sum <= 0:
                continue
            intersection = float((predicted[index] * masks[index]).sum())
            union = float(predicted[index].sum()) + mask_sum - intersection
            ious.append(intersection / max(1e-8, union))

        scores = F.adaptive_max_pool2d(probabilities, output_size=1).flatten(1)[:, 0]
        image_scores.extend(scores.detach().cpu().tolist())
        image_labels.extend(labels[:, 0].detach().cpu().tolist())

    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    micro_iou = tp / max(1e-8, tp + fp + fn)
    micro_precision = tp / max(1e-8, tp + fp)
    micro_recall = tp / max(1e-8, tp + fn)
    micro_f1 = (
        2 * micro_precision * micro_recall / max(1e-8, micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    scores_array = np.asarray(image_scores, dtype=float)
    labels_array = np.asarray(image_labels, dtype=float)
    auc = None
    positives = labels_array == 1
    negatives = labels_array == 0
    if positives.any() and negatives.any():
        # Average ranks for tied scores (Mann-Whitney U with ties).
        order = np.argsort(scores_array, kind="mergesort")
        sorted_scores = scores_array[order]
        ranks = np.empty(len(scores_array), dtype=np.float64)
        position = 0
        while position < len(sorted_scores):
            end = position
            while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[position]:
                end += 1
            ranks[order[position : end + 1]] = (position + end + 2) / 2.0
            position = end + 1
        auc = float(
            (ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2.0)
            / (positives.sum() * negatives.sum())
        )

    return {
        "loss": round(loss_sum / max(1, len(loader.dataset)), 6),
        "images": int(len(loader.dataset)),
        "mean_image_iou": round(float(np.mean(ious)) if ious else 0.0, 6),
        "micro_iou": round(micro_iou, 6),
        "micro_precision": round(micro_precision, 6),
        "micro_recall": round(micro_recall, 6),
        "micro_f1": round(micro_f1, 6),
        "image_level_auc": None if auc is None else round(auc, 6),
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _jsonable(value):
    """Convert argparse values (notably ``Path``) into JSON-safe objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _new_model(device: torch.device) -> nn.Module:
    model = get_mvss(
        backbone="resnet50",
        # The weights come from --init-checkpoint / --resume, so the ImageNet
        # base download is unnecessary.  This also matches the app's loader and
        # keeps the pipeline reproducible without network access.
        pretrained_base=False,
        nclass=1,
        sobel=True,
        constrain=True,
        n_input=3,
    )
    return model.to(device)


def train(args, model_factory=None) -> dict:
    device = _resolve_device(args.device)
    _seed_everything(args.seed)
    data_root = Path(args.data_root).resolve()

    records, skipped = build_records(Path(args.manifest), Path(args.split_csv), data_root)
    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "validation"]
    if not train_records or not val_records:
        raise ValueError(
            f"Split '{args.split_csv}' has {len(train_records)} train and "
            f"{len(val_records)} validation rows; both are required."
        )

    if args.smoke:
        # A balanced handful of real files is enough to verify shapes and losses.
        def pick(rows, label, count):
            selected = [row for row in rows if (row.class_name == "Tampered") == label]
            return selected[:count]

        train_records = pick(train_records, True, 2) + pick(train_records, False, 2)
        val_records = pick(val_records, True, 2) + pick(val_records, False, 1)
        random.Random(args.seed).shuffle(train_records)

    train_dataset = GroupAwareDataset(
        train_records, image_size=args.image_size, augment=True, limit=args.limit_train
    )
    val_dataset = GroupAwareDataset(
        val_records, image_size=args.image_size, augment=False, limit=args.limit_val
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite and not args.resume:
        raise SystemExit(
            f"Refusing to overwrite non-empty output directory: {output_dir}\n"
            "Choose a new versioned directory, pass --overwrite, or resume with --resume."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = (model_factory or _new_model)(device)
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if init_path.resolve() == (PROJECT_ROOT / "ckpt" / "mvssnet_casia.pt").resolve():
            print(f"Initialising from the preserved checkpoint (read-only): {init_path}")
        state = torch.load(init_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state, strict=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.learning_rate * 0.05
    )

    start_epoch = 0
    best_iou = -1.0
    history: list[dict] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        if checkpoint.get("scheduler_state") is not None:
            # Restore the cosine schedule so a resumed run does not restart at
            # the initial learning rate.  Older checkpoints lack this key, in
            # which case the schedule restarts and that is reported.
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            # ``load_state_dict`` also restores the saved T_max.  Re-apply the
            # horizon requested for this run so that extending a finished
            # schedule continues the cosine curve instead of jumping back to
            # the base learning rate.
            scheduler.T_max = max(1, args.epochs)
        else:
            print(
                "Resume checkpoint has no scheduler_state; the cosine schedule "
                "will restart at the initial learning rate."
            )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_iou = float(checkpoint.get("best_iou", -1.0))
        history = list(checkpoint.get("history", []))
        print(f"Resumed at epoch {start_epoch + 1} from {args.resume}")

    log_path = output_dir / "training_log.csv"
    fieldnames = [
        "epoch", "train_loss", "train_seg", "train_edge", "train_cls",
        "val_loss", "val_micro_iou", "val_mean_image_iou", "val_micro_f1",
        "val_image_level_auc", "learning_rate", "elapsed_seconds",
    ]
    append_log = bool(args.resume) and log_path.exists()
    log_handle = log_path.open("a" if append_log else "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(log_handle, fieldnames=fieldnames)
    if not append_log:
        writer.writeheader()

    started = time.perf_counter()
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            running_loss = 0.0
            running_parts = {"seg": 0.0, "edge": 0.0, "cls": 0.0}
            progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
            for images, masks, edges, labels in progress:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                edges = edges.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    edge_logits, mask_logits = model(images)
                    loss, parts = compute_loss(edge_logits, mask_logits, masks, edges, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

                running_loss += float(loss.detach()) * images.size(0)
                for key in running_parts:
                    running_parts[key] += parts[key] * images.size(0)
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

            scheduler.step()
            train_count = max(1, len(train_dataset))
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": round(running_loss / train_count, 6),
                "train_seg": round(running_parts["seg"] / train_count, 6),
                "train_edge": round(running_parts["edge"] / train_count, 6),
                "train_cls": round(running_parts["cls"] / train_count, 6),
            }
            val_metrics = evaluate(model, val_loader, device, threshold=args.val_threshold)
            epoch_record.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_micro_iou": val_metrics["micro_iou"],
                    "val_mean_image_iou": val_metrics["mean_image_iou"],
                    "val_micro_f1": val_metrics["micro_f1"],
                    "val_image_level_auc": val_metrics["image_level_auc"],
                    "learning_rate": round(optimizer.param_groups[0]["lr"], 8),
                    "elapsed_seconds": round(time.perf_counter() - started, 1),
                }
            )
            history.append({**epoch_record, "val": val_metrics})
            writer.writerow(epoch_record)
            log_handle.flush()

            score = val_metrics["mean_image_iou"]
            if score > best_iou:
                best_iou = score
                best_path = output_dir / "best.pt"
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_iou": best_iou,
                        "history": history,
                        "config": _jsonable(vars(args)),
                    },
                    best_path,
                )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_iou": best_iou,
                    "history": history,
                    "config": _jsonable(vars(args)),
                },
                output_dir / "last.pt",
            )
            print(
                f"epoch {epoch + 1} | train {epoch_record['train_loss']:.4f} "
                f"| val micro IoU {val_metrics['micro_iou']:.4f} "
                f"| val mean image IoU {val_metrics['mean_image_iou']:.4f} "
                f"| val AUC {val_metrics['image_level_auc']}"
            )
    finally:
        log_handle.close()

    best_path = output_dir / "best.pt"
    checkpoint = torch.load(
        best_path if best_path.exists() else output_dir / "last.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    # Plain state_dict for the Streamlit application.  Never written to ckpt/.
    torch.save(model.state_dict(), output_dir / "mvssnet_groupaware.pt")

    final_validation = evaluate(model, val_loader, device, threshold=args.val_threshold)

    split_used = output_dir / "split_used.csv"
    pd.DataFrame(
        [
            {
                "sample_id": record.sample_id,
                "class_name": record.class_name,
                "split": record.split,
                "image_path": str(record.image_path),
                "mask_path": str(record.mask_path) if record.mask_path else "",
            }
            for record in records
            if record.split in {"train", "validation"}
        ]
    ).to_csv(split_used, index=False)

    summary = {
        "artifact": "module1_groupaware_training_run.json",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "smoke" if args.smoke else "full",
        "device": str(device),
        "config": _jsonable(vars(args)),
        "loss_weights": LOSS_WEIGHTS,
        "samples": {
            "train": len(train_dataset),
            "validation": len(val_dataset),
            "train_tampered": sum(r.class_name == "Tampered" for r in train_records),
            "train_authentic": sum(r.class_name == "Authentic" for r in train_records),
            "validation_tampered": sum(r.class_name == "Tampered" for r in val_records),
            "validation_authentic": sum(r.class_name == "Authentic" for r in val_records),
            "excluded_tampered_without_mask": len(skipped),
        },
        "excluded_rows": skipped,
        "best_validation_mean_image_iou": round(best_iou, 6),
        "final_validation": final_validation,
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "outputs": {
            "last": str(output_dir / "last.pt"),
            "best": str(output_dir / "best.pt"),
            "app_checkpoint": str(output_dir / "mvssnet_groupaware.pt"),
            "split_used": str(split_used),
        },
        "notes": [
            "Training used the leakage-free group-aware split; the historical split was not modified.",
            "The original checkpoint was only read as an initialisation source.",
            "Validation mean image IoU at a fixed 0.5 threshold selected the best epoch.",
            "Test-split metrics require a separate evaluation run and are not produced here.",
        ],
    }
    (output_dir / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "results" / "manifests" / "casia2_manifest.csv")
    parser.add_argument("--split-csv", type=Path,
                        default=PROJECT_ROOT / "results" / "manifests" / "casia2_groupaware_split.csv")
    parser.add_argument("--data-root", type=Path, default=settings.data_root)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path,
                        default=PROJECT_ROOT / "ckpt" / "mvssnet_casia.pt")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny balanced pass to verify the pipeline (few images, 1 epoch).")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.image_size = min(args.image_size, 256)
        args.num_workers = 0

    try:
        summary = train(args)
    except (FileNotFoundError, ValueError, RuntimeError, SystemExit) as exc:
        print(f"Training error: {exc}", file=sys.stderr)
        return 2

    print("\n=== MODULE 1 GROUP-AWARE TRAINING COMPLETE ===")
    print(f"Mode: {summary['mode']} | device: {summary['device']}")
    print(f"Samples: {json.dumps(summary['samples'])}")
    print(f"Best validation mean image IoU: {summary['best_validation_mean_image_iou']}")
    print(f"Final validation: {json.dumps(summary['final_validation'])}")
    print(f"App checkpoint: {summary['outputs']['app_checkpoint']}")
    print("Test-split metrics are NOT produced here; run "
          "evaluation/evaluate_groupaware_checkpoint.py separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
