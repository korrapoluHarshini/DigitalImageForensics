"""Reproducible training and export harness for Module 2 (AI-generated images).

No detector weights ship with this repository, and this script does not invent
any: it trains only on folders the user explicitly provides.  The harness is
dataset-agnostic so that GenImage (or another legitimately obtained benchmark)
can be used once it is available.  It exports a TorchScript model plus a
manifest that the Streamlit application already knows how to load.

Expected input layout::

    <real-dir>/            real photographs (any nested layout)
    <ai-dir>/
        <generator_a>/     images from one generator
        <generator_b>/     images from another generator

Held-out generators
-------------------
Use ``--holdout-generators gen_a,gen_b`` to keep entire generators out of
training.  Cross-generator metrics are always written per generator, so a run
without held-out generators still shows whether the model generalises.

Example (small CPU smoke run)::

    python training/train_ai_detector.py \
        --real-dir D:\\ImageForensics_Datasets\\module2\\real \
        --ai-dir D:\\ImageForensics_Datasets\\module2\\ai \
        --output-dir experiments\\module2\\efficientnet_b0_v1 \
        --backbone efficientnet_b0 --epochs 20

The script writes ``best.pt``, ``last.pt``, ``training_log.csv``,
``metrics.json``, ``split.csv``, ``detector.ts`` and ``manifest.json``.  It
never modifies an existing experiment directory unless ``--overwrite`` is
given.
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

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import settings

try:  # sklearn is used for AUC; a manual fallback keeps the harness usable.
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - optional dependency
    roc_auc_score = None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ["real", "ai_generated"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def collect_images(directory: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def generator_name(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return "unknown"
    return relative.parts[0] if len(relative.parts) > 1 else "root"


@dataclass
class Sample:
    path: Path
    label: int
    generator: str
    split: str = "unassigned"


def build_split(
    real_paths: list[Path],
    ai_paths: list[Path],
    ai_root: Path,
    *,
    holdout_generators: set[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> list[Sample]:
    rng = random.Random(seed)
    samples = [Sample(path, 0, "real") for path in real_paths]
    samples += [Sample(path, 1, generator_name(path, ai_root)) for path in ai_paths]

    held_out = [sample for sample in samples if sample.label == 1 and sample.generator in holdout_generators]
    held_out_ids = {id(sample) for sample in held_out}
    generator_set = {sample.generator for sample in samples if sample.label == 1}

    if holdout_generators:
        if not holdout_generators <= generator_set:
            unknown = sorted(holdout_generators - generator_set)
            raise ValueError(
                f"Holdout generators not found under {ai_root}: {unknown}. "
                f"Available: {sorted(generator_set)}"
            )
        if len(generator_set - holdout_generators) == 0:
            raise ValueError(
                "All AI generators are held out; at least one generator must remain for training."
            )

    remaining = [sample for sample in samples if id(sample) not in held_out_ids]
    rng.shuffle(remaining)
    test_count = int(round(len(remaining) * test_fraction))
    val_count = int(round(len(remaining) * val_fraction))
    for index, sample in enumerate(remaining):
        if index < val_count:
            sample.split = "validation"
        elif index < val_count + test_count:
            sample.split = "test"
        else:
            sample.split = "train"

    for sample in held_out:
        sample.split = "test"

    for split in ("train", "validation", "test"):
        if not any(sample.split == split for sample in samples):
            raise ValueError(f"Split '{split}' is empty; provide more images or change the fractions.")
    if not any(sample.split == "train" and sample.label == 1 for sample in samples):
        raise ValueError("No AI-generated training images remain after the split.")
    if not any(sample.split == "train" and sample.label == 0 for sample in samples):
        raise ValueError("No real training images remain after the split.")
    return samples


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        *,
        input_size: int,
        train: bool,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.input_size = input_size
        self.train = train
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = self._load(sample.path)

        if self.train:
            # Mild geometric augmentation only: strong colour/photometric
            # changes can erase the generator traces this module must detect.
            if self.rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            width, height = image.size
            crop_scale = self.rng.uniform(0.80, 1.0)
            crop_width = max(8, int(width * crop_scale))
            crop_height = max(8, int(height * crop_scale))
            left = self.rng.randint(0, max(0, width - crop_width))
            top = self.rng.randint(0, max(0, height - crop_height))
            image = image.crop((left, top, left + crop_width, top + crop_height))

        image = image.resize((self.input_size, self.input_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).float()
        return tensor, sample.label


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SmallCNN(nn.Module):
    """Compact CPU-friendly baseline (no external weights)."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(torch.flatten(features, 1))


def build_model(backbone: str, *, pretrained: bool = False, num_classes: int = 2) -> nn.Module:
    backbone = backbone.lower()
    if backbone == "small_cnn":
        return SmallCNN(num_classes=num_classes)

    try:
        import torchvision.models as models
    except Exception as exc:  # pragma: no cover - torchvision is a declared dependency
        raise RuntimeError(f"torchvision is required for backbone '{backbone}': {exc}") from exc

    weights = "DEFAULT" if pretrained else None
    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif backbone == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Choose efficientnet_b0, resnet18, "
            "resnet50, convnext_tiny or small_cnn."
        )
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def binary_metrics(labels: list[int], probabilities: list[float], threshold: float = 0.5) -> dict:
    labels_array = np.asarray(labels, dtype=int)
    probabilities_array = np.asarray(probabilities, dtype=float)
    predicted = (probabilities_array >= threshold).astype(int)

    tp = int(np.sum((predicted == 1) & (labels_array == 1)))
    tn = int(np.sum((predicted == 0) & (labels_array == 0)))
    fp = int(np.sum((predicted == 1) & (labels_array == 0)))
    fn = int(np.sum((predicted == 0) & (labels_array == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    accuracy = (tp + tn) / max(1, len(labels_array))

    auc = None
    if len(set(labels_array.tolist())) == 2:
        if roc_auc_score is not None:
            auc = float(roc_auc_score(labels_array, probabilities_array))
        else:  # pragma: no cover - fallback when sklearn is absent
            order = np.argsort(probabilities_array)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(order) + 1)
            positives = labels_array.sum()
            negatives = len(labels_array) - positives
            auc = float(
                (ranks[labels_array == 1].sum() - positives * (positives + 1) / 2)
                / (positives * negatives)
            )

    return {
        "threshold": float(threshold),
        "images": int(len(labels_array)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "false_positive_rate": round(fpr, 6),
        "false_negative_rate": round(fnr, 6),
        "roc_auc": None if auc is None else round(auc, 6),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    backbone: str
    input_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    num_workers: int
    patience: int
    pretrained: bool
    device: str
    label_smoothing: float = 0.05
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "backbone": self.backbone,
            "input_size": self.input_size,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "num_workers": self.num_workers,
            "patience": self.patience,
            "pretrained": self.pretrained,
            "device": self.device,
            "label_smoothing": self.label_smoothing,
            "mean": list(MEAN),
            "std": list(STD),
            "class_names": CLASS_NAMES,
            **self.extra,
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[dict, float, list[int], list[float]]:
    model.eval()
    total_loss = 0.0
    labels: list[int] = []
    probabilities: list[float] = []
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            total_loss += float(criterion(logits, targets)) * images.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            labels.extend(targets.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
    metrics = binary_metrics(labels, probabilities)
    metrics["loss"] = round(total_loss / max(1, len(loader.dataset)), 6)
    return metrics, metrics["loss"], labels, probabilities


def export_torchscript(model: nn.Module, path: Path, input_size: int, device: torch.device) -> None:
    model = model.to(device).eval()
    example = torch.randn(1, 3, input_size, input_size, device=device)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=False)
    torch.jit.save(traced, str(path))


def write_manifest(
    output_dir: Path,
    *,
    backbone: str,
    input_size: int,
    dataset: str,
    license_text: str,
    provenance: str,
    metrics: dict,
) -> Path:
    manifest = {
        "kind": "torchscript",
        "path": "detector.ts",
        "input_size": int(input_size),
        "mean": list(MEAN),
        "std": list(STD),
        "class_names": CLASS_NAMES,
        "ai_class_index": 1,
        "backbone": backbone,
        "dataset": dataset,
        "dataset_license": license_text,
        "provenance": provenance,
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validation_metrics": metrics,
        "note": (
            "Trained detector exported by training/train_ai_detector.py. Metrics refer only "
            "to the supplied split; cross-generator generalisation must be checked on held-out "
            "generators before claiming robustness."
        ),
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def train(
    config: TrainingConfig,
    samples: list[Sample],
    output_dir: Path,
    *,
    dataset: str,
    license_text: str,
    provenance: str,
    resume: Path | None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(config.seed)
    device = _resolve_device(config.device)

    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split == "validation"]
    test_samples = [sample for sample in samples if sample.split == "test"]

    train_dataset = ImageFolderDataset(
        train_samples, input_size=config.input_size, train=True, seed=config.seed
    )
    val_dataset = ImageFolderDataset(
        val_samples, input_size=config.input_size, train=False, seed=config.seed
    )
    test_dataset = ImageFolderDataset(
        test_samples, input_size=config.input_size, train=False, seed=config.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config.backbone, pretrained=config.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs)
    )

    start_epoch = 0
    best_auc = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_auc = float(checkpoint.get("best_auc", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        history = list(checkpoint.get("history", []))
        print(f"Resumed from {resume} at epoch {start_epoch + 1}")

    log_path = output_dir / "training_log.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_roc_auc",
        "val_loss",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_false_positive_rate",
        "val_false_negative_rate",
        "val_roc_auc",
        "learning_rate",
    ]
    write_header = not log_path.exists() or resume is None
    log_handle = log_path.open("a" if resume is not None else "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(log_handle, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    started = time.perf_counter()
    try:
        for epoch in range(start_epoch, config.epochs):
            model.train()
            total_loss = 0.0
            labels: list[int] = []
            probabilities: list[float] = []
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * images.size(0)
                labels.extend(targets.cpu().tolist())
                probabilities.extend(torch.softmax(logits.detach(), dim=1)[:, 1].cpu().tolist())

            train_metrics = binary_metrics(labels, probabilities)
            val_metrics, _val_loss, _labels, _probs = evaluate(
                model, val_loader, device, criterion
            )
            scheduler.step()

            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": round(total_loss / max(1, len(train_dataset)), 6),
                "train_accuracy": train_metrics["accuracy"],
                "train_roc_auc": train_metrics["roc_auc"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_false_positive_rate": val_metrics["false_positive_rate"],
                "val_false_negative_rate": val_metrics["false_negative_rate"],
                "val_roc_auc": val_metrics["roc_auc"],
                "learning_rate": round(optimizer.param_groups[0]["lr"], 8),
            }
            history.append(epoch_record)
            writer.writerow(epoch_record)
            log_handle.flush()

            score = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else val_metrics["accuracy"]
            improved = score > best_auc + 1e-6
            if improved:
                best_auc = float(score)
                best_epoch = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1

            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_auc": best_auc,
                "best_epoch": best_epoch,
                "history": history,
                "config": config.as_dict(),
            }
            torch.save(checkpoint, output_dir / "last.pt")
            if improved:
                torch.save(checkpoint, output_dir / "best.pt")

            print(
                f"epoch {epoch + 1}/{config.epochs} | train loss {epoch_record['train_loss']:.4f} "
                f"| val auc {val_metrics['roc_auc']} | val acc {val_metrics['accuracy']:.4f}"
            )

            if stale_epochs >= config.patience:
                print(f"Early stopping after {stale_epochs} epochs without improvement.")
                break
    finally:
        log_handle.close()

    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    model.eval()

    validation_metrics, _loss, _labels, _probs = evaluate(model, val_loader, device, criterion)
    test_metrics, _test_loss, test_labels, test_probabilities = evaluate(
        model, test_loader, device, criterion
    )

    # Per-generator test breakdown: real images form the negative class for
    # every generator.  This exposes cross-generator generalisation honestly.
    per_generator: dict[str, dict] = {}
    test_real = [index for index, sample in enumerate(test_samples) if sample.label == 0]
    for generator in sorted({sample.generator for sample in test_samples if sample.label == 1}):
        generator_indices = [
            index
            for index, sample in enumerate(test_samples)
            if sample.label == 1 and sample.generator == generator
        ]
        indices = sorted(set(test_real + generator_indices))
        if not indices:
            continue
        labels = [test_labels[index] for index in indices]
        probabilities = [test_probabilities[index] for index in indices]
        per_generator[generator] = binary_metrics(labels, probabilities)
        per_generator[generator]["note"] = (
            "Metrics compare this generator against the real test images. If the generator "
            "was used in training, this is not a held-out-generator result."
        )

    export_torchscript(model, output_dir / "detector.ts", config.input_size, device)
    manifest_path = write_manifest(
        output_dir,
        backbone=config.backbone,
        input_size=config.input_size,
        dataset=dataset,
        license_text=license_text,
        provenance=provenance,
        metrics=validation_metrics,
    )

    summary = {
        "artifact": "module2_training_run.json",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backbone": config.backbone,
        "dataset": dataset,
        "dataset_license": license_text,
        "provenance": provenance,
        "device": str(device),
        "config": config.as_dict(),
        "samples": {
            "total": len(samples),
            "train": len(train_samples),
            "validation": len(val_samples),
            "test": len(test_samples),
            "real_train": sum(sample.label == 0 for sample in train_samples),
            "ai_train": sum(sample.label == 1 for sample in train_samples),
            "generators_in_train": sorted(
                {sample.generator for sample in train_samples if sample.label == 1}
            ),
            "generators_in_test": sorted(
                {sample.generator for sample in test_samples if sample.label == 1}
            ),
        },
        "best_epoch": best_epoch,
        "best_validation_auc": round(best_auc, 6),
        "validation": validation_metrics,
        "test": test_metrics,
        "per_generator_test": per_generator,
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "manifest": str(manifest_path),
        "checkpoint": str(output_dir / "best.pt"),
        "torchscript": str(output_dir / "detector.ts"),
        "limitations": [
            "Metrics describe only the supplied split and generators.",
            "Held-out-generator results require --holdout-generators to exclude entire "
            "generators from training.",
            "A detector trained on one dataset may exploit dataset-specific post-processing; "
            "test on independent real photographs and unseen generators.",
            "No deployment claim should be made without a frozen independent test set.",
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    split_path = output_dir / "split.csv"
    with split_path.open("w", encoding="utf-8", newline="") as handle:
        split_writer = csv.DictWriter(
            handle, fieldnames=["path", "label", "class_name", "generator", "split"]
        )
        split_writer.writeheader()
        for sample in samples:
            split_writer.writerow(
                {
                    "path": str(sample.path),
                    "label": sample.label,
                    "class_name": CLASS_NAMES[sample.label],
                    "generator": sample.generator,
                    "split": sample.split,
                }
            )

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--ai-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backbone",
        default="efficientnet_b0",
        choices=["efficientnet_b0", "resnet18", "resnet50", "convnext_tiny", "small_cnn"],
    )
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet weights (downloads).")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--holdout-generators", default="", help="Comma-separated generator folder names.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dataset-name", default="user-supplied")
    parser.add_argument("--dataset-license", default="not specified")
    parser.add_argument("--provenance", default="")
    args = parser.parse_args()

    real_paths = collect_images(args.real_dir)
    ai_paths = collect_images(args.ai_dir)
    if not real_paths or not ai_paths:
        print(
            "Module 2 training requires both real photographs and AI-generated images.\n"
            f"  real found: {len(real_paths)} ({args.real_dir})\n"
            f"  AI found:   {len(ai_paths)} ({args.ai_dir})\n\n"
            "Provide a legitimately obtained dataset (for example GenImage) before training. "
            "No model is trained on placeholder data.",
            file=sys.stderr,
        )
        return 2

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        print(
            f"Refusing to overwrite non-empty experiment directory: {args.output_dir}\n"
            "Choose a new versioned directory or pass --overwrite.",
            file=sys.stderr,
        )
        return 2

    holdout = {value.strip() for value in args.holdout_generators.split(",") if value.strip()}
    try:
        samples = build_split(
            real_paths,
            ai_paths,
            args.ai_dir,
            holdout_generators=holdout,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
    except ValueError as exc:
        print(f"Split error: {exc}", file=sys.stderr)
        return 2

    config = TrainingConfig(
        backbone=args.backbone,
        input_size=args.input_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        patience=args.patience,
        pretrained=args.pretrained,
        device=args.device,
        extra={
            "holdout_generators": sorted(holdout),
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
        },
    )

    summary = train(
        config,
        samples,
        args.output_dir,
        dataset=args.dataset_name,
        license_text=args.dataset_license,
        provenance=args.provenance,
        resume=args.resume,
    )

    print("\n=== MODULE 2 TRAINING COMPLETE ===")
    print(f"Best validation AUC: {summary['best_validation_auc']}")
    print(f"Test metrics: {json.dumps(summary['test'])}")
    print(f"Per-generator test: {json.dumps(summary['per_generator_test'])}")
    print(f"\nManifest: {summary['manifest']}")
    print(f"Checkpoint: {summary['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
