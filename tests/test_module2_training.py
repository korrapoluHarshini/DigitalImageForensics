"""Tests for the Module 2 training/export harness.

These tests never train on real data.  A tiny synthetic dataset exercises the
split logic, the checkpoint/export path and the application's TorchScript
loader contract so that a future run on a legitimate AI-image dataset does not
fail for structural reasons.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from modules.ai_generated_detection.model import (
    integration_status,
    load_backend,
    load_manifest,
)
from training.train_ai_detector import (
    CLASS_NAMES,
    ImageFolderDataset,
    Sample,
    binary_metrics,
    build_split,
    collect_images,
    generator_name,
    train,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_image(path: Path, value: int, *, size: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(value)
    grid = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    Image.fromarray(grid).save(path)


@pytest.fixture()
def smoke_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    for index in range(8):
        _write_image(root / "real" / f"real_{index:02d}.png", index)
    for generator, offset in (("gen_a", 40), ("gen_b", 80)):
        for index in range(6):
            _write_image(root / "ai" / generator / f"{generator}_{index:02d}.png", offset + index)
    return root


# ---------------------------------------------------------------------------
# Split and metrics logic
# ---------------------------------------------------------------------------


def test_generator_name_uses_first_subfolder(tmp_path):
    root = tmp_path / "ai"
    assert generator_name(root / "gen_a" / "img.png", root) == "gen_a"
    assert generator_name(root / "img.png", root) == "root"


def test_build_split_is_deterministic_and_keeps_holdouts_out_of_training(smoke_dataset):
    real_paths = collect_images(smoke_dataset / "real")
    ai_paths = collect_images(smoke_dataset / "ai")
    kwargs = dict(
        holdout_generators={"gen_b"},
        val_fraction=0.2,
        test_fraction=0.2,
        seed=11,
    )
    first = build_split(real_paths, ai_paths, smoke_dataset / "ai", **kwargs)
    second = build_split(real_paths, ai_paths, smoke_dataset / "ai", **kwargs)

    assert [(s.path, s.split) for s in first] == [(s.path, s.split) for s in second]
    assert {s.split for s in first} == {"train", "validation", "test"}
    held_out = [s for s in first if s.generator == "gen_b"]
    assert held_out and all(s.split == "test" for s in held_out)
    assert not any(s.generator == "gen_b" and s.split == "train" for s in first)
    assert all(s.generator == "gen_a" or s.label == 0 for s in first if s.split != "test")


def test_build_split_rejects_unknown_and_complete_holdouts(smoke_dataset):
    real_paths = collect_images(smoke_dataset / "real")
    ai_paths = collect_images(smoke_dataset / "ai")
    ai_root = smoke_dataset / "ai"

    with pytest.raises(ValueError, match="not found"):
        build_split(
            real_paths, ai_paths, ai_root,
            holdout_generators={"missing"}, val_fraction=0.2, test_fraction=0.2, seed=1,
        )
    with pytest.raises(ValueError, match="All AI generators"):
        build_split(
            real_paths, ai_paths, ai_root,
            holdout_generators={"gen_a", "gen_b"}, val_fraction=0.2, test_fraction=0.2, seed=1,
        )


def test_binary_metrics_matches_hand_computation():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.2])
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["roc_auc"] == 0.5


def test_dataset_returns_normalised_tensors(smoke_dataset):
    samples = [Sample(smoke_dataset / "real" / "real_00.png", 0, "real")]
    dataset = ImageFolderDataset(samples, input_size=32, train=False)
    tensor, label = dataset[0]
    assert tuple(tensor.shape) == (3, 32, 32)
    assert label == 0
    assert float(tensor.min()) >= -2.5 and float(tensor.max()) <= 2.5


# ---------------------------------------------------------------------------
# End-to-end training and export
# ---------------------------------------------------------------------------


def test_tiny_training_exports_manifest_and_loadable_torchscript(smoke_dataset, tmp_path):
    real_paths = collect_images(smoke_dataset / "real")
    ai_paths = collect_images(smoke_dataset / "ai")
    samples = build_split(
        real_paths,
        ai_paths,
        smoke_dataset / "ai",
        holdout_generators={"gen_b"},
        val_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )
    config = TrainingConfig(
        backbone="small_cnn",
        input_size=32,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=7,
        num_workers=0,
        patience=1,
        pretrained=False,
        device="cpu",
    )
    output_dir = tmp_path / "run"
    summary = train(
        config,
        samples,
        output_dir,
        dataset="synthetic smoke",
        license_text="synthetic test data",
        provenance="generated in test",
        resume=None,
    )

    for name in ("best.pt", "last.pt", "detector.ts", "manifest.json", "metrics.json",
                 "split.csv", "training_log.csv"):
        assert (output_dir / name).exists(), name

    manifest = load_manifest(output_dir / "manifest.json")
    assert manifest["kind"] == "torchscript"
    assert manifest["class_names"] == CLASS_NAMES
    assert manifest["input_size"] == 32
    assert manifest["dataset_license"] == "synthetic test data"

    status = integration_status(output_dir / "manifest.json")
    assert status.available is True
    assert status.kind == "torchscript"

    backend = load_backend(output_dir / "manifest.json")
    probabilities = backend.predict(
        np.asarray(Image.open(smoke_dataset / "real" / "real_00.png").convert("RGB"))
    )
    assert probabilities.shape == (2,)
    assert abs(float(probabilities.sum()) - 1.0) < 1e-4

    # Metrics must be per-generator so cross-generator behaviour is visible.
    assert set(summary["per_generator_test"]) == {"gen_a", "gen_b"}
    assert summary["samples"]["generators_in_test"] == ["gen_a", "gen_b"]
    assert summary["best_epoch"] >= 1
    stored = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert stored["test"]["images"] == summary["test"]["images"]


def test_training_refuses_to_overwrite_existing_experiment(smoke_dataset, tmp_path):
    from training.train_ai_detector import main
    import sys

    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("do not delete", encoding="utf-8")

    argv = [
        "train_ai_detector.py",
        "--real-dir", str(smoke_dataset / "real"),
        "--ai-dir", str(smoke_dataset / "ai"),
        "--output-dir", str(output_dir),
        "--backbone", "small_cnn",
        "--input-size", "32",
        "--epochs", "1",
    ]
    original = sys.argv
    sys.argv = argv
    try:
        exit_code = main()
    finally:
        sys.argv = original

    assert exit_code == 2
    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "do not delete"
