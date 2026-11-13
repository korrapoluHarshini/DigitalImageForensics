"""Tests for the leakage-free Module 1 retraining harness.

The tests never touch the real CASIA split, the real checkpoint or the group-aware
split artefact.  A tiny synthetic manifest exercises record building, mask
exclusion, dataset tensor parity and a one-epoch training pass with a stub model,
so a real Colab run cannot fail for structural reasons.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_mvss_groupaware import (  # noqa: E402
    GroupAwareDataset,
    _jsonable,
    build_records,
    train,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_image(path: Path, seed: int, *, size: int = 40, mask: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if mask:
        array = np.zeros((size, size), dtype=np.uint8)
        array[size // 4 : size // 2, size // 4 : size // 2] = 255
        Image.fromarray(array, mode="L").save(path)
    else:
        array = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        Image.fromarray(array).save(path)


@pytest.fixture()
def tiny_dataset(tmp_path: Path):
    root = tmp_path / "data"
    rows = []
    split_rows = []
    plans = [
        ("train", "Tampered", 0, True),
        ("train", "Tampered", 1, True),
        ("train", "Authentic", 2, False),
        ("train", "Authentic", 3, False),
        ("validation", "Tampered", 4, True),
        ("validation", "Tampered", 5, True),
        ("validation", "Authentic", 6, False),
        ("test", "Tampered", 7, True),
        ("test", "Authentic", 8, False),
    ]
    for index, (split, class_name, seed, has_mask) in enumerate(plans):
        image_rel = f"images/{class_name.lower()}_{index}.png"
        _write_image(root / image_rel, seed)
        mask_rel = ""
        if has_mask:
            mask_rel = f"masks/{class_name.lower()}_{index}_gt.png"
            _write_image(root / mask_rel, seed, mask=True)
        sample_id = f"sample-{index}"
        rows.append(
            {
                "sample_id": sample_id,
                "class_name": class_name,
                "image_path": image_rel,
                "mask_path": mask_rel,
                "split": split,
            }
        )
        split_rows.append(
            {
                "sample_id": sample_id,
                "experimental_split": split,
                "class_name": class_name,
            }
        )

    # Tampered rows without a readable mask must be excluded and counted.
    _write_image(root / "images/tampered_broken.png", 99)
    rows.append(
        {
            "sample_id": "sample-broken",
            "class_name": "Tampered",
            "image_path": "images/tampered_broken.png",
            "mask_path": "masks/does_not_exist.png",
            "split": "train",
        }
    )
    split_rows.append(
        {
            "sample_id": "sample-broken",
            "experimental_split": "train",
            "class_name": "Tampered",
        }
    )

    manifest = tmp_path / "manifest.csv"
    split_csv = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    pd.DataFrame(split_rows).to_csv(split_csv, index=False)
    return root, manifest, split_csv


class _TinyMVSS(nn.Module):
    """Stub with the real model's output contract but a few parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, images: torch.Tensor):
        count, _, height, width = images.shape
        edge_logits = torch.zeros(
            (count, 1, height // 4, width // 4), device=images.device
        ) + self.bias
        mask_logits = torch.zeros((count, 1, height, width), device=images.device) + self.bias
        return edge_logits, mask_logits


def _args(tmp_path: Path, root: Path, manifest: Path, split_csv: Path) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=manifest,
        split_csv=split_csv,
        data_root=root,
        output_dir=tmp_path / "run",
        init_checkpoint=None,
        resume=None,
        overwrite=False,
        smoke=True,
        limit_train=0,
        limit_val=0,
        image_size=32,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        val_threshold=0.5,
        num_workers=0,
        seed=7,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------


def test_build_records_excludes_unmasked_tampered_rows(tiny_dataset):
    root, manifest, split_csv = tiny_dataset
    records, skipped = build_records(manifest, split_csv, root)

    assert len(records) == 9
    assert [row["sample_id"] for row in skipped] == ["sample-broken"]
    assert all(record.mask_path is not None for record in records if record.class_name == "Tampered")
    assert {record.split for record in records} == {"train", "validation", "test"}
    assert all(record.image_path.is_file() for record in records)


def test_build_records_reports_missing_images(tiny_dataset):
    root, manifest, split_csv = tiny_dataset
    (root / "images" / "authentic_2.png").unlink()

    with pytest.raises(FileNotFoundError):
        build_records(manifest, split_csv, root)


def test_build_records_ignores_rows_outside_the_split(tiny_dataset, tmp_path):
    root, manifest, split_csv = tiny_dataset
    split_frame = pd.read_csv(split_csv)
    reduced = tmp_path / "reduced_split.csv"
    split_frame[split_frame["experimental_split"] == "train"].to_csv(reduced, index=False)

    records, _ = build_records(manifest, reduced, root)
    assert {record.split for record in records} == {"train"}


# ---------------------------------------------------------------------------
# Dataset parity
# ---------------------------------------------------------------------------


def test_dataset_tensor_shapes_and_masks(tiny_dataset):
    root, manifest, split_csv = tiny_dataset
    records, _ = build_records(manifest, split_csv, root)
    dataset = GroupAwareDataset(records, image_size=32, augment=False)

    image, mask, edge, label = dataset[0]
    assert image.shape == (3, 32, 32) and image.dtype == torch.float32
    assert mask.shape == (1, 32, 32) and mask.dtype == torch.float32
    assert edge.shape == (1, 8, 8) and edge.dtype == torch.float32
    assert label.shape == (1,)
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}

    authentic_index = next(
        index for index, record in enumerate(dataset.records) if record.class_name == "Authentic"
    )
    _, authentic_mask, _, authentic_label = dataset[authentic_index]
    assert float(authentic_mask.sum()) == 0.0
    assert float(authentic_label.item()) == 0.0


# ---------------------------------------------------------------------------
# Training pass
# ---------------------------------------------------------------------------


def test_training_writes_artifacts_and_stays_off_the_test_split(tiny_dataset, tmp_path):
    root, manifest, split_csv = tiny_dataset
    args = _args(tmp_path, root, manifest, split_csv)

    summary = train(args, model_factory=lambda device: _TinyMVSS().to(device))

    output = args.output_dir
    for name in ("last.pt", "best.pt", "mvssnet_groupaware.pt", "training_log.csv", "run.json", "split_used.csv"):
        assert (output / name).is_file(), name

    assert summary["mode"] == "smoke"
    assert summary["samples"]["excluded_tampered_without_mask"] == 1
    assert "test" not in json.dumps(summary["final_validation"]).lower()

    used = pd.read_csv(output / "split_used.csv")
    assert set(used["split"]) == {"train", "validation"}

    payload = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert isinstance(payload["config"]["output_dir"], str)
    assert payload["config"]["output_dir"].endswith("run")

    state = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert state["epoch"] == 0
    assert isinstance(state["config"]["manifest"], str)
    assert set(state["model_state"]) == {"bias"}

    exported = torch.load(output / "mvssnet_groupaware.pt", map_location="cpu", weights_only=False)
    assert isinstance(exported, dict) and set(exported) == {"bias"}


def test_training_refuses_to_overwrite_output(tiny_dataset, tmp_path):
    root, manifest, split_csv = tiny_dataset
    args = _args(tmp_path, root, manifest, split_csv)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        train(args, model_factory=lambda device: _TinyMVSS().to(device))

    assert (args.output_dir / "existing.txt").read_text(encoding="utf-8") == "keep"


def test_jsonable_handles_nested_paths():
    payload = _jsonable({"a": Path("x") / "y", "b": [Path("z"), 1], "c": ("w",)})
    assert payload == {"a": str(Path("x") / "y"), "b": ["z", 1], "c": ["w"]}
    json.dumps(payload)
