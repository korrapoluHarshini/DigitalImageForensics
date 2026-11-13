"""Tests for the independent group-aware Module 1 evaluation script."""

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

from evaluation.evaluate_groupaware_checkpoint import (  # noqa: E402
    _auc,
    _load_state_dict,
    _select_records,
    run,
)
from training.train_mvss_groupaware import Record  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_image(path: Path, seed: int, *, size: int = 32, mask: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask:
        array = np.zeros((size, size), dtype=np.uint8)
        array[size // 2 :, size // 2 :] = 255
        Image.fromarray(array, mode="L").save(path)
    else:
        rng = np.random.default_rng(seed)
        array = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        Image.fromarray(array).save(path)


@pytest.fixture()
def eval_dataset(tmp_path: Path):
    root = tmp_path / "data"
    rows = []
    split_rows = []
    for index, (class_name, split, has_mask) in enumerate(
        [
            ("Tampered", "test", True),
            ("Tampered", "validation", True),
            ("Authentic", "test", False),
            ("Authentic", "test", False),
            ("Tampered", "test", True),
        ]
    ):
        image_rel = f"images/{class_name.lower()}_{index}.png"
        _write_image(root / image_rel, index)
        mask_rel = ""
        if has_mask:
            mask_rel = f"masks/{class_name.lower()}_{index}_gt.png"
            _write_image(root / mask_rel, index, mask=True)
        sample_id = f"sample-{index}"
        rows.append(
            {
                "sample_id": sample_id,
                "class_name": class_name,
                "image_path": image_rel,
                "mask_path": mask_rel,
            }
        )
        split_rows.append(
            {
                "sample_id": sample_id,
                "experimental_split": split,
                "class_name": class_name,
            }
        )
    manifest = tmp_path / "manifest.csv"
    split_csv = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    pd.DataFrame(split_rows).to_csv(split_csv, index=False)
    return root, manifest, split_csv


class _AllPositiveStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(2.0))

    def forward(self, images: torch.Tensor):
        count, _, height, width = images.shape
        logits = torch.zeros((count, 1, height, width), device=images.device) + self.bias
        return logits, logits


def _args(tmp_path: Path, root: Path, manifest: Path, split_csv: Path) -> argparse.Namespace:
    checkpoint = tmp_path / "model.pt"
    torch.save({"bias": torch.tensor(2.0)}, checkpoint)
    return argparse.Namespace(
        checkpoint=checkpoint,
        manifest=manifest,
        split_csv=split_csv,
        data_root=root,
        split="test",
        output=tmp_path / "metrics.json",
        per_image_csv=tmp_path / "per_image.csv",
        save_probabilities=None,
        image_size=32,
        batch_size=2,
        num_workers=0,
        torch_threads=1,
        pixel_threshold=0.4,
        image_threshold=0.42,
        image_uncertainty_margin=0.13,
        limit_tampered=0,
        limit_authentic=0,
        seed=3,
        smoke=False,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Unit logic
# ---------------------------------------------------------------------------


def test_auc_handles_ties_and_ordering():
    assert _auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == pytest.approx(1.0)
    assert _auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == pytest.approx(0.0)
    assert _auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) == pytest.approx(0.5)
    assert _auc(np.array([0.1, 0.2]), np.array([1, 1])) is None


def test_load_state_dict_unwraps_and_strips_module_prefix(tmp_path: Path):
    state = {"module.weight": torch.ones(2), "module.bias": torch.zeros(2)}
    for key in ("model_state", "state_dict", "model"):
        path = tmp_path / f"{key}.pt"
        torch.save({key: state, "epoch": 3}, path)
        loaded = _load_state_dict(path)
        assert set(loaded) == {"weight", "bias"}

    plain = tmp_path / "plain.pt"
    torch.save({"weight": torch.ones(1)}, plain)
    assert set(_load_state_dict(plain)) == {"weight"}


def test_select_records_is_deterministic_and_reports_pool_size():
    records = [
        Record(f"t{i}", "Tampered", Path(f"t{i}.png"), Path(f"t{i}.png"), "test") for i in range(6)
    ] + [
        Record(f"a{i}", "Authentic", Path(f"a{i}.png"), None, "test") for i in range(4)
    ] + [Record("v0", "Tampered", Path("v0.png"), Path("v0.png"), "validation")]

    first = _select_records(records, split="test", limit_tampered=2, limit_authentic=1, seed=5)
    second = _select_records(records, split="test", limit_tampered=2, limit_authentic=1, seed=5)
    assert [record.sample_id for record in first[0]] == [record.sample_id for record in second[0]]
    assert len(first[0]) == 2 and len(first[1]) == 1
    assert first[2] == 10

    all_records = _select_records(
        records, split="test", limit_tampered=0, limit_authentic=0, seed=1
    )
    assert len(all_records[0]) == 6 and len(all_records[1]) == 4


# ---------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------


def test_run_writes_metrics_for_test_split(eval_dataset, tmp_path):
    root, manifest, split_csv = eval_dataset
    args = _args(tmp_path, root, manifest, split_csv)

    summary = run(args, model_factory=lambda device: _AllPositiveStub().to(device))

    assert Path(summary["output"]).is_file()
    payload = json.loads(Path(summary["output"]).read_text(encoding="utf-8"))
    assert payload["split"] == "test"
    assert payload["samples"]["tampered"] == 2
    assert payload["samples"]["authentic"] == 2
    assert payload["samples"]["evaluated"] == 4
    assert payload["samples"]["sampled"] is False
    assert payload["warnings"] == []
    assert len(payload["checkpoint"]["sha256"]) == 64
    assert payload["pixel_metrics_at_fixed_threshold"]["micro_recall"] == pytest.approx(1.0)
    # Two authentic images contribute only false positives: TP=512, FP=3584.
    assert payload["pixel_metrics_at_fixed_threshold"]["micro_iou"] == pytest.approx(0.125, abs=1e-6)
    assert payload["image_level_metrics_at_fixed_threshold"]["tampered_recall"] == 1.0
    assert payload["image_level_metrics_at_fixed_threshold"]["authentic_false_positive_rate"] == 1.0

    frame = pd.read_csv(args.per_image_csv)
    assert len(frame) == 4
    assert set(frame["class_name"]) == {"Tampered", "Authentic"}


def test_run_flags_sampled_subset(eval_dataset, tmp_path):
    root, manifest, split_csv = eval_dataset
    args = _args(tmp_path, root, manifest, split_csv)
    args.limit_tampered = 1
    args.limit_authentic = 1

    summary = run(args, model_factory=lambda device: _AllPositiveStub().to(device))

    assert summary["samples"]["evaluated"] == 2
    assert summary["samples"]["sampled"] is True
    assert any("subset" in warning for warning in summary["warnings"])
