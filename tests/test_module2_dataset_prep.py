"""Tests for the GenImage-processed preparation helper.

The helper flattens real images and groups AI images by generator so that
``training/train_ai_detector.py`` sees a stable layout. These tests use a
synthetic tree shaped like the real Hugging Face extraction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training.prepare_module2_dataset import prepare


def _write_image(path: Path, value: int, *, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(value)
    grid = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    Image.fromarray(grid).save(path)


@pytest.fixture()
def extracted(tmp_path: Path) -> Path:
    root = tmp_path / "images"
    for split, offset in (("train", 0), ("val", 100), ("test", 200)):
        _write_image(root / split / "nature" / "ADM" / f"real_{split}_0.jpg", offset)
        _write_image(root / split / "nature" / "GLIDE" / f"real_{split}_1.jpg", offset + 1)
        _write_image(root / split / "ai" / "ADM" / f"adm_{split}_0.jpg", offset + 2)
        _write_image(root / split / "ai" / "GLIDE" / f"glide_{split}_0.jpg", offset + 3)
    return root


def test_prepare_builds_flat_real_and_generator_folders(extracted: Path, tmp_path: Path):
    output = tmp_path / "prepared"
    summary = prepare(extracted, output)

    assert summary["real"] == 6
    assert summary["ai"] == {"ADM": 3, "GLIDE": 3}
    assert summary["ai_total"] == 6

    real_files = sorted((output / "real").iterdir())
    assert len(real_files) == 6
    assert all(path.suffix == ".jpg" for path in real_files)

    assert sorted(path.name for path in (output / "ai").iterdir()) == ["ADM", "GLIDE"]
    assert len(list((output / "ai" / "ADM").iterdir())) == 3
    assert not (output / "ai" / "unknown").exists()


def test_prepare_is_idempotent(extracted: Path, tmp_path: Path):
    output = tmp_path / "prepared"
    first = prepare(extracted, output)
    second = prepare(extracted, output)
    assert first == second
    assert len(list((output / "real").iterdir())) == 6
    assert len(list((output / "ai" / "ADM").iterdir())) == 3


def test_prepare_smoke_limits_composition(extracted: Path, tmp_path: Path):
    output = tmp_path / "smoke"
    summary = prepare(extracted, output, smoke=True)
    # 6 real and 3 per generator are below the 200/100 smoke caps, so nothing
    # is removed; the mode and counts must still be reported truthfully.
    assert summary["mode"] == "smoke"
    assert summary["real"] == 6
    assert summary["ai"] == {"ADM": 3, "GLIDE": 3}


def test_prepare_rejects_a_tree_without_images(tmp_path: Path):
    source = tmp_path / "empty"
    (source / "train" / "ai" / "ADM").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        prepare(source, tmp_path / "output")
