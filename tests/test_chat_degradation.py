"""Tests for the Module 3 degradation stress-test helpers."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.evaluate_chat_degradation import (
    CONDITIONS,
    _degrade,
    _jpeg,
    _rescale,
)


def _sample_image(seed: int = 0, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    image[8:24, 8:56] = 255
    return image


def test_no_degradation_is_a_no_op():
    image = _sample_image()
    assert _degrade(image, None) is image


def test_jpeg_preserves_shape_and_dtype_and_changes_values():
    image = _sample_image()
    degraded = _jpeg(image, 50)
    assert degraded.shape == image.shape
    assert degraded.dtype == np.uint8
    assert not np.array_equal(degraded, image)


def test_jpeg_quality_controls_error():
    image = _sample_image()
    high = _jpeg(image, 95).astype(np.float32)
    low = _jpeg(image, 30).astype(np.float32)
    high_error = np.abs(high - image.astype(np.float32)).mean()
    low_error = np.abs(low - image.astype(np.float32)).mean()
    assert low_error > high_error


def test_rescale_round_trips_to_original_size():
    image = _sample_image(size=80)
    degraded = _rescale(image, 0.5)
    assert degraded.shape == image.shape
    assert degraded.dtype == np.uint8
    assert not np.array_equal(degraded, image)


def test_conditions_cover_clean_and_degraded_cases():
    names = {condition["name"] for condition in CONDITIONS}
    assert "clean" in names
    assert any("jpeg" in name for name in names)
    assert any("rescale" in name for name in names)
    clean = next(condition for condition in CONDITIONS if condition["name"] == "clean")
    assert clean["reference"] is None and clean["suspect"] is None


def test_unknown_degradation_is_rejected():
    with pytest.raises(ValueError, match="Unknown degradation"):
        _degrade(_sample_image(), {"blur": 3})
