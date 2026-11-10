from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from modules.ai_generated_detection.preprocessing import (
    preprocess_ai_image,
    probabilities_from_output,
)
from modules.image_manipulation.preprocessing import (
    count_tiles,
    downscale_for_tile_budget,
    gaussian_window,
    get_patch_positions,
    preprocess_bgr,
    preprocess_rgb,
    resize_probability_map,
)


def test_preprocess_matches_official_direct_val(rgb_image):
    direct_val = pytest.importorskip("common.transforms").direct_val
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    # direct_val() only normalises; MVSS-Net evaluation resizes to 512x512 first.
    resized = cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
    expected = direct_val(resized.reshape(1, *resized.shape))[0].numpy()
    actual = preprocess_bgr(bgr)[0].numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_preprocess_rgb_equals_bgr_wrapper(rgb_image):
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    np.testing.assert_allclose(
        preprocess_rgb(rgb_image).numpy(),
        preprocess_bgr(bgr).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_preprocess_shape_and_dtype(rgb_image):
    tensor = preprocess_rgb(rgb_image, size=256)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.dtype == torch.float32


def test_patch_positions_cover_both_edges():
    positions = get_patch_positions(1300, 512, 128)
    assert positions[0] == 0
    assert positions[-1] == 1300 - 512
    assert all(b > a for a, b in zip(positions, positions[1:]))


def test_patch_positions_short_image():
    assert get_patch_positions(200, 512, 128) == [0]


def test_gaussian_window_is_normalised():
    window = gaussian_window(512)
    assert window.shape == (512, 512)
    assert window.dtype == np.float32
    assert pytest.approx(float(window.max()), rel=1e-6) == 1.0
    assert window[256, 256] > window[0, 0]


def test_downscale_respects_tile_budget():
    image = np.zeros((2200, 1300, 3), dtype=np.uint8)
    resized, scale = downscale_for_tile_budget(image, patch=512, overlap=128, max_tiles=4)
    assert scale <= 1.0
    assert count_tiles(resized.shape[0], resized.shape[1], 512, 128) <= 4


def test_probability_map_resize():
    probability = np.random.default_rng(0).random((32, 48)).astype(np.float32)
    resized = resize_probability_map(probability, 160, 240)
    assert resized.shape == (160, 240)
    assert resized.dtype == np.float32
    assert resized.min() >= 0.0 and resized.max() <= 1.0


def test_ai_preprocess_shape(rgb_image):
    tensor = preprocess_ai_image(rgb_image, input_size=224)
    assert tensor.shape == (1, 3, 224, 224)


def test_ai_preprocess_accepts_rectangular_size(rgb_image):
    tensor = preprocess_ai_image(rgb_image, input_size=(128, 96))
    assert tensor.shape == (1, 3, 128, 96)


def test_probabilities_from_logits_softmax():
    probabilities = probabilities_from_output(np.array([[0.0, 3.0]]))
    assert probabilities.shape == (2,)
    assert pytest.approx(float(probabilities.sum()), rel=1e-8) == 1.0
    assert probabilities[1] > probabilities[0]


def test_probabilities_from_single_logit():
    probabilities = probabilities_from_output(np.array([[0.0]]))
    assert probabilities.shape == (2,)
    assert pytest.approx(float(probabilities.sum()), rel=1e-8) == 1.0


def test_probabilities_from_already_normalised():
    input_probabilities = np.array([[0.25, 0.75]])
    probabilities = probabilities_from_output(input_probabilities)
    np.testing.assert_allclose(probabilities, input_probabilities[0], rtol=1e-9)
