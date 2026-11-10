"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def rgb_image() -> np.ndarray:
    """A small deterministic RGB image with a visible rectangle."""
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    image[..., 0] = np.linspace(20, 220, 240, dtype=np.uint8)[None, :]
    image[..., 1] = np.linspace(10, 200, 160, dtype=np.uint8)[:, None]
    image[40:120, 80:180] = (240, 60, 40)
    return image


class FakePredictor:
    """Duck-typed stand-in for MvssNetPredictor (no weights, fast)."""

    def __init__(self, probability: float = 0.9, region: bool = True):
        self.probability = probability
        self.region = region

    def predict_probability(self, rgb_image, tiled=False, max_tiles=16, **_kwargs):
        height, width = rgb_image.shape[:2]
        probability_map = np.full((height, width), 0.02, dtype=np.float32)
        if self.region:
            probability_map[
                height // 4 : 3 * height // 4,
                width // 4 : 3 * width // 4,
            ] = self.probability
        return probability_map, {"tiled": bool(tiled), "tiles": 1, "device": "fake"}

    def describe(self):
        return {
            "name": "FakePredictor",
            "checkpoint": "none",
            "device": "fake",
            "input_size": 512,
            "loaded": True,
        }


@pytest.fixture()
def fake_predictor() -> FakePredictor:
    return FakePredictor()
