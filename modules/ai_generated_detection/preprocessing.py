"""Preprocessing helpers for the pluggable AI-generated-image detector.

The concrete normalisation is declared by the integration manifest, because a
model supplied by another team member may use different input size or ImageNet
statistics.  Nothing here assumes a particular architecture.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


DEFAULT_INPUT_SIZE = 224
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def preprocess_ai_image(
    image_rgb: np.ndarray,
    input_size: int | tuple[int, int] = DEFAULT_INPUT_SIZE,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
) -> torch.Tensor:
    """RGB uint8 image -> normalised ``1x3xHxW`` float tensor.

    ``input_size`` may be an int (square) or ``(height, width)``.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with three channels.")
    if isinstance(input_size, int):
        target = (input_size, input_size)
    else:
        target = (int(input_size[0]), int(input_size[1]))

    resized = cv2.resize(image_rgb, (target[1], target[0]), interpolation=cv2.INTER_LINEAR)
    array = resized.astype(np.float32) / 255.0
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    normalized = (array - mean_array) / np.maximum(std_array, 1e-6)
    chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0).float()


def probabilities_from_output(output) -> np.ndarray:
    """Normalise a backend output into a 1-D probability vector."""
    array = np.asarray(output, dtype=np.float64)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    array = np.ravel(array)

    if array.size == 1:
        value = float(1.0 / (1.0 + np.exp(-array[0])))
        return np.array([1.0 - value, value], dtype=np.float64)

    if array.size < 2:
        raise ValueError("The AI detector must output at least two class scores.")

    # Already probabilities?
    if np.all(array >= 0) and abs(float(array.sum()) - 1.0) < 1e-3:
        return array.astype(np.float64)

    shifted = array - np.max(array)
    exponentials = np.exp(shifted)
    return (exponentials / np.maximum(exponentials.sum(), 1e-12)).astype(np.float64)
