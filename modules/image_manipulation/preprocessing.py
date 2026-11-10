"""Preprocessing for MVSS-Net.

The official evaluation pipeline reads images with OpenCV (BGR), resizes them to
512x512 and normalises with ImageNet statistics *without* an RGB/BGR swap.
``preprocess_bgr`` reproduces that pipeline and is verified against the
repository's own ``common.transforms.direct_val`` in the test suite.

Public functions accept RGB arrays (the convention used by the app) and convert
to BGR internally.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from utils.image_utils import ensure_rgb_uint8, rgb_to_bgr


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_INPUT_SIZE = 512
DEFAULT_PATCH_SIZE = 512
DEFAULT_OVERLAP = 128


def resize_square(bgr_image: np.ndarray, size: int = DEFAULT_INPUT_SIZE) -> np.ndarray:
    """Resize a BGR image to ``size x size`` using the official interpolation."""
    return cv2.resize(bgr_image, (size, size), interpolation=cv2.INTER_LINEAR)


def preprocess_bgr(bgr_image: np.ndarray, size: int = DEFAULT_INPUT_SIZE) -> torch.Tensor:
    """Convert a BGR uint8 image into a normalised ``1x3xSxS`` float tensor."""
    if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
        raise ValueError("MVSS-Net expects a 3-channel BGR image.")
    resized = resize_square(bgr_image, size=size).astype(np.float32) / 255.0
    normalized = (resized - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0).float()


def preprocess_rgb(rgb_image: np.ndarray, size: int = DEFAULT_INPUT_SIZE) -> torch.Tensor:
    """Convenience wrapper: RGB uint8 -> normalised BGR-order tensor."""
    return preprocess_bgr(rgb_to_bgr(ensure_rgb_uint8(rgb_image)), size=size)


def resize_probability_map(probability_map: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a probability map to the original image dimensions."""
    return cv2.resize(
        np.asarray(probability_map, dtype=np.float32),
        (int(width), int(height)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)


def get_patch_positions(length: int, patch: int, overlap: int) -> list[int]:
    """Start positions for overlapping patches covering ``length``."""
    if length <= patch:
        return [0]
    stride = max(1, patch - overlap)
    positions = list(range(0, length - patch + 1, stride))
    last = length - patch
    if positions[-1] != last:
        positions.append(last)
    return positions


def gaussian_window(patch: int, sigma: float | None = None) -> np.ndarray:
    """Normalised 2D Gaussian used to blend overlapping patch predictions."""
    center = patch // 2
    sigma = sigma or patch / 4.0
    y, x = np.ogrid[:patch, :patch]
    window = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2.0 * sigma**2)).astype(np.float32)
    window /= max(float(window.max()), 1e-8)
    return window


def count_tiles(height: int, width: int, patch: int, overlap: int) -> int:
    return len(get_patch_positions(height, patch, overlap)) * len(
        get_patch_positions(width, patch, overlap)
    )


def downscale_for_tile_budget(
    bgr_image: np.ndarray,
    patch: int,
    overlap: int,
    max_tiles: int,
) -> tuple[np.ndarray, float]:
    """Downscale an image until the patch grid fits within ``max_tiles``.

    MVSS-Net's attention blocks are memory hungry, so tall chat screenshots can
    otherwise create dozens of patches.  The returned scale is relative to the
    input image.
    """
    height, width = bgr_image.shape[:2]
    if max_tiles <= 0:
        return bgr_image, 1.0
    scale = 1.0
    current = bgr_image
    while count_tiles(current.shape[0], current.shape[1], patch, overlap) > max_tiles:
        scale *= 0.85
        if scale < 0.15:
            break
        new_width = max(patch, int(round(width * scale)))
        new_height = max(patch, int(round(height * scale)))
        current = cv2.resize(bgr_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return current, scale
