"""Preprocessing helpers for chat screenshots."""

from __future__ import annotations

import cv2
import numpy as np

from utils.image_utils import ensure_rgb_uint8, resize_long_side


MIN_OCR_SIDE = 700
MAX_OCR_SIDE = 3200


def prepare_chat_image(image_rgb: np.ndarray, max_side: int = 4096) -> tuple[np.ndarray, float]:
    """Bound memory use while keeping the whole screenshot in view."""
    rgb = ensure_rgb_uint8(image_rgb)
    return resize_long_side(rgb, max_side=max_side)


def prepare_for_ocr(
    image_rgb: np.ndarray,
    min_side: int = MIN_OCR_SIDE,
    max_side: int = MAX_OCR_SIDE,
) -> tuple[np.ndarray, float]:
    """Upscale tiny screenshots and downscale huge ones before OCR.

    Returns the prepared image and the scale factor relative to the input.
    """
    rgb = ensure_rgb_uint8(image_rgb)
    height, width = rgb.shape[:2]
    shortest = min(height, width)
    longest = max(height, width)

    scale = 1.0
    if shortest < min_side and shortest > 0:
        scale = min_side / float(shortest)
        # Do not let the other side explode past the max bound.
        if longest * scale > max_side:
            scale = max_side / float(longest)
    elif longest > max_side:
        scale = max_side / float(longest)

    if abs(scale - 1.0) < 1e-3:
        return rgb, 1.0

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(rgb, (new_width, new_height), interpolation=interpolation), scale


def crop_bbox(image_rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop ``(x1, y1, x2, y2)`` with bounds clamping."""
    x1, y1, x2, y2 = [int(value) for value in bbox]
    height, width = image_rgb.shape[:2]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return image_rgb[y1:y2, x1:x2]
