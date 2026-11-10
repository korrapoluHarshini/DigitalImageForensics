"""Visualisation helpers specific to chat screenshots."""

from __future__ import annotations

import cv2
import numpy as np

from utils.image_utils import ensure_rgb_uint8


OCR_COLOR = (0, 190, 120)  # green in RGB


def draw_ocr_blocks(rgb_image: np.ndarray, blocks, color: tuple[int, int, int] = OCR_COLOR) -> np.ndarray:
    """Draw OCR block boxes with neutral B1/B2 labels (never the text itself)."""
    canvas = ensure_rgb_uint8(rgb_image).copy()
    if not blocks:
        return canvas
    height, width = canvas.shape[:2]
    thickness = max(1, int(round(min(height, width) / 600)))
    bgr = (int(color[2]), int(color[1]), int(color[0]))
    font_scale = max(0.4, min(1.0, min(height, width) / 1000))
    for block in blocks[:60]:
        x1, y1, x2, y2 = [int(value) for value in block.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr, thickness)
        label = f"B{block.block_id}"
        cv2.putText(
            canvas,
            label,
            (x1 + 3, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            bgr,
            max(1, thickness),
            cv2.LINE_AA,
        )
    return canvas


def combine_forensic_and_ocr(
    forensic_overlay: np.ndarray,
    blocks,
) -> np.ndarray:
    """Forensic highlight + OCR boxes in one review image."""
    return draw_ocr_blocks(forensic_overlay, blocks)
