"""Visual explanations for manipulation/localization results."""

from __future__ import annotations

import cv2
import numpy as np

from utils.image_utils import ensure_rgb_uint8, rgb_to_bgr


ACCENT = (255, 64, 64)      # red-ish in RGB
BOX_COLOR = (255, 210, 0)   # amber in RGB


def probability_to_heatmap(probability_map: np.ndarray) -> np.ndarray:
    """JET colour map of a probability map (RGB uint8)."""
    probability = np.clip(np.asarray(probability_map, dtype=np.float32), 0.0, 1.0)
    grayscale = (probability * 255.0).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(grayscale, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def mask_to_rgb(binary_mask: np.ndarray) -> np.ndarray:
    """White-on-black visualization of a binary mask."""
    mask = np.asarray(binary_mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    grayscale = (mask > 0).astype(np.uint8) * 255
    return np.repeat(grayscale[..., None], 3, axis=2)


def overlay_mask(
    rgb_image: np.ndarray,
    binary_mask: np.ndarray,
    color: tuple[int, int, int] = ACCENT,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a translucent colour over every masked pixel."""
    rgb = ensure_rgb_uint8(rgb_image)
    mask = np.asarray(binary_mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    suspicious = mask > 0
    if not suspicious.any():
        return rgb.copy()

    overlay = rgb.copy()
    color_layer = np.zeros_like(rgb)
    color_layer[:, :] = np.array(color, dtype=np.uint8)
    blended = cv2.addWeighted(rgb, 1.0 - alpha, color_layer, alpha, 0)
    overlay[suspicious] = blended[suspicious]
    return overlay


def draw_regions(
    rgb_image: np.ndarray,
    regions,
    color: tuple[int, int, int] = BOX_COLOR,
    label: bool = True,
) -> np.ndarray:
    """Draw numbered bounding boxes for suspicious regions."""
    canvas = ensure_rgb_uint8(rgb_image).copy()
    if not regions:
        return canvas

    height, width = canvas.shape[:2]
    thickness = max(2, int(round(min(height, width) / 320)))
    font_scale = max(0.45, min(1.2, min(height, width) / 900))
    bgr_color = (int(color[2]), int(color[1]), int(color[0]))

    for region in regions:
        x, y, w, h = [int(value) for value in region.bbox]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), bgr_color, thickness)
        if not label:
            continue
        text = f"R{region.region_id}"
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1)
        )
        box_top = max(0, y - text_h - baseline - 6)
        cv2.rectangle(
            canvas,
            (x, box_top),
            (x + text_w + 8, y),
            bgr_color,
            thickness=-1,
        )
        cv2.putText(
            canvas,
            text,
            (x + 4, y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (20, 20, 20),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
    return canvas


def build_manipulation_visuals(
    rgb_image: np.ndarray,
    probability_map: np.ndarray,
    binary_mask: np.ndarray,
    regions,
) -> dict:
    """Return the four visual explanations shown in the UI."""
    heatmap = probability_to_heatmap(probability_map)
    mask_visual = mask_to_rgb(binary_mask)
    overlay = overlay_mask(rgb_image, binary_mask)
    annotated = draw_regions(overlay, regions)
    return {
        "heatmap": heatmap,
        "mask": mask_visual,
        "overlay": overlay,
        "annotated": annotated,
    }


def render_ocr_boxes(rgb_image: np.ndarray, lines, color: tuple[int, int, int] = (0, 200, 120)) -> np.ndarray:
    """Draw OCR line boxes (green) without printing the recognised text."""
    canvas = ensure_rgb_uint8(rgb_image).copy()
    if not lines:
        return canvas
    height, width = canvas.shape[:2]
    thickness = max(1, int(round(min(height, width) / 500)))
    bgr_color = (int(color[2]), int(color[1]), int(color[0]))
    for line in lines[:80]:
        x1, y1, x2, y2 = [int(value) for value in line.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr_color, thickness)
    return canvas
