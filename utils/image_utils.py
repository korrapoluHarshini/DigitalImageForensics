"""Image decoding, conversion and safe in-memory helpers."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import settings
from .validation import UploadValidationError, sanitize_filename, validate_image_array, validate_upload


Image.MAX_IMAGE_PIXELS = settings.max_image_pixels


def decode_image_bytes(data: bytes, filename: str | None = None, *, max_mb: int | None = None) -> np.ndarray:
    """Decode uploaded bytes into an RGB ``uint8`` HxWx3 array.

    EXIF orientation is applied so that analysis matches what the user sees.
    """
    validate_upload(data, filename, max_mb=max_mb)
    try:
        with Image.open(io.BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image)
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8).copy()
    except UploadValidationError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise UploadValidationError("The image could not be decoded.") from exc

    validate_image_array(array)
    return array


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image from disk as RGB ``uint8`` (evaluation scripts)."""
    path = Path(path)
    data = path.read_bytes()
    return decode_image_bytes(data, filename=path.name)


def ensure_rgb_uint8(image: np.ndarray, *, allow_rgba: bool = True) -> np.ndarray:
    """Normalise a NumPy image to RGB ``uint8`` HxWx3."""
    validate_image_array(image)
    array = image
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, 0.0, 1.0) * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.ndim == 3 and array.shape[2] == 4:
        if not allow_rgba:
            raise UploadValidationError("Unexpected alpha channel.")
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    return np.ascontiguousarray(array)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(ensure_rgb_uint8(image_rgb), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_long_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is at most ``max_side``.

    Returns the resized image and the scale factor that was applied.
    """
    if max_side <= 0:
        return image, 1.0
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def encode_png_bytes(image_rgb: np.ndarray) -> bytes:
    """Encode an RGB image as PNG bytes for download buttons."""
    rgb = ensure_rgb_uint8(image_rgb)
    success, buffer = cv2.imencode(".png", rgb_to_bgr(rgb))
    if not success:  # pragma: no cover - OpenCV failure
        raise RuntimeError("Could not encode the image as PNG.")
    return buffer.tobytes()


def encode_mask_png_bytes(mask: np.ndarray) -> bytes:
    """Encode a binary/grayscale uint8 mask as PNG bytes."""
    array = np.asarray(mask)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    success, buffer = cv2.imencode(".png", array)
    if not success:  # pragma: no cover
        raise RuntimeError("Could not encode the mask as PNG.")
    return buffer.tobytes()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_fingerprint(image_rgb: np.ndarray) -> str:
    """Short content fingerprint used in history/reporting (no filename)."""
    return hashlib.sha256(np.ascontiguousarray(image_rgb).tobytes()).hexdigest()[:16]


def read_uploaded_file(uploaded_file) -> tuple[bytes, str]:
    """Read a Streamlit UploadedFile into ``(bytes, display_name)``."""
    if uploaded_file is None:
        raise UploadValidationError("No file was uploaded.")
    data = uploaded_file.getvalue()
    name = sanitize_filename(getattr(uploaded_file, "name", None))
    return data, name


def describe_image(image_rgb: np.ndarray) -> dict:
    height, width = image_rgb.shape[:2]
    return {
        "width": int(width),
        "height": int(height),
        "megapixels": round((width * height) / 1_000_000, 2),
    }


def read_binary(source: str | Path | BinaryIO) -> bytes:
    if hasattr(source, "read"):
        return source.read()
    return Path(source).read_bytes()
