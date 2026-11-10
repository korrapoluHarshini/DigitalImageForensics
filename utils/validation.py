"""Upload validation.

The web app must never trust a filename, a MIME type supplied by the browser,
or the bytes themselves.  Validation happens in three layers:

1. size limit (checked before decoding),
2. Pillow format verification (the bytes must actually decode),
3. pixel-count limit (protects against decompression bombs).
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .config import settings


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}

# Pillow raises DecompressionBombError above this, and a warning above half.
Image.MAX_IMAGE_PIXELS = settings.max_image_pixels


class UploadValidationError(ValueError):
    """Raised when an uploaded file is not a safe, supported image."""


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def sanitize_filename(name: str | None, fallback: str = "upload") -> str:
    """Return a short display-only filename without paths or control chars.

    This value is never used to build filesystem paths.
    """
    if not name:
        return fallback
    base = Path(str(name).replace("\\", "/")).name
    base = _CONTROL_CHARS.sub("", base).strip()
    if not base:
        return fallback
    stem = Path(base).stem[:60] or fallback
    suffix = Path(base).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        suffix = ""
    return f"{stem}{suffix}"


def validate_upload(
    data: bytes,
    filename: str | None,
    *,
    max_mb: int | None = None,
) -> str:
    """Validate uploaded bytes and return the detected Pillow format.

    Parameters
    ----------
    data:
        Raw uploaded bytes.
    filename:
        Original client filename (used only for the extension check).
    max_mb:
        Optional override for the configured size limit.

    Raises
    ------
    UploadValidationError
        If the file is empty, too large, has a disallowed extension, is not a
        supported image format, is corrupted, or exceeds the pixel limit.
    """
    limit_mb = settings.max_upload_mb if max_mb is None else int(max_mb)
    limit_bytes = limit_mb * 1024 * 1024

    if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
        raise UploadValidationError("The uploaded file is empty.")
    if len(data) > limit_bytes:
        raise UploadValidationError(
            f"The file is larger than the {limit_mb} MB limit. "
            "Please upload a smaller image."
        )

    suffix = Path(str(filename or "")).suffix.lower()
    if suffix and suffix not in ALLOWED_EXTENSIONS:
        raise UploadValidationError(
            "Unsupported file extension. Use JPG, JPEG, PNG or WEBP."
        )

    try:
        with Image.open(io.BytesIO(data)) as probe:
            detected = (probe.format or "").upper()
            width, height = probe.size
            probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise UploadValidationError(
            "The file could not be decoded as an image. "
            "It may be corrupted or mislabelled."
        ) from exc
    except Image.DecompressionBombError as exc:  # pragma: no cover - Pillow version dependent
        raise UploadValidationError(
            "The image dimensions are too large to analyse safely."
        ) from exc

    if detected not in ALLOWED_PIL_FORMATS:
        raise UploadValidationError(
            f"Detected format {detected or 'unknown'} is not supported. "
            "Use JPG, JPEG, PNG or WEBP."
        )

    if width <= 0 or height <= 0 or width * height > settings.max_image_pixels:
        raise UploadValidationError(
            "The image dimensions are too large to analyse safely."
        )

    return detected


def validate_image_array(array) -> None:
    """Validate a decoded image array passed directly to a module."""
    import numpy as np

    if not isinstance(array, np.ndarray):
        raise UploadValidationError("The image must be a NumPy array.")
    if array.ndim not in (2, 3):
        raise UploadValidationError("The image array has an unexpected shape.")
    if array.ndim == 3 and array.shape[2] not in (3, 4):
        raise UploadValidationError("The image must have 1, 3 or 4 channels.")
    if array.size == 0:
        raise UploadValidationError("The image is empty.")
    if array.shape[0] < 8 or array.shape[1] < 8:
        raise UploadValidationError("The image is too small to analyse.")
