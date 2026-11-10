from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from utils.image_utils import decode_image_bytes, encode_png_bytes, image_fingerprint
from utils.validation import (
    UploadValidationError,
    sanitize_filename,
    validate_image_array,
    validate_upload,
)


def _png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_valid_png_returns_rgb(rgb_image):
    decoded = decode_image_bytes(_png_bytes(rgb_image), "photo.png")
    assert decoded.shape == rgb_image.shape
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, rgb_image)


def test_decode_applies_exif_orientation_or_at_least_preserves_size(rgb_image):
    decoded = decode_image_bytes(_png_bytes(rgb_image), "photo.PNG")
    assert decoded.shape[:2] == rgb_image.shape[:2]


def test_rejects_empty_upload():
    with pytest.raises(UploadValidationError):
        validate_upload(b"", "empty.png")


def test_rejects_oversized_upload():
    with pytest.raises(UploadValidationError):
        validate_upload(b"x" * (2 * 1024 * 1024), "large.png", max_mb=1)


def test_rejects_bad_extension():
    valid = _png_bytes(np.zeros((32, 32, 3), dtype=np.uint8))
    with pytest.raises(UploadValidationError):
        validate_upload(valid, "image.gif")


def test_rejects_corrupted_bytes():
    with pytest.raises(UploadValidationError):
        decode_image_bytes(b"not an image", "fake.png")


def test_rejects_text_masquerading_as_png():
    with pytest.raises(UploadValidationError):
        validate_upload(b"<html>hello</html>", "payload.png")


def test_sanitize_filename_strips_paths_and_control_characters():
    assert sanitize_filename(r"..\..\secret image.PNG") == "secret image.png"
    assert "\x00" not in sanitize_filename("bad\x00name.png")
    assert sanitize_filename("") == "upload"
    assert sanitize_filename("image.exe") == "image"


def test_validate_image_array_rejects_wrong_shape():
    with pytest.raises(UploadValidationError):
        validate_image_array(np.zeros((10, 10, 5), dtype=np.uint8))
    with pytest.raises(UploadValidationError):
        validate_image_array(np.zeros((4, 4, 3), dtype=np.uint8))


def test_png_encoding_round_trip(rgb_image):
    encoded = encode_png_bytes(rgb_image)
    assert encoded.startswith(b"\x89PNG")
    decoded = decode_image_bytes(encoded, "roundtrip.png")
    np.testing.assert_array_equal(decoded, rgb_image)


def test_fingerprint_is_stable_and_short(rgb_image):
    first = image_fingerprint(rgb_image)
    second = image_fingerprint(rgb_image.copy())
    assert first == second
    assert len(first) == 16
