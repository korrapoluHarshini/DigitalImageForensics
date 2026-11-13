"""Tests for Module 3 reference-screenshot comparison."""

from __future__ import annotations

import json

import numpy as np
import pytest

from modules.chat_tampering.inference import detect_chat_tampering
from modules.chat_tampering.ocr import OCRLine, OCRResult
from modules.chat_tampering.reference import (
    CHANGE_AMOUNT,
    CHANGE_DATE,
    CHANGE_MESSAGE,
    CHANGE_NAME,
    CHANGE_TIME,
    FINDING_DIFFERENCES,
    FINDING_MISMATCH,
    FINDING_NO_DIFFERENCES,
    STATUS_INVALID,
    STATUS_MISMATCH,
    STATUS_OK,
    align_reference,
    classify_change,
    compare_chat_screenshot,
    detect_text_changes,
)


def _chat_screenshot() -> np.ndarray:
    """Small deterministic screenshot-like image (header, bubbles, footer)."""
    image = np.full((420, 300, 3), 246, dtype=np.uint8)
    image[0:52, :] = (235, 238, 242)
    image[8:30, 12:120] = (60, 64, 72)          # header name
    image[70:130, 20:270] = (255, 255, 255)     # received bubble
    image[86:104, 36:220] = (30, 30, 30)        # message text
    image[160:220, 40:285] = (198, 235, 205)    # sent bubble
    image[176:194, 60:250] = (20, 60, 30)       # amount text
    image[380:400, 90:210] = (120, 120, 120)    # timestamp
    return image


def _ocr(lines: list[tuple[str, tuple[int, int, int, int], float]]) -> OCRResult:
    return OCRResult(
        available=True,
        backend="fake",
        lines=[OCRLine(text=text, confidence=confidence, bbox=bbox) for text, bbox, confidence in lines],
        full_text="\n".join(text for text, _bbox, _confidence in lines),
    )


@pytest.fixture()
def reference_ocr() -> OCRResult:
    return _ocr(
        [
            ("Alice", (12, 8, 120, 30), 0.95),
            ("Paid 500", (36, 86, 220, 104), 0.92),
            ("12/05/2024", (90, 380, 210, 400), 0.90),
        ]
    )


def test_classify_change_categories():
    assert classify_change("Paid 500", "Paid 5,000") == CHANGE_AMOUNT
    assert classify_change("12/05/2024", "13/05/2024") == CHANGE_DATE
    assert classify_change("12 May 2024", "14 Jul 2025") == CHANGE_DATE
    assert classify_change("10:30 AM", "11:30 AM") == CHANGE_TIME
    assert classify_change("11.38", "12.38") == CHANGE_TIME
    assert classify_change("Alice", "Bob", location="header") == CHANGE_NAME
    assert classify_change("See you soon", "See you later") == CHANGE_MESSAGE


def test_adjacent_ocr_fragments_of_one_edit_are_merged():
    reference = _ocr([("Aarav Mehta", (124, 66, 320, 95), 0.93)])
    suspect = _ocr(
        [
            ("A", (124, 64, 140, 80), 0.80),
            ("Mehta", (146, 66, 240, 95), 0.92),
        ]
    )
    changes = detect_text_changes(reference, suspect, image_size=(720, 1280))
    assert len(changes) == 1
    assert changes[0].kind == CHANGE_NAME
    assert changes[0].location == "header"


def test_separate_edits_are_not_merged():
    reference = _ocr(
        [
            ("Paid 500", (36, 300, 220, 330), 0.9),
            ("See you", (36, 520, 220, 550), 0.9),
        ]
    )
    suspect = _ocr(
        [
            ("Paid 900", (36, 300, 220, 330), 0.9),
            ("See you soon", (36, 520, 220, 550), 0.9),
        ]
    )
    changes = detect_text_changes(reference, suspect, image_size=(720, 1280))
    assert len(changes) == 2


def test_detect_text_changes_reports_kinds_and_boxes(reference_ocr):
    suspect_ocr = _ocr(
        [
            ("Bob", (12, 8, 120, 30), 0.93),
            ("Paid 5,000", (36, 86, 220, 104), 0.94),
            ("12/05/2024", (90, 380, 210, 400), 0.90),
        ]
    )
    changes = detect_text_changes(reference_ocr, suspect_ocr, image_size=(300, 420))
    kinds = {change.kind for change in changes}
    assert CHANGE_NAME in kinds
    assert CHANGE_AMOUNT in kinds
    # The identical date line must not be reported.
    assert len(changes) == 2
    assert all(change.confidence > 0 for change in changes)
    assert all(change.reference_bbox and change.suspect_bbox for change in changes)


def test_detect_text_changes_does_not_require_both_sides():
    reference = _ocr([("Only here", (10, 10, 100, 30), 0.9)])
    changes = detect_text_changes(reference, _ocr([]), image_size=(300, 420))
    assert len(changes) == 1
    assert changes[0].reference_text == "Only here"
    assert changes[0].suspect_text == ""
    assert changes[0].confidence < 0.9


def test_identical_reference_has_no_differences():
    image = _chat_screenshot()
    comparison = compare_chat_screenshot(image, image.copy(), run_ocr=False)
    assert comparison.status == STATUS_OK
    assert comparison.finding == FINDING_NO_DIFFERENCES
    assert comparison.changed_pixels == 0
    assert comparison.regions == []
    assert comparison.overlay is not None
    assert comparison.side_by_side is not None
    assert comparison.change_mask is not None


def test_visible_edit_produces_region_and_overlay():
    reference = _chat_screenshot()
    suspect = reference.copy()
    suspect[176:194, 60:250] = (250, 250, 250)
    suspect[180:190, 70:240] = (10, 10, 10)

    comparison = compare_chat_screenshot(suspect, reference, run_ocr=False, min_region_area=10)
    assert comparison.status == STATUS_OK
    assert comparison.finding == FINDING_DIFFERENCES
    assert comparison.changed_pixels > 0
    assert comparison.regions
    x1, y1, x2, y2 = comparison.regions[0].bbox
    assert y1 < 195 and y2 > 175
    assert 0.0 < comparison.changed_pixel_ratio < 0.5
    assert comparison.change_mask.shape == reference.shape[:2]
    assert comparison.overlay.shape == reference.shape


def test_resized_reference_is_resized_then_aligned():
    reference = _chat_screenshot()
    suspect = reference.copy()
    suspect[176:194, 60:250] = (250, 250, 250)

    small = reference[::2, ::2].copy()
    aligned, alignment = align_reference(small, suspect)
    assert alignment.resized_reference is True
    assert aligned.shape == suspect.shape
    # The registration must not move the comparison outside the image.
    assert np.isfinite(alignment.correlation)


def test_unrelated_reference_is_reported_as_mismatch():
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 256, (420, 300, 3), dtype=np.uint8)
    comparison = compare_chat_screenshot(_chat_screenshot(), noise, run_ocr=False)
    assert comparison.status in {STATUS_MISMATCH}
    assert comparison.finding == FINDING_MISMATCH
    assert any("alignment" in warning.lower() or "global difference" in warning.lower()
               for warning in comparison.warnings)


def test_invalid_reference_returns_explicit_error():
    comparison = compare_chat_screenshot(_chat_screenshot(), np.zeros((0, 0, 3), np.uint8))
    assert comparison.status == STATUS_INVALID
    assert comparison.error_message


def test_comparison_summary_never_contains_ocr_text_by_default(reference_ocr, monkeypatch):
    suspect_ocr = _ocr([("Bob", (12, 8, 120, 30), 0.93)])
    image = _chat_screenshot()

    calls = {"count": 0}

    def fake_extract(_image, languages=None):
        calls["count"] += 1
        return reference_ocr if calls["count"] == 1 else suspect_ocr

    monkeypatch.setattr("modules.chat_tampering.reference.extract_text", fake_extract)
    comparison = compare_chat_screenshot(image, image, run_ocr=True)
    payload = json.dumps(comparison.as_dict())
    assert "Alice" not in payload
    assert "Bob" not in payload
    assert "reference_text" not in payload
    assert comparison.as_dict(include_text=True)["text_changes"][0]["reference_text"] == "Alice"


def test_inference_accepts_optional_reference(monkeypatch, rgb_image, fake_predictor):
    reference = _chat_screenshot()
    suspect = reference.copy()
    suspect[176:194, 60:250] = (250, 250, 250)

    fake_ocr = _ocr([("Paid 500", (60, 170, 250, 200), 0.9)])
    monkeypatch.setattr("modules.chat_tampering.inference.extract_text", lambda _image: fake_ocr)
    monkeypatch.setattr(
        "modules.chat_tampering.reference.extract_text", lambda _image, languages=None: fake_ocr
    )

    result = detect_chat_tampering(
        suspect,
        predictor=fake_predictor,
        reference_rgb=reference,
    )
    assert result.reference_comparison is not None
    assert result.reference_comparison.status == STATUS_OK
    assert any("reference screenshot was supplied" in warning.lower() for warning in result.warnings)
    assert result.to_summary()["reference_comparison"]["finding"] in {
        FINDING_DIFFERENCES,
        FINDING_NO_DIFFERENCES,
        FINDING_MISMATCH,
    }


def test_inference_without_reference_keeps_no_reference_state(rgb_image, fake_predictor):
    result = detect_chat_tampering(rgb_image, predictor=fake_predictor, run_ocr=False)
    assert result.reference_comparison is None
    assert result.to_summary()["reference_comparison"] is None
    assert any("no original chat record" in warning.lower() for warning in result.warnings)
