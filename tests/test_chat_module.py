from __future__ import annotations

import numpy as np
import pytest

from modules.chat_tampering import detect_chat_tampering
from modules.chat_tampering.inference import (
    FINDING_NONE,
    FINDING_POSSIBLE,
    _associate_regions_with_text,
    _intersection_over_area,
)
from modules.chat_tampering.ocr import OCRBlock, OCRLine, OCRResult, _group_lines
from modules.image_manipulation.inference import SuspiciousRegion


def test_group_lines_merges_nearby_and_splits_far_rows():
    lines = [
        OCRLine("Hello", 0.9, (10, 10, 120, 30)),
        OCRLine("there", 0.9, (10, 34, 120, 54)),
        OCRLine("Later", 0.8, (10, 200, 120, 220)),
    ]
    blocks = _group_lines(lines, gap_factor=1.5)
    assert len(blocks) == 2
    assert blocks[0].text == "Hello\nthere"
    assert blocks[1].text == "Later"
    assert blocks[0].lines[0].block_id == 1


def test_group_lines_empty():
    assert _group_lines([]) == []


def test_intersection_over_area():
    assert _intersection_over_area((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 100)
    assert _intersection_over_area((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_associate_regions_with_text():
    regions = [
        SuspiciousRegion(1, (10, 10, 100, 50), 5000, 0.01, 0.8, 0.9),
        SuspiciousRegion(2, (300, 300, 40, 40), 1600, 0.004, 0.7, 0.8),
    ]
    blocks = [
        OCRBlock(1, "hello", 0.9, (20, 20, 90, 40)),
        OCRBlock(2, "elsewhere", 0.9, (400, 400, 500, 420)),
    ]
    matches = _associate_regions_with_text(regions, blocks)
    assert len(matches) == 1
    assert matches[0].region_id == 1
    assert matches[0].block_ids == [1]


def test_chat_pipeline_without_ocr(rgb_image, fake_predictor):
    result = detect_chat_tampering(
        rgb_image, threshold=0.5, predictor=fake_predictor, run_ocr=False
    )
    assert result.status == "ok"
    assert result.finding == FINDING_POSSIBLE
    assert result.ocr is None
    assert len(result.regions) >= 1
    assert result.annotated.shape == rgb_image.shape
    assert any("not message-level verification" in warning.lower() for warning in result.warnings)
    assert any("cannot recover the original message" in item.lower() for item in result.limitations)


def test_chat_pipeline_with_fake_ocr(rgb_image, fake_predictor, monkeypatch):
    fake_ocr = OCRResult(
        available=True,
        backend="fake",
        lines=[
            OCRLine("hello", 0.9, (90, 50, 170, 70)),
            OCRLine("world", 0.9, (90, 74, 170, 94)),
        ],
        blocks=[OCRBlock(1, "hello\nworld", 0.9, (90, 50, 170, 94))],
        full_text="hello\nworld",
        elapsed_seconds=0.1,
    )
    monkeypatch.setattr(
        "modules.chat_tampering.inference.extract_text",
        lambda _image: fake_ocr,
    )
    result = detect_chat_tampering(rgb_image, threshold=0.5, predictor=fake_predictor)
    assert result.status == "ok"
    assert result.ocr is not None and result.ocr.available
    assert result.ocr_visual is not None
    assert result.combined_visual is not None
    assert result.region_matches
    assert any("co-location" in warning.lower() for warning in result.warnings)


def test_chat_pipeline_visual_failure_still_returns_ocr(rgb_image, monkeypatch):
    class BrokenPredictor:
        def predict_probability(self, *_args, **_kwargs):
            raise RuntimeError("visual failure")

        def describe(self):
            return {}

    fake_ocr = OCRResult(
        available=True,
        backend="fake",
        lines=[OCRLine("hello", 0.9, (10, 10, 100, 30))],
        blocks=[OCRBlock(1, "hello", 0.9, (10, 10, 100, 30))],
        full_text="hello",
    )
    monkeypatch.setattr(
        "modules.chat_tampering.inference.extract_text",
        lambda _image: fake_ocr,
    )
    result = detect_chat_tampering(rgb_image, predictor=BrokenPredictor())
    assert result.status == "partial"
    assert result.ocr is not None and result.ocr.available
    assert result.manipulation is None
    assert any("only ocr" in warning.lower() for warning in result.warnings)


def test_chat_pipeline_negative_result(rgb_image, fake_predictor):
    fake_predictor.probability = 0.1
    fake_predictor.region = False
    result = detect_chat_tampering(rgb_image, predictor=fake_predictor, run_ocr=False)
    assert result.status == "ok"
    assert result.finding == FINDING_NONE
    assert result.regions == []
