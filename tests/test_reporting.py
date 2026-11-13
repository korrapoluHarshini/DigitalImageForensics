from __future__ import annotations

import json

from modules.chat_tampering import detect_chat_tampering
from modules.image_manipulation import detect_image_manipulation
from utils.reporting import (
    append_history,
    build_chat_report,
    build_manipulation_report,
    clear_history,
    load_history,
    record_from_chat,
    record_from_manipulation,
)


def test_history_round_trip_is_metadata_only(tmp_path, rgb_image, fake_predictor):
    result = detect_image_manipulation(rgb_image, threshold=0.5, predictor=fake_predictor)
    record = record_from_manipulation(result, "abc123")
    path = tmp_path / "history.jsonl"
    append_history(record, path)

    raw = path.read_text(encoding="utf-8")
    assert "image" not in raw.lower() or "Image Manipulation" in raw
    payload = json.loads(raw.strip())
    assert payload["module"] == "Image Manipulation"
    assert payload["fingerprint"] == "abc123"
    assert "probability_map" not in payload
    assert "regions" in payload["extra"]
    assert payload["extra"]["image_decision"] == result.image_decision
    assert payload["extra"]["localization_threshold"] == result.localization_threshold
    assert payload["extra"]["image_threshold"] == result.image_threshold
    assert payload["extra"]["localization_label"] == result.localization_label
    assert payload["extra"]["raw_suspicious_pixels"] == result.raw_suspicious_pixels
    assert (
        payload["extra"]["processed_suspicious_pixels"]
        == result.processed_suspicious_pixels
    )

    loaded = load_history(path)
    assert len(loaded) == 1
    assert loaded[0]["finding"] == result.finding

    clear_history(path)
    assert load_history(path) == []


def test_load_history_skips_corrupt_lines(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text('{"module":"x"}\nnot-json\n{"module":"y"}\n', encoding="utf-8")
    loaded = load_history(path)
    assert [entry["module"] for entry in loaded] == ["y", "x"]


def test_manipulation_report_contains_limitations(rgb_image, fake_predictor):
    result = detect_image_manipulation(rgb_image, threshold=0.5, predictor=fake_predictor)
    report = build_manipulation_report(result, "fingerprint")
    assert "IMAGE-LEVEL DECISION" in report
    assert "PIXEL-LEVEL LOCALIZATION SIGNAL" in report
    assert "Tampered (Predicted)" in report
    assert "Image decision threshold" in report
    assert "Localization threshold" in report
    assert "Why this prediction" in report
    assert result.localization_label in report
    assert "Raw thresholded pixels" in report
    assert "Retained highlighted pixels" in report
    assert "does not prove" in report.lower()
    assert "LIMITATIONS" in report
    assert "not proof" in report.lower() or "not calibrated" in report.lower()
    assert "fingerprint" in report


def test_chat_report_contains_scope_statement(rgb_image, fake_predictor):
    result = detect_chat_tampering(rgb_image, threshold=0.5, predictor=fake_predictor, run_ocr=False)
    report = build_chat_report(result, "fingerprint")
    assert "CHAT SCREENSHOT" in report.upper()
    assert "cannot be recovered" in report.lower() or "not evidence" in report.lower()


def test_chat_record_has_no_ocr_text(tmp_path, rgb_image, fake_predictor):
    result = detect_chat_tampering(rgb_image, threshold=0.5, predictor=fake_predictor, run_ocr=False)
    record = record_from_chat(result, "fp")
    path = tmp_path / "history.jsonl"
    append_history(record, path)
    raw = path.read_text(encoding="utf-8").lower()
    assert "ocr_lines" in raw
    assert "hello" not in raw
    assert "pixel-level signal" in raw


def test_chat_report_and_record_include_reference_summary(rgb_image, fake_predictor):
    reference = rgb_image.copy()
    suspect = rgb_image.copy()
    suspect[40:120, 80:180] = (10, 200, 240)

    result = detect_chat_tampering(
        suspect,
        threshold=0.5,
        predictor=fake_predictor,
        run_ocr=False,
        reference_rgb=reference,
    )
    assert result.reference_comparison is not None

    record = record_from_chat(result, "fp")
    assert record.extra["reference_supplied"] is True
    assert record.extra["reference_changed_regions"] >= 1
    assert record.extra["reference_finding"] == result.reference_comparison.finding

    report = build_chat_report(result, "fingerprint")
    assert "REFERENCE COMPARISON" in report
    assert "Recognised text is intentionally omitted" in report
