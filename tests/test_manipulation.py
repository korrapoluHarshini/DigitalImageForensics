from __future__ import annotations

import numpy as np
import pytest

from modules.image_manipulation.inference import (
    FINDING_MANIPULATED,
    FINDING_NO_EVIDENCE,
    IMAGE_DECISION_AUTHENTIC,
    IMAGE_DECISION_TAMPERED,
    IMAGE_DECISION_UNCERTAIN,
    _extract_regions,
    classify_image_score,
    detect_image_manipulation,
)
from modules.image_manipulation.visualization import (
    build_manipulation_visuals,
    draw_regions,
    mask_to_rgb,
    overlay_mask,
    probability_to_heatmap,
)


def test_extract_regions_finds_two_components():
    probability = np.zeros((200, 200), dtype=np.float32)
    probability[20:60, 20:60] = 0.9
    probability[100:150, 120:180] = 0.8
    mask = (probability >= 0.5).astype(np.uint8) * 255

    regions, cleaned = _extract_regions(probability, mask, min_region_area_ratio=0.0, max_regions=10)
    assert len(regions) == 2
    # Regions are ordered by area, so the larger 60x60 blob is R1 even though
    # the smaller blob has the higher peak probability.
    assert regions[0].area_px > regions[1].area_px
    assert regions[0].region_id == 1
    assert regions[0].max_score == pytest.approx(0.8)
    assert regions[1].max_score == pytest.approx(0.9)
    assert (cleaned > 0).sum() == (mask > 0).sum()


def test_extract_regions_filters_small_components():
    probability = np.zeros((200, 200), dtype=np.float32)
    probability[10:30, 10:30] = 0.9
    probability[100:102, 100:102] = 0.9  # 4 pixels
    mask = (probability >= 0.5).astype(np.uint8) * 255
    regions, cleaned = _extract_regions(probability, mask, min_region_area_ratio=0.001, max_regions=10)
    assert len(regions) == 1
    assert (cleaned > 0).sum() == 400


def test_extract_regions_respects_max_regions():
    probability = np.zeros((300, 300), dtype=np.float32)
    for index in range(8):
        y = 10 + index * 30
        probability[y : y + 20, 10:30] = 0.9
    mask = (probability >= 0.5).astype(np.uint8) * 255
    regions, _ = _extract_regions(probability, mask, min_region_area_ratio=0.0, max_regions=3)
    assert len(regions) == 3


def test_detect_manipulation_with_fake_predictor(rgb_image, fake_predictor):
    result = detect_image_manipulation(
        rgb_image,
        threshold=0.5,
        image_threshold=0.6,
        image_uncertainty_margin=0.05,
        predictor=fake_predictor,
    )
    assert result.status == "ok"
    assert result.finding == FINDING_MANIPULATED
    assert result.image_score == pytest.approx(0.9)
    assert len(result.regions) >= 1
    assert result.annotated.shape == rgb_image.shape
    assert result.heatmap.shape == rgb_image.shape
    assert result.binary_mask.shape == rgb_image.shape[:2]
    assert any("maximum pixel probability" in warning.lower() for warning in result.warnings)
    # Image-level decision and localization are separate settings.
    assert result.threshold == pytest.approx(0.5)
    assert result.localization_threshold == pytest.approx(0.5)
    assert result.image_threshold == pytest.approx(0.6)
    assert result.image_uncertainty_margin == pytest.approx(0.05)
    assert result.image_decision == IMAGE_DECISION_TAMPERED
    assert result.image_decision_label == "Tampered (Predicted)"
    assert result.image_decision_margin == pytest.approx(0.3)
    assert result.uncertain is False


def test_detect_manipulation_negative_result(rgb_image, fake_predictor):
    fake_predictor.probability = 0.2
    fake_predictor.region = False
    result = detect_image_manipulation(
        rgb_image, threshold=0.4, image_threshold=0.6, predictor=fake_predictor
    )
    assert result.status == "ok"
    assert result.finding == FINDING_NO_EVIDENCE
    assert result.regions == []
    assert result.suspicious_area_ratio == 0.0
    assert result.image_decision == IMAGE_DECISION_AUTHENTIC
    assert result.image_decision_label == "Authentic (Predicted)"


def test_detect_manipulation_threshold_above_score(rgb_image, fake_predictor):
    # The localization threshold can suppress the mask while the image-level
    # decision (a different threshold) still calls the image tampered.
    result = detect_image_manipulation(
        rgb_image, threshold=0.95, image_threshold=0.6, predictor=fake_predictor
    )
    assert result.finding == FINDING_NO_EVIDENCE
    assert result.binary_mask is not None and not result.binary_mask.any()
    assert result.image_decision == IMAGE_DECISION_TAMPERED


def test_detect_manipulation_uncertain_band(rgb_image, fake_predictor):
    fake_predictor.probability = 0.62
    result = detect_image_manipulation(
        rgb_image,
        threshold=0.4,
        image_threshold=0.6,
        image_uncertainty_margin=0.05,
        predictor=fake_predictor,
    )
    assert result.image_decision == IMAGE_DECISION_UNCERTAIN
    assert result.uncertain is True
    assert result.image_decision_label == "Uncertain (Insufficient Evidence)"
    assert any("uncertainty band" in warning.lower() for warning in result.warnings)


def test_decision_explanation_and_labels_are_consistent(rgb_image, fake_predictor):
    fake_predictor.probability = 0.5
    result = detect_image_manipulation(
        rgb_image,
        threshold=0.4,
        image_threshold=0.42,
        image_uncertainty_margin=0.13,
        predictor=fake_predictor,
    )
    # The image-level result can be uncertain while the pixel-level signal is
    # positive; the two labels must stay separate.
    assert result.image_decision == IMAGE_DECISION_UNCERTAIN
    assert result.finding == FINDING_MANIPULATED
    assert result.localization_label == "Pixel-level signal: Possible manipulation"
    assert result.image_decision_label == "Uncertain (Insufficient Evidence)"
    explanation = result.decision_explanation.lower()
    for phrase in (
        "maximum pixel probability",
        "image-level threshold",
        "uncertainty band",
        "localization threshold",
        "does not prove",
    ):
        assert phrase in explanation
    summary = result.to_summary()
    assert summary["localization_label"] == result.localization_label
    assert summary["image_decision"] == IMAGE_DECISION_UNCERTAIN
    assert summary["raw_suspicious_pixels"] == result.raw_suspicious_pixels
    assert summary["processed_suspicious_pixels"] == result.processed_suspicious_pixels


def test_raw_area_is_reported_when_filter_drops_every_component(rgb_image):
    class SinglePixelPredictor:
        def predict_probability(self, rgb_image, **_kwargs):
            height, width = rgb_image.shape[:2]
            probability = np.zeros((height, width), dtype=np.float32)
            probability[0, 0] = 0.9
            return probability, {"tiled": False, "tiles": 1, "device": "fake"}

        def describe(self):
            return {"name": "SinglePixelPredictor", "checkpoint": "none"}

    result = detect_image_manipulation(
        rgb_image,
        threshold=0.4,
        image_threshold=0.6,
        predictor=SinglePixelPredictor(),
        min_region_area_ratio=0.01,
    )
    total = rgb_image.shape[0] * rgb_image.shape[1]
    assert result.raw_suspicious_pixels == 1
    assert result.processed_suspicious_pixels == 0
    assert result.raw_suspicious_area_ratio == pytest.approx(1.0 / total)
    assert result.processed_suspicious_area_ratio == 0.0
    assert result.raw_component_count == 1
    assert result.dropped_component_count == 1
    assert result.regions == []
    assert result.binary_mask is not None and not result.binary_mask.any()
    assert "minimum-area filter" in result.decision_explanation.lower()
    assert any("crossed the localization threshold" in w for w in result.warnings)
    assert any("does not mean" in w for w in result.warnings)


def test_raw_and_processed_areas_match_when_region_is_retained(rgb_image, fake_predictor):
    result = detect_image_manipulation(
        rgb_image, threshold=0.5, predictor=fake_predictor
    )
    assert result.raw_suspicious_pixels > 0
    assert result.processed_suspicious_pixels == result.raw_suspicious_pixels
    assert result.raw_suspicious_area_ratio == pytest.approx(
        result.processed_suspicious_area_ratio
    )
    assert result.dropped_component_count == 0


def test_classify_image_score_boundaries():
    assert classify_image_score(0.2, 0.6, 0.05)[0] == IMAGE_DECISION_AUTHENTIC
    assert classify_image_score(0.9, 0.6, 0.05)[0] == IMAGE_DECISION_TAMPERED
    assert classify_image_score(0.6, 0.6, 0.05)[0] == IMAGE_DECISION_UNCERTAIN
    assert classify_image_score(0.55, 0.6, 0.05)[0] == IMAGE_DECISION_UNCERTAIN
    assert classify_image_score(0.54, 0.6, 0.05)[0] == IMAGE_DECISION_AUTHENTIC
    assert classify_image_score(0.66, 0.6, 0.05)[0] == IMAGE_DECISION_TAMPERED
    # A zero-width band still treats an exact tie as uncertain.
    assert classify_image_score(0.6, 0.6, 0.0)[0] == IMAGE_DECISION_UNCERTAIN
    decision, margin = classify_image_score(0.75, 0.6, 0.05)
    assert decision == IMAGE_DECISION_TAMPERED
    assert margin == pytest.approx(0.15)


def test_validated_image_level_defaults():
    from utils.config import settings

    assert settings.image_level_threshold == pytest.approx(0.42)
    assert settings.image_uncertainty_margin == pytest.approx(0.13)


def test_detect_manipulation_predictor_failure_is_contained(rgb_image):
    class BrokenPredictor:
        def predict_probability(self, *_args, **_kwargs):
            raise RuntimeError("simulated model failure")

        def describe(self):
            return {"name": "broken"}

    result = detect_image_manipulation(rgb_image, predictor=BrokenPredictor())
    assert result.status == "error"
    assert "simulated model failure" in result.error_message
    assert any("other modules remain available" in warning.lower() for warning in result.warnings)


def test_detect_manipulation_invalid_image_returns_error():
    result = detect_image_manipulation(np.zeros((4, 4, 3), dtype=np.uint8))
    assert result.status == "error"
    assert result.error_message is not None


def test_visualisations_have_expected_shapes(rgb_image, fake_predictor):
    result = detect_image_manipulation(rgb_image, threshold=0.5, predictor=fake_predictor)
    visuals = build_manipulation_visuals(
        rgb_image, result.probability_map, result.binary_mask, result.regions
    )
    for key in ("heatmap", "mask", "overlay", "annotated"):
        assert visuals[key].shape == rgb_image.shape
        assert visuals[key].dtype == np.uint8


def test_probability_heatmap_range():
    probability = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    heatmap = probability_to_heatmap(probability)
    assert heatmap.shape == (8, 8, 3)
    assert heatmap.dtype == np.uint8


def test_mask_to_rgb_accepts_3d_mask():
    mask = np.zeros((10, 10, 3), dtype=np.uint8)
    mask[:5] = 255
    visual = mask_to_rgb(mask)
    assert visual.shape == (10, 10, 3)
    assert visual[0, 0, 0] == 255
    assert visual[-1, -1, 0] == 0


def test_overlay_no_mask_is_copy(rgb_image):
    overlay = overlay_mask(rgb_image, np.zeros(rgb_image.shape[:2], dtype=np.uint8))
    np.testing.assert_array_equal(overlay, rgb_image)
    assert overlay is not rgb_image


def test_draw_regions_without_regions(rgb_image):
    annotated = draw_regions(rgb_image, [])
    np.testing.assert_array_equal(annotated, rgb_image)
