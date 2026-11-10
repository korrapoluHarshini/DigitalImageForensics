"""Module 1 - image manipulation detection and pixel-level localization."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from utils.config import settings
from utils.image_utils import ensure_rgb_uint8

from .model import MvssNetPredictor, get_default_predictor
from .visualization import build_manipulation_visuals


FINDING_MANIPULATED = "possible_manipulation"
FINDING_NO_EVIDENCE = "no_strong_evidence"

IMAGE_DECISION_AUTHENTIC = "authentic"
IMAGE_DECISION_TAMPERED = "tampered"
IMAGE_DECISION_UNCERTAIN = "uncertain"

IMAGE_DECISION_LABELS = {
    IMAGE_DECISION_AUTHENTIC: "Authentic (Predicted)",
    IMAGE_DECISION_TAMPERED: "Tampered (Predicted)",
    IMAGE_DECISION_UNCERTAIN: "Uncertain (Insufficient Evidence)",
}

LOCALIZATION_LABELS = {
    FINDING_MANIPULATED: "Pixel-level signal: Possible manipulation",
    FINDING_NO_EVIDENCE: "Pixel-level signal: No strong evidence",
}


def classify_image_score(
    image_score: float,
    image_threshold: float,
    uncertainty_margin: float,
) -> tuple[str, float]:
    """Map the image-level score to an Authentic/Tampered/Uncertain decision.

    The image-level score is the maximum MVSS-Net pixel probability, a number
    that already exists for localization.  Classification is a thresholded
    version of that same score; no separate classifier is trained.

    Returns the decision and the signed margin (``score - threshold``).
    Values inside ``threshold +/- uncertainty_margin`` are reported as
    uncertain rather than forced into one class.
    """
    score = float(image_score)
    threshold = float(image_threshold)
    margin = score - threshold
    band = max(0.0, float(uncertainty_margin))
    if abs(margin) <= band:
        decision = IMAGE_DECISION_UNCERTAIN
    elif margin > 0:
        decision = IMAGE_DECISION_TAMPERED
    else:
        decision = IMAGE_DECISION_AUTHENTIC
    return decision, margin


@dataclass
class SuspiciousRegion:
    region_id: int
    bbox: tuple[int, int, int, int]  # x, y, width, height (pixels)
    area_px: int
    area_ratio: float
    mean_score: float
    max_score: float

    def as_dict(self) -> dict:
        return {
            "region": self.region_id,
            "x": self.bbox[0],
            "y": self.bbox[1],
            "width": self.bbox[2],
            "height": self.bbox[3],
            "area_px": self.area_px,
            "area_percent": round(self.area_ratio * 100, 3),
            "mean_score": round(self.mean_score, 4),
            "max_score": round(self.max_score, 4),
        }


@dataclass
class ManipulationResult:
    status: str
    finding: str
    image_score: float
    threshold: float  # pixel-level localization threshold (legacy name)
    confidence: float
    image_size: tuple[int, int]
    elapsed_seconds: float
    mode: str = "single-scale"
    image_threshold: float = 0.0
    image_uncertainty_margin: float = 0.0
    image_decision: str = IMAGE_DECISION_UNCERTAIN
    image_decision_label: str = IMAGE_DECISION_LABELS[IMAGE_DECISION_UNCERTAIN]
    image_decision_margin: float = 0.0
    decision_explanation: str = ""
    suspicious_area_ratio: float = 0.0
    raw_suspicious_area_ratio: float = 0.0
    raw_suspicious_pixels: int = 0
    processed_suspicious_pixels: int = 0
    total_pixels: int = 0
    raw_component_count: int = 0
    dropped_component_count: int = 0
    regions: list[SuspiciousRegion] = field(default_factory=list)
    probability_map: np.ndarray | None = None
    binary_mask: np.ndarray | None = None
    heatmap: np.ndarray | None = None
    mask_visual: np.ndarray | None = None
    overlay: np.ndarray | None = None
    annotated: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    model_info: dict[str, Any] = field(default_factory=dict)
    inference_meta: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    @property
    def localization_threshold(self) -> float:
        """Pixel-level threshold used to build the binary mask."""
        return self.threshold

    @property
    def processed_suspicious_area_ratio(self) -> float:
        """Area of the retained, highlighted regions (after component filtering)."""
        return self.suspicious_area_ratio

    @property
    def localization_label(self) -> str:
        """Human-readable pixel-level signal, distinct from the image decision."""
        return LOCALIZATION_LABELS.get(
            self.finding, self.finding.replace("_", " ").title()
        )

    @property
    def uncertain(self) -> bool:
        """True when the score falls inside the uncertainty band."""
        return self.image_decision == IMAGE_DECISION_UNCERTAIN

    def to_summary(self) -> dict:
        return {
            "status": self.status,
            "finding": self.finding,
            "localization_label": self.localization_label,
            "image_score": round(self.image_score, 4),
            "threshold": round(self.threshold, 4),
            "localization_threshold": round(self.threshold, 4),
            "image_threshold": round(self.image_threshold, 4),
            "image_uncertainty_margin": round(self.image_uncertainty_margin, 4),
            "image_decision": self.image_decision,
            "image_decision_label": self.image_decision_label,
            "image_decision_margin": round(self.image_decision_margin, 4),
            "decision_explanation": self.decision_explanation,
            "uncertain": self.uncertain,
            "confidence": round(self.confidence, 4),
            "image_size": list(self.image_size),
            "mode": self.mode,
            "suspicious_area_percent": round(self.suspicious_area_ratio * 100, 8),
            "processed_suspicious_area_percent": round(
                self.processed_suspicious_area_ratio * 100, 8
            ),
            "raw_suspicious_area_percent": round(
                self.raw_suspicious_area_ratio * 100, 8
            ),
            "raw_suspicious_pixels": self.raw_suspicious_pixels,
            "processed_suspicious_pixels": self.processed_suspicious_pixels,
            "total_pixels": self.total_pixels,
            "raw_component_count": self.raw_component_count,
            "dropped_component_count": self.dropped_component_count,
            "regions": [region.as_dict() for region in self.regions],
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "model": dict(self.model_info),
            "inference": dict(self.inference_meta),
            "error": self.error_message,
        }


def _extract_regions(
    probability_map: np.ndarray,
    binary_mask: np.ndarray,
    min_region_area_ratio: float,
    max_regions: int,
) -> tuple[list[SuspiciousRegion], np.ndarray]:
    """Connected-component regions plus a cleaned mask containing those regions."""
    height, width = binary_mask.shape[:2]
    total_pixels = height * width
    min_pixels = max(16, int(round(total_pixels * max(0.0, min_region_area_ratio))))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary_mask > 0).astype(np.uint8), connectivity=8
    )
    candidates: list[tuple[int, SuspiciousRegion]] = []
    for label in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[label])
        if area < min_pixels:
            continue
        component = labels == label
        scores = probability_map[component]
        candidates.append(
            (
                label,
                SuspiciousRegion(
                    region_id=0,
                    bbox=(x, y, w, h),
                    area_px=area,
                    area_ratio=area / float(total_pixels),
                    mean_score=float(scores.mean()) if scores.size else 0.0,
                    max_score=float(scores.max()) if scores.size else 0.0,
                ),
            )
        )

    candidates.sort(key=lambda item: (item[1].area_px, item[1].max_score), reverse=True)
    candidates = candidates[: max(1, max_regions)]

    cleaned = np.zeros_like(binary_mask)
    regions: list[SuspiciousRegion] = []
    for index, (label, region) in enumerate(candidates, start=1):
        region.region_id = index
        regions.append(region)
        cleaned[labels == label] = 255

    return regions, cleaned


def _mask_diagnostics(
    raw_mask: np.ndarray,
    processed_mask: np.ndarray,
    regions: list[SuspiciousRegion],
    min_region_area_ratio: float,
) -> dict:
    """Describe raw thresholded pixels and what survived post-processing."""
    total_pixels = int(raw_mask.size)
    raw_pixels = int(np.count_nonzero(raw_mask))
    processed_pixels = int(np.count_nonzero(processed_mask))
    min_pixels = max(
        16, int(round(total_pixels * max(0.0, min_region_area_ratio)))
    )
    component_count = int(
        cv2.connectedComponentsWithStats(
            (raw_mask > 0).astype(np.uint8), connectivity=8
        )[0]
    )
    raw_components = max(0, component_count - 1)
    return {
        "raw_suspicious_pixels": raw_pixels,
        "processed_suspicious_pixels": processed_pixels,
        "total_pixels": total_pixels,
        "raw_suspicious_area_ratio": (raw_pixels / float(total_pixels)) if total_pixels else 0.0,
        "processed_suspicious_area_ratio": (
            processed_pixels / float(total_pixels)
        )
        if total_pixels
        else 0.0,
        "raw_component_count": raw_components,
        "dropped_component_count": max(0, raw_components - len(regions)),
        "min_region_pixels": min_pixels,
    }


def build_decision_explanation(
    image_score: float,
    image_threshold: float,
    uncertainty_margin: float,
    image_decision: str,
    localization_threshold: float,
    raw_suspicious_pixels: int,
    processed_suspicious_pixels: int,
    raw_area_ratio: float,
    processed_area_ratio: float,
    regions: list[SuspiciousRegion],
) -> str:
    """Plain-language reason for the image-level prediction and pixel signal."""
    distance = image_score - image_threshold
    direction = "above" if distance >= 0 else "below"
    band_low = max(0.0, image_threshold - uncertainty_margin)
    band_high = min(1.0, image_threshold + uncertainty_margin)
    lead = (
        f"The maximum pixel probability is {image_score:.4f}, which is "
        f"{abs(distance):.4f} {direction} the image-level threshold "
        f"({image_threshold:.2f})"
    )
    if image_decision == IMAGE_DECISION_TAMPERED:
        lead += f" and outside the +/-{uncertainty_margin:.2f} uncertainty band."
        decision_sentence = "The image-level prediction is Tampered."
    elif image_decision == IMAGE_DECISION_AUTHENTIC:
        lead += f" and outside the +/-{uncertainty_margin:.2f} uncertainty band."
        decision_sentence = "The image-level prediction is Authentic."
    else:
        lead += (
            f" but inside the +/-{uncertainty_margin:.2f} uncertainty band "
            f"({band_low:.2f}-{band_high:.2f})."
        )
        decision_sentence = (
            "The image-level prediction is Uncertain because the evidence is not "
            "separated enough from the decision boundary to assert Authentic or Tampered."
        )

    if raw_suspicious_pixels == 0:
        pixel_sentence = (
            f"No pixel reached the localization threshold "
            f"({localization_threshold:.2f}), so no suspicious area is highlighted."
        )
    elif processed_suspicious_pixels == 0:
        pixel_sentence = (
            f"{raw_suspicious_pixels:,} pixel(s) ({raw_area_ratio * 100:.6f}% of the "
            f"image) reached the localization threshold ({localization_threshold:.2f}), "
            "but no connected component passed the minimum-area filter. The highlighted "
            "mask is therefore empty; the raw pixel signal is reported for transparency."
        )
    else:
        pixel_sentence = (
            f"{len(regions)} connected region(s) at the localization threshold "
            f"({localization_threshold:.2f}) were retained, covering "
            f"{processed_suspicious_pixels:,} pixel(s) "
            f"({processed_area_ratio * 100:.6f}% of the image). The raw thresholded "
            f"mask covered {raw_suspicious_pixels:,} pixel(s) "
            f"({raw_area_ratio * 100:.6f}%)."
        )

    caveat = (
        "The image-level label thresholds the same probability map; it is not a "
        "separate calibrated classifier. A pixel-level signal alone does not prove "
        "that the whole image was manipulated."
    )
    return " ".join([lead, decision_sentence, pixel_sentence, caveat])


def _build_warnings(
    image_score: float,
    image_threshold: float,
    localization_threshold: float,
    uncertainty_margin: float,
    image_decision: str,
    raw_area_ratio: float,
    raw_suspicious_pixels: int,
    processed_suspicious_pixels: int,
    min_region_pixels: int,
    mode: str,
    regions: list[SuspiciousRegion],
) -> list[str]:
    warnings = [
        "The image-level decision reuses the maximum pixel probability from "
        "MVSS-Net; the architecture has no separate calibrated authenticity classifier.",
        f"The image-level threshold ({image_threshold:.2f}) is separate from the "
        f"pixel localization threshold ({localization_threshold:.2f}). A pixel can "
        "be highlighted without the image being classified as tampered, and vice versa.",
    ]
    if image_decision == IMAGE_DECISION_UNCERTAIN:
        warnings.append(
            f"The score is inside the uncertainty band "
            f"({image_threshold - uncertainty_margin:.2f} to "
            f"{image_threshold + uncertainty_margin:.2f}). The app does not assert "
            "Authentic or Tampered for this image."
        )
    elif image_decision == IMAGE_DECISION_AUTHENTIC:
        warnings.append(
            "The score is below the image-level threshold. This is a negative "
            "result, not proof that the image is authentic."
        )
    if image_score >= image_threshold and raw_area_ratio < 0.001:
        warnings.append(
            "Only a very small fraction of pixels carries the image-level evidence. "
            "Treat this as weak evidence."
        )
    if raw_suspicious_pixels > 0 and processed_suspicious_pixels == 0:
        warnings.append(
            f"{raw_suspicious_pixels:,} pixel(s) crossed the localization threshold, "
            f"but every connected component was smaller than the {min_region_pixels:,}-"
            "pixel minimum-area filter. The highlighted mask is empty; this does not "
            "mean that the model found no pixel-level signal."
        )
    if mode == "tiled":
        warnings.append(
            "Tiled inference blends overlapping patches and can differ from the "
            "official single-scale evaluation."
        )
    if not regions and raw_suspicious_pixels > 0:
        warnings.append(
            "No connected region survived the minimum-area filter, so no area is highlighted."
        )
    return warnings


def _base_limitations() -> list[str]:
    return [
        "MVSS-Net was trained on CASIA2.0 photographs; performance on other "
        "cameras, social-media compression or synthetic imagery can drop sharply.",
        "The predicted mask localizes statistical anomalies, not a proven edit.",
        "A negative (Authentic) result does not prove that an image is authentic.",
        "The image-level threshold was selected on the validation split; the test "
        "split was only used afterwards to report performance.",
        "Scores are not calibrated probabilities and should not be presented as certainty.",
        "Both thresholds are fixed from validation data and are not adapted per image.",
    ]


def detect_image_manipulation(
    image_rgb: np.ndarray,
    *,
    threshold: float | None = None,
    image_threshold: float | None = None,
    image_uncertainty_margin: float | None = None,
    predictor: MvssNetPredictor | None = None,
    tiled: bool = False,
    max_tiles: int = 16,
    min_region_area_ratio: float = 0.0002,
    max_regions: int = 25,
) -> ManipulationResult:
    """Analyse an RGB image for manipulation evidence and localize it.

    Parameters
    ----------
    image_rgb:
        HxWx3 uint8 RGB image.
    threshold:
        Pixel-level localization threshold. Defaults to the validation-selected
        value (0.4). This controls the binary mask and highlighted overlay only.
    image_threshold:
        Image-level Authentic/Tampered decision threshold applied to the maximum
        pixel probability. Defaults to the validation-selected value.
    image_uncertainty_margin:
        Half-width of the uncertainty band around ``image_threshold``. Scores
        inside the band are reported as uncertain.
    predictor:
        Injectable MVSS-Net wrapper (used by tests and the Streamlit cache).
    tiled:
        Blend overlapping 512x512 patches instead of resizing the whole image.
    """
    started = time.perf_counter()
    applied_threshold = float(settings.manipulation_threshold if threshold is None else threshold)
    applied_threshold = min(max(applied_threshold, 0.01), 0.99)
    applied_image_threshold = float(
        settings.image_level_threshold if image_threshold is None else image_threshold
    )
    applied_image_threshold = min(max(applied_image_threshold, 0.01), 0.99)
    applied_margin = float(
        settings.image_uncertainty_margin
        if image_uncertainty_margin is None
        else image_uncertainty_margin
    )
    applied_margin = min(max(applied_margin, 0.0), 0.49)

    try:
        rgb = ensure_rgb_uint8(image_rgb)
    except Exception as exc:
        return ManipulationResult(
            status="error",
            finding=FINDING_NO_EVIDENCE,
            image_score=0.0,
            threshold=applied_threshold,
            confidence=0.0,
            image_size=(0, 0),
            elapsed_seconds=time.perf_counter() - started,
            image_threshold=applied_image_threshold,
            image_uncertainty_margin=applied_margin,
            image_decision=IMAGE_DECISION_UNCERTAIN,
            image_decision_label=IMAGE_DECISION_LABELS[IMAGE_DECISION_UNCERTAIN],
            error_message=f"Invalid image: {exc}",
            warnings=["The uploaded image could not be prepared for analysis."],
            limitations=_base_limitations(),
        )

    height, width = rgb.shape[:2]
    active_predictor = predictor or get_default_predictor()

    try:
        probability_map, inference_meta = active_predictor.predict_probability(
            rgb, tiled=tiled, max_tiles=max_tiles
        )
    except Exception as exc:
        return ManipulationResult(
            status="error",
            finding=FINDING_NO_EVIDENCE,
            image_score=0.0,
            threshold=applied_threshold,
            confidence=0.0,
            image_size=(width, height),
            elapsed_seconds=time.perf_counter() - started,
            image_threshold=applied_image_threshold,
            image_uncertainty_margin=applied_margin,
            image_decision=IMAGE_DECISION_UNCERTAIN,
            image_decision_label=IMAGE_DECISION_LABELS[IMAGE_DECISION_UNCERTAIN],
            error_message=str(exc),
            warnings=[
                "Module 1 could not run. Other modules remain available.",
                "Check that ckpt/mvssnet_casia.pt exists and that PyTorch is installed.",
            ],
            limitations=_base_limitations(),
            model_info=active_predictor.describe() if hasattr(active_predictor, "describe") else {},
        )

    raw_mask = (probability_map >= applied_threshold).astype(np.uint8) * 255
    regions, binary_mask = _extract_regions(
        probability_map,
        raw_mask,
        min_region_area_ratio=min_region_area_ratio,
        max_regions=max_regions,
    )
    image_score = float(probability_map.max()) if probability_map.size else 0.0
    diagnostics = _mask_diagnostics(
        raw_mask, binary_mask, regions, min_region_area_ratio
    )
    area_ratio = float(diagnostics["processed_suspicious_area_ratio"])
    raw_area_ratio = float(diagnostics["raw_suspicious_area_ratio"])
    finding = FINDING_MANIPULATED if image_score >= applied_threshold else FINDING_NO_EVIDENCE
    image_decision, decision_margin = classify_image_score(
        image_score, applied_image_threshold, applied_margin
    )
    explanation = build_decision_explanation(
        image_score,
        applied_image_threshold,
        applied_margin,
        image_decision,
        applied_threshold,
        int(diagnostics["raw_suspicious_pixels"]),
        int(diagnostics["processed_suspicious_pixels"]),
        raw_area_ratio,
        area_ratio,
        regions,
    )

    visuals = build_manipulation_visuals(rgb, probability_map, binary_mask, regions)
    model_info = active_predictor.describe() if hasattr(active_predictor, "describe") else {}
    mode = "tiled" if tiled else "single-scale"

    return ManipulationResult(
        status="ok",
        finding=finding,
        image_score=image_score,
        threshold=applied_threshold,
        confidence=image_score,
        image_size=(width, height),
        elapsed_seconds=time.perf_counter() - started,
        mode=mode,
        image_threshold=applied_image_threshold,
        image_uncertainty_margin=applied_margin,
        image_decision=image_decision,
        image_decision_label=IMAGE_DECISION_LABELS[image_decision],
        image_decision_margin=decision_margin,
        decision_explanation=explanation,
        suspicious_area_ratio=area_ratio,
        raw_suspicious_area_ratio=raw_area_ratio,
        raw_suspicious_pixels=int(diagnostics["raw_suspicious_pixels"]),
        processed_suspicious_pixels=int(diagnostics["processed_suspicious_pixels"]),
        total_pixels=int(diagnostics["total_pixels"]),
        raw_component_count=int(diagnostics["raw_component_count"]),
        dropped_component_count=int(diagnostics["dropped_component_count"]),
        regions=regions,
        probability_map=probability_map,
        binary_mask=binary_mask,
        heatmap=visuals["heatmap"],
        mask_visual=visuals["mask"],
        overlay=visuals["overlay"],
        annotated=visuals["annotated"],
        warnings=_build_warnings(
            image_score,
            applied_image_threshold,
            applied_threshold,
            applied_margin,
            image_decision,
            raw_area_ratio,
            int(diagnostics["raw_suspicious_pixels"]),
            int(diagnostics["processed_suspicious_pixels"]),
            int(diagnostics["min_region_pixels"]),
            mode,
            regions,
        ),
        limitations=_base_limitations(),
        model_info=model_info,
        inference_meta=inference_meta,
    )
