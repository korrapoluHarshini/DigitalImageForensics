"""Module 3 - chat screenshot tampering analysis.

This pipeline deliberately reuses Module 1's forensic localization model but
keeps its own preprocessing, OCR, region/text association and reporting.  It
does **not** verify messages against original chat records, and OCR output is
never treated as proof of manipulation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modules.image_manipulation.inference import ManipulationResult, detect_image_manipulation
from modules.image_manipulation.model import MvssNetPredictor
from utils.config import settings
from utils.image_utils import ensure_rgb_uint8

from .ocr import OCRResult, extract_text
from .preprocessing import prepare_chat_image
from .reference import ReferenceComparison, compare_chat_screenshot
from .visualization import combine_forensic_and_ocr, draw_ocr_blocks


FINDING_POSSIBLE = "possible_visual_manipulation"
FINDING_NONE = "no_strong_visual_evidence"


@dataclass
class RegionTextMatch:
    region_id: int
    block_ids: list[int]
    overlap_ratio: float

    def as_dict(self) -> dict:
        return {
            "region": self.region_id,
            "ocr_blocks": list(self.block_ids),
            "overlap_ratio": round(self.overlap_ratio, 4),
        }


@dataclass
class ChatTamperingResult:
    status: str  # ok | partial | error
    finding: str
    visual_score: float
    threshold: float
    image_size: tuple[int, int]
    elapsed_seconds: float
    manipulation: ManipulationResult | None = None
    ocr: OCRResult | None = None
    regions: list = field(default_factory=list)
    region_matches: list[RegionTextMatch] = field(default_factory=list)
    heatmap: np.ndarray | None = None
    mask_visual: np.ndarray | None = None
    overlay: np.ndarray | None = None
    annotated: np.ndarray | None = None
    ocr_visual: np.ndarray | None = None
    combined_visual: np.ndarray | None = None
    reference_comparison: ReferenceComparison | None = None
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    error_message: str | None = None

    def to_summary(self) -> dict:
        return {
            "status": self.status,
            "finding": self.finding,
            "visual_score": round(self.visual_score, 4),
            "threshold": round(self.threshold, 4),
            "image_size": list(self.image_size),
            "regions": [region.as_dict() for region in self.regions],
            "region_text_matches": [match.as_dict() for match in self.region_matches],
            "ocr": self.ocr.as_dict(include_text=False) if self.ocr else None,
            "reference_comparison": (
                self.reference_comparison.as_dict(include_text=False)
                if self.reference_comparison
                else None
            ),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "error": self.error_message,
        }


def _intersection_over_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area_first = max(1, (first[2] - first[0]) * (first[3] - first[1]))
    area_second = max(1, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / float(min(area_first, area_second))


def _associate_regions_with_text(regions, blocks) -> list[RegionTextMatch]:
    matches: list[RegionTextMatch] = []
    for region in regions:
        x, y, w, h = region.bbox
        region_box = (x, y, x + w, y + h)
        overlapping: list[int] = []
        best_ratio = 0.0
        for block in blocks:
            ratio = _intersection_over_area(region_box, block.bbox)
            if ratio >= 0.05:
                overlapping.append(block.block_id)
                best_ratio = max(best_ratio, ratio)
        if overlapping:
            matches.append(
                RegionTextMatch(
                    region_id=region.region_id,
                    block_ids=sorted(set(overlapping)),
                    overlap_ratio=best_ratio,
                )
            )
    return matches


def _chat_limitations() -> list[str]:
    return [
        "MVSS-Net was trained on CASIA2.0 photographs, not on messaging-app screenshots; "
        "visual scores are out-of-domain screening signals.",
        "On the controlled synthetic evaluation set, visual screening flagged every "
        "authentic screenshot (false-positive rate 1.0) and localisation IoU was 0.0. "
        "Do not treat a highlighted region as evidence of tampering.",
        "Chat apps re-render text with subpixel antialiasing and compression, which can "
        "produce false positives or hide real edits.",
        "The analysis cannot recover the original message or compare it with a trusted "
        "chat record.",
        "Without a trusted original screenshot the app cannot verify any message, amount "
        "or date; use the reference comparison for message-level differences.",
        "OCR text is shown for readability only; unusual wording is never treated as tampering.",
        "Regions overlapping text indicate co-location, not proof that the text was edited.",
        "Stickers, emoji, images and non-Latin scripts are outside the validated scope.",
    ]


def detect_chat_tampering(
    image_rgb: np.ndarray,
    *,
    threshold: float | None = None,
    detailed: bool = False,
    max_tiles: int = 12,
    run_ocr: bool = True,
    predictor: MvssNetPredictor | None = None,
    reference_rgb: np.ndarray | None = None,
) -> ChatTamperingResult:
    """Screen a chat screenshot for visual anomalies and extract visible text.

    Parameters
    ----------
    detailed:
        Use overlapping-patch inference.  Slower on CPU but preserves small UI
        details better than squashing the screenshot to 512x512.
    run_ocr:
        When False, only visual analysis runs.
    reference_rgb:
        Optional trusted original screenshot.  When supplied, the two images are
        aligned and visible pixel/text differences are reported.  Without it no
        message-level claim is made.
    """
    started = time.perf_counter()
    applied_threshold = float(settings.manipulation_threshold if threshold is None else threshold)

    try:
        rgb = ensure_rgb_uint8(image_rgb)
    except Exception as exc:
        return ChatTamperingResult(
            status="error",
            finding=FINDING_NONE,
            visual_score=0.0,
            threshold=applied_threshold,
            image_size=(0, 0),
            elapsed_seconds=time.perf_counter() - started,
            error_message=f"Invalid image: {exc}",
            limitations=_chat_limitations(),
        )

    prepared, _scale = prepare_chat_image(rgb, max_side=settings.max_analysis_side)
    height, width = prepared.shape[:2]

    manipulation = detect_image_manipulation(
        prepared,
        threshold=applied_threshold,
        predictor=predictor,
        tiled=detailed,
        max_tiles=max_tiles,
    )

    ocr = extract_text(prepared) if run_ocr else None

    warnings: list[str] = [
        "This is a visual screening result for a chat screenshot, not message-level verification.",
    ]
    if reference_rgb is None:
        warnings.append("No original chat record or reference screenshot was used for comparison.")
    if manipulation.status == "ok":
        warnings.extend(manipulation.warnings)
    else:
        warnings.append("Visual localization failed; only OCR information may be available.")

    ocr_visual = None
    combined_visual = None
    region_matches: list[RegionTextMatch] = []
    if ocr is not None and ocr.available:
        ocr_visual = draw_ocr_blocks(prepared, ocr.blocks)
        if manipulation.status == "ok":
            region_matches = _associate_regions_with_text(manipulation.regions, ocr.blocks)
            combined_visual = combine_forensic_and_ocr(manipulation.annotated, ocr.blocks)
            if region_matches:
                warnings.append(
                    "Some highlighted regions overlap OCR text blocks. This co-location "
                    "is a review cue, not evidence that the text was edited."
                )
        if not ocr.lines:
            warnings.append("OCR found no readable text in the screenshot.")
    elif run_ocr:
        warnings.append(
            "OCR was unavailable or failed, so no text transcription is shown. "
            "Visual analysis was unaffected."
        )

    if manipulation.status == "ok":
        finding = FINDING_POSSIBLE if manipulation.finding == "possible_manipulation" else FINDING_NONE
        if not manipulation.regions:
            warnings.append(
                "No strong visual anomaly was localized. This does not prove the screenshot is authentic."
            )
        status = "ok"
        error_message = None
    else:
        finding = FINDING_NONE
        status = "partial" if (ocr is not None and ocr.available) else "error"
        error_message = manipulation.error_message

    reference_comparison: ReferenceComparison | None = None
    if reference_rgb is not None:
        try:
            reference_prepared, _reference_scale = prepare_chat_image(
                ensure_rgb_uint8(reference_rgb),
                max_side=settings.max_analysis_side,
            )
            reference_comparison = compare_chat_screenshot(
                prepared,
                reference_prepared,
                suspect_ocr=ocr if (ocr is not None and ocr.available) else None,
                run_ocr=run_ocr,
            )
        except Exception as exc:  # noqa: BLE001 - reference is optional
            warnings.append(f"The reference screenshot could not be compared: {exc}")
        else:
            warnings.append(
                "A reference screenshot was supplied. Reported differences are relative to "
                "that reference; they do not establish which screenshot is the original."
            )
            warnings.extend(reference_comparison.warnings)

    return ChatTamperingResult(
        status=status,
        finding=finding,
        visual_score=float(manipulation.image_score),
        threshold=applied_threshold,
        image_size=(width, height),
        elapsed_seconds=time.perf_counter() - started,
        manipulation=manipulation if manipulation.status == "ok" else None,
        ocr=ocr,
        regions=manipulation.regions if manipulation.status == "ok" else [],
        region_matches=region_matches,
        heatmap=manipulation.heatmap,
        mask_visual=manipulation.mask_visual,
        overlay=manipulation.overlay,
        annotated=manipulation.annotated,
        ocr_visual=ocr_visual,
        combined_visual=combined_visual,
        reference_comparison=reference_comparison,
        warnings=warnings,
        limitations=_chat_limitations(),
        error_message=error_message,
    )
