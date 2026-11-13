"""Reference-screenshot comparison for Module 3.

When the user supplies a second, trusted screenshot, the module can align the
two images and report *visible* pixel and text differences.  This is the only
setting in which the application may say that a message differs from a
reference: without a reference, no chat message can be verified.

What this module does **not** do
--------------------------------
* It does not establish which screenshot is the original.
* It does not prove who edited an image, when, or with which tool.
* It does not treat OCR disagreement as evidence unless the pixels also differ
  or the same text is read with high confidence on both sides.

The comparison is deliberately conservative: a large global difference or a
failed alignment is reported as ``reference_mismatch`` instead of emitting
thousands of "changed" regions.
"""

from __future__ import annotations

import difflib
import re
import time
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from utils.image_utils import ensure_rgb_uint8

from .ocr import OCRResult, extract_text

# ---------------------------------------------------------------------------
# Result vocabulary
# ---------------------------------------------------------------------------

FINDING_DIFFERENCES = "reference_differences_detected"
FINDING_NO_DIFFERENCES = "reference_no_differences_detected"
FINDING_MISMATCH = "reference_mismatch"

STATUS_OK = "ok"
STATUS_MISMATCH = "reference_mismatch"
STATUS_INVALID = "reference_invalid"

CHANGE_AMOUNT = "amount_or_number"
CHANGE_DATE = "date_or_timestamp"
CHANGE_TIME = "time"
CHANGE_NAME = "name_or_header"
CHANGE_MESSAGE = "message_text"

CHANGE_KINDS = (
    CHANGE_AMOUNT,
    CHANGE_DATE,
    CHANGE_TIME,
    CHANGE_NAME,
    CHANGE_MESSAGE,
)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


@dataclass
class AlignmentResult:
    """Outcome of registering the reference screenshot to the suspect one."""

    method: str
    correlation: float
    aligned: bool
    resized_reference: bool
    scale: float
    reference_size: tuple[int, int]
    suspect_size: tuple[int, int]
    reason: str | None = None
    translation: tuple[float, float] = (0.0, 0.0)

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "correlation": round(float(self.correlation), 4),
            "aligned": bool(self.aligned),
            "resized_reference": bool(self.resized_reference),
            "scale": round(float(self.scale), 4),
            "reference_size": list(self.reference_size),
            "suspect_size": list(self.suspect_size),
            "translation": [round(float(value), 2) for value in self.translation],
            "reason": self.reason,
        }


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2GRAY)


def _rescale_warp(warp: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """Scale an affine warp between downscaled and full-resolution images.

    ECC is run on a downscaled pair for speed.  Rotation/scale terms are
    dimensionless and stay unchanged; the translation is in pixels and must be
    divided by the downscale factor.
    """

    scaled = np.array(warp, dtype=np.float32, copy=True)
    scaled[0, 2] *= float(scale_x)
    scaled[1, 2] *= float(scale_y)
    return scaled


def align_reference(
    reference_rgb: np.ndarray,
    suspect_rgb: np.ndarray,
    *,
    max_alignment_side: int = 1200,
    min_correlation: float = 0.10,
) -> tuple[np.ndarray, AlignmentResult]:
    """Resize and register ``reference_rgb`` onto the suspect screenshot.

    Returns the aligned reference and an :class:`AlignmentResult`.  A failed or
    weak registration falls back to a plain resize with ``aligned=False``; the
    caller must then treat differences cautiously.
    """

    reference = ensure_rgb_uint8(reference_rgb)
    suspect = ensure_rgb_uint8(suspect_rgb)
    ref_h, ref_w = reference.shape[:2]
    sus_h, sus_w = suspect.shape[:2]
    if ref_h == 0 or ref_w == 0 or sus_h == 0 or sus_w == 0:
        raise ValueError("Reference and suspect images must be non-empty.")

    resized = (ref_w, ref_h) != (sus_w, sus_h)
    base = reference
    if resized:
        base = cv2.resize(reference, (sus_w, sus_h), interpolation=cv2.INTER_AREA)

    longest = max(sus_w, sus_h)
    scale = 1.0 if longest <= max_alignment_side else max_alignment_side / float(longest)
    small_w = max(8, int(round(sus_w * scale)))
    small_h = max(8, int(round(sus_h * scale)))

    gray_suspect = _gray(suspect)
    gray_reference = _gray(base)
    if (small_w, small_h) != (sus_w, sus_h):
        gray_suspect = cv2.resize(gray_suspect, (small_w, small_h), interpolation=cv2.INTER_AREA)
        gray_reference = cv2.resize(gray_reference, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Local contrast normalisation keeps the registration usable across light
    # and dark chat themes without changing the compared pixels.
    gray_suspect = cv2.equalizeHist(gray_suspect)
    gray_reference = cv2.equalizeHist(gray_reference)

    warp_small = np.eye(2, 3, dtype=np.float32)
    correlation = 0.0
    aligned = False
    reason: str | None = None
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-5)
        correlation, warp_small = cv2.findTransformECC(
            gray_suspect,
            gray_reference,
            warp_small,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            5,
        )
        correlation = float(correlation)
        aligned = correlation >= min_correlation
        if not aligned:
            reason = (
                "Reference and suspect screenshots are only weakly correlated "
                f"(correlation {correlation:.3f}); alignment was not trusted."
            )
    except cv2.error as exc:  # ECC does not converge on unrelated images
        reason = f"Automatic alignment failed ({type(exc).__name__}); used a plain resize."
        correlation = 0.0
        warp_small = np.eye(2, 3, dtype=np.float32)

    if aligned:
        if (small_w, small_h) != (sus_w, sus_h):
            warp_full = _rescale_warp(warp_small, sus_w / float(small_w), sus_h / float(small_h))
        else:
            warp_full = warp_small
        aligned_reference = cv2.warpAffine(
            base,
            warp_full,
            (sus_w, sus_h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
    else:
        warp_full = np.eye(2, 3, dtype=np.float32)
        aligned_reference = base

    return aligned_reference, AlignmentResult(
        method="ecc_euclidean" if aligned else "resize_only",
        correlation=correlation,
        aligned=aligned,
        resized_reference=resized,
        scale=scale,
        reference_size=(ref_w, ref_h),
        suspect_size=(sus_w, sus_h),
        reason=reason,
        translation=(float(warp_full[0, 2]), float(warp_full[1, 2])),
    )


# ---------------------------------------------------------------------------
# Pixel-level change detection
# ---------------------------------------------------------------------------


@dataclass
class ChangedRegion:
    region_id: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    area_px: int
    max_difference: float
    mean_difference: float

    def as_dict(self) -> dict:
        x1, y1, x2, y2 = self.bbox
        return {
            "region": self.region_id,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "area_px": int(self.area_px),
            "max_difference": round(float(self.max_difference), 2),
            "mean_difference": round(float(self.mean_difference), 2),
        }


def compute_change_map(
    aligned_reference: np.ndarray,
    suspect_rgb: np.ndarray,
    *,
    pixel_threshold: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(magnitude, mask)`` describing per-pixel absolute differences.

    ``magnitude`` is the maximum per-channel difference after a small Gaussian
    blur (which suppresses single-pixel JPEG noise).  ``mask`` is the raw
    thresholded change mask before any morphology, so its area is measurable.
    """

    reference = ensure_rgb_uint8(aligned_reference)
    suspect = ensure_rgb_uint8(suspect_rgb)
    if reference.shape != suspect.shape:
        raise ValueError("Aligned reference and suspect must have the same shape.")

    reference_blur = cv2.GaussianBlur(reference, (3, 3), 0)
    suspect_blur = cv2.GaussianBlur(suspect, (3, 3), 0)
    difference = cv2.absdiff(reference_blur, suspect_blur)
    magnitude = difference.max(axis=2).astype(np.uint8)
    mask = (magnitude >= float(pixel_threshold)).astype(np.uint8)
    return magnitude, mask


def extract_changed_regions(
    mask: np.ndarray,
    magnitude: np.ndarray,
    *,
    min_area: int = 20,
    max_regions: int = 30,
) -> list[ChangedRegion]:
    """Group changed pixels into connected regions, largest first."""

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if not np.any(binary):
        return []
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    magnitude_values = np.asarray(magnitude)

    regions: list[ChangedRegion] = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < int(min_area):
            continue
        component = labels == label
        values = magnitude_values[component]
        regions.append(
            ChangedRegion(
                region_id=0,
                bbox=(x, y, x + width, y + height),
                area_px=area,
                max_difference=float(values.max()) if values.size else 0.0,
                mean_difference=float(values.mean()) if values.size else 0.0,
            )
        )

    regions.sort(key=lambda item: item.area_px, reverse=True)
    regions = regions[: int(max_regions)]
    for index, region in enumerate(regions, start=1):
        region.region_id = index
    return regions


# ---------------------------------------------------------------------------
# Text-level comparison
# ---------------------------------------------------------------------------

_AMOUNT_RE = re.compile(
    r"(?<![\w])(?:rs\.?|inr|usd|eur|gbp|₹|\$|€|£|¥)?\s?\d[\d,]*(?:\.\d+)?",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})\b"
)
_DAY_MONTH_RE = re.compile(
    r"\b\d{1,2}(?:st|nd|rd|th)?\.?\s+"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?,?\s+\d{2,4}\b",
    re.IGNORECASE,
)
_MONTH_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
    r"[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{2,4})?\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:[ap]\.?m\.?)?\b",
    re.IGNORECASE,
)
_DOT_TIME_RE = re.compile(r"^\s*(\d{1,2})[.](\d{2})\s*$")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'.\-]*")


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _numbers(text: str) -> list[str]:
    return [match.group(0).strip() for match in _AMOUNT_RE.finditer(text)]


def _mostly_alphabetic(text: str) -> bool:
    letters = sum(char.isalpha() for char in text)
    digits = sum(char.isdigit() for char in text)
    return letters >= 3 and letters >= digits * 2


def _location(y: int, height: int) -> str:
    if height <= 0:
        return "body"
    ratio = y / float(height)
    if ratio <= 0.15:
        return "header"
    if ratio >= 0.85:
        return "footer"
    return "body"


def _dot_time(text: str) -> str | None:
    """Recognise OCR output such as ``11.38`` as a clock time.

    Some OCR backends replace the colon with a period.  Only a standalone
    ``H.MM`` token with a valid hour/minute is accepted, so decimals inside
    longer amounts are not silently reclassified.
    """

    match = _DOT_TIME_RE.match(str(text or ""))
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def _time_tokens(text: str) -> list[str]:
    matches = [match.strip().lower() for match in _TIME_RE.findall(text)]
    if matches:
        return matches
    dot = _dot_time(text)
    return [dot] if dot else []


def _looks_like_time(text: str) -> bool:
    return bool(_time_tokens(text))


def classify_change(reference_text: str, suspect_text: str, *, location: str = "body") -> str:
    """Assign one of :data:`CHANGE_KINDS` to a detected text difference.

    Classification is heuristic and is reported to the user as a *category*,
    never as proof of what was edited.
    """

    reference = _normalise_text(reference_text)
    suspect = _normalise_text(suspect_text)

    reference_has_date = bool(
        _DATE_RE.search(reference) or _DAY_MONTH_RE.search(reference) or _MONTH_RE.search(reference)
    )
    suspect_has_date = bool(
        _DATE_RE.search(suspect) or _DAY_MONTH_RE.search(suspect) or _MONTH_RE.search(suspect)
    )
    if reference_has_date and suspect_has_date and reference != suspect:
        date_patterns = (_DATE_RE, _DAY_MONTH_RE, _MONTH_RE)
        reference_dates = [match.lower() for pattern in date_patterns for match in pattern.findall(reference)]
        suspect_dates = [match.lower() for pattern in date_patterns for match in pattern.findall(suspect)]
        if reference_dates != suspect_dates:
            return CHANGE_DATE

    reference_has_time = _looks_like_time(reference)
    suspect_has_time = _looks_like_time(suspect)
    if reference_has_time and suspect_has_time and _time_tokens(reference) != _time_tokens(suspect):
        return CHANGE_TIME

    if _numbers(reference) != _numbers(suspect):
        return CHANGE_AMOUNT

    words = _WORD_RE.findall(reference)
    if (
        location == "header"
        and len(words) <= 5
        and _mostly_alphabetic(reference)
        and _mostly_alphabetic(suspect)
    ):
        return CHANGE_NAME

    return CHANGE_MESSAGE


@dataclass
class ReferenceTextChange:
    change_id: int
    kind: str
    reference_text: str
    suspect_text: str
    reference_bbox: tuple[int, int, int, int] | None
    suspect_bbox: tuple[int, int, int, int] | None
    location: str
    confidence: float

    def as_dict(self, include_text: bool = False) -> dict:
        payload = {
            "change": self.change_id,
            "kind": self.kind,
            "location": self.location,
            "confidence": round(float(self.confidence), 4),
            "reference_bbox": list(self.reference_bbox) if self.reference_bbox else None,
            "suspect_bbox": list(self.suspect_bbox) if self.suspect_bbox else None,
        }
        if include_text:
            payload["reference_text"] = self.reference_text
            payload["suspect_text"] = self.suspect_text
        return payload


@dataclass
class _TextLine:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]


_KIND_PRIORITY = {
    CHANGE_NAME: 0,
    CHANGE_DATE: 1,
    CHANGE_TIME: 2,
    CHANGE_AMOUNT: 3,
    CHANGE_MESSAGE: 4,
}


def _union_boxes(
    boxes: Sequence[tuple[int, int, int, int] | None],
) -> tuple[int, int, int, int] | None:
    present = [box for box in boxes if box is not None]
    if not present:
        return None
    return (
        min(box[0] for box in present),
        min(box[1] for box in present),
        max(box[2] for box in present),
        max(box[3] for box in present),
    )


def _change_anchor(change: ReferenceTextChange) -> tuple[int, int, int, int] | None:
    return _union_boxes([change.reference_bbox, change.suspect_bbox])


def _merge_adjacent_changes(
    changes: Sequence[ReferenceTextChange], *, max_gap: int = 10
) -> list[ReferenceTextChange]:
    """Merge OCR-diff fragments caused by inconsistent line segmentation.

    When OCR splits one edited line into two boxes on one side only, the raw
    line diff reports one change plus one insertion.  Merging changes that sit
    on the same visual line and share a location keeps the report readable and
    stops one edit from being counted twice.
    """

    ordered = sorted(
        changes,
        key=lambda change: (
            (_change_anchor(change) or (0, 0, 0, 0))[1],
            (_change_anchor(change) or (0, 0, 0, 0))[0],
        ),
    )
    groups: list[list[ReferenceTextChange]] = []
    for change in ordered:
        if not groups:
            groups.append([change])
            continue
        current = groups[-1]
        current_box = _union_boxes([_change_anchor(item) for item in current])
        change_box = _change_anchor(change)
        same_location = current[0].location == change.location
        if current_box is None or change_box is None:
            gap = 0
        else:
            gap = max(change_box[1] - current_box[3], current_box[1] - change_box[3], 0)
        if same_location and gap <= max_gap:
            current.append(change)
        else:
            groups.append([change])

    merged: list[ReferenceTextChange] = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue
        ordered_group = sorted(
            group,
            key=lambda change: (
                (_change_anchor(change) or (0, 0, 0, 0))[1],
                (_change_anchor(change) or (0, 0, 0, 0))[0],
            ),
        )
        reference_text = " ".join(
            dict.fromkeys(item.reference_text for item in ordered_group if item.reference_text)
        )
        suspect_text = " ".join(
            dict.fromkeys(item.suspect_text for item in ordered_group if item.suspect_text)
        )
        merged.append(
            ReferenceTextChange(
                change_id=0,
                kind=classify_change(
                    reference_text, suspect_text, location=ordered_group[0].location
                ),
                reference_text=reference_text,
                suspect_text=suspect_text,
                reference_bbox=_union_boxes([item.reference_bbox for item in ordered_group]),
                suspect_bbox=_union_boxes([item.suspect_bbox for item in ordered_group]),
                location=ordered_group[0].location,
                confidence=min(item.confidence for item in ordered_group),
            )
        )
    return merged


def _text_lines(ocr: OCRResult | None) -> list[_TextLine]:
    if ocr is None or not ocr.available:
        return []
    lines: list[_TextLine] = []
    if ocr.lines:
        for line in ocr.lines:
            text = _normalise_text(line.text)
            if text:
                lines.append(_TextLine(text=text, confidence=float(line.confidence), bbox=line.bbox))
    elif ocr.blocks:
        for block in ocr.blocks:
            text = _normalise_text(block.text)
            if text:
                lines.append(
                    _TextLine(text=text, confidence=float(block.confidence), bbox=block.bbox)
                )
    lines.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    return lines


def detect_text_changes(
    reference_ocr: OCRResult | None,
    suspect_ocr: OCRResult | None,
    *,
    image_size: tuple[int, int] | None = None,
    min_confidence: float = 0.25,
    max_changes: int = 40,
) -> list[ReferenceTextChange]:
    """Diff the two OCR transcripts and describe line-level differences.

    OCR errors can appear here.  Each change therefore records both bounding
    boxes and the weaker OCR confidence, and callers must present the diff as a
    review aid rather than proof.
    """

    reference_lines = [line for line in _text_lines(reference_ocr) if line.confidence >= min_confidence]
    suspect_lines = [line for line in _text_lines(suspect_ocr) if line.confidence >= min_confidence]
    if not reference_lines and not suspect_lines:
        return []

    if image_size is None:
        height = 1
        for line in reference_lines + suspect_lines:
            height = max(height, int(line.bbox[3]))
    else:
        height = max(1, int(image_size[1]))

    matcher = difflib.SequenceMatcher(
        a=[line.text.lower() for line in reference_lines],
        b=[line.text.lower() for line in suspect_lines],
        autojunk=False,
    )

    changes: list[ReferenceTextChange] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        reference_group = reference_lines[i1:i2]
        suspect_group = suspect_lines[j1:j2]
        pairs = max(len(reference_group), len(suspect_group))
        for offset in range(pairs):
            reference_line = reference_group[offset] if offset < len(reference_group) else None
            suspect_line = suspect_group[offset] if offset < len(suspect_group) else None
            reference_text = reference_line.text if reference_line else ""
            suspect_text = suspect_line.text if suspect_line else ""
            if _normalise_text(reference_text).lower() == _normalise_text(suspect_text).lower():
                continue
            anchor = reference_line or suspect_line
            location = _location(anchor.bbox[1], height) if anchor else "body"
            confidences = [line.confidence for line in (reference_line, suspect_line) if line]
            confidence = min(confidences) if confidences else 0.0
            if reference_line is None or suspect_line is None:
                confidence *= 0.75
            changes.append(
                ReferenceTextChange(
                    change_id=0,
                    kind=classify_change(reference_text, suspect_text, location=location),
                    reference_text=reference_text,
                    suspect_text=suspect_text,
                    reference_bbox=reference_line.bbox if reference_line else None,
                    suspect_bbox=suspect_line.bbox if suspect_line else None,
                    location=location,
                    confidence=float(confidence),
                )
            )

    changes.sort(key=lambda item: (item.location != "header", item.kind, -item.confidence))
    changes = _merge_adjacent_changes(changes)
    changes.sort(key=lambda item: (item.location != "header", item.kind, -item.confidence))
    changes = changes[: int(max_changes)]
    for index, change in enumerate(changes, start=1):
        change.change_id = index
    return changes


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------


def colorize_change_map(magnitude: np.ndarray) -> np.ndarray:
    """Turn a 0..255 change magnitude into an RGB heatmap."""

    values = np.asarray(magnitude)
    if values.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if values.dtype != np.uint8:
        values = np.clip(values, 0, 255).astype(np.uint8)
    # Doubling the gain makes faint differences visible without changing the
    # underlying mask or any reported metric.
    boosted = np.clip(values.astype(np.uint16) * 2, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(boosted, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def _box(image: np.ndarray, bbox, color_bgr, label: str | None = None) -> None:
    x1, y1, x2, y2 = (int(value) for value in bbox)
    height, width = image.shape[:2]
    thickness = max(1, int(round(min(height, width) / 500)))
    cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, thickness)
    if label:
        font_scale = max(0.4, min(0.9, min(height, width) / 1400))
        text_y = max(12, y1 - 5)
        cv2.putText(
            image,
            label,
            (max(0, x1 + 2), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color_bgr,
            max(1, thickness - 1),
            cv2.LINE_AA,
        )


def build_comparison_visuals(
    aligned_reference: np.ndarray,
    suspect_rgb: np.ndarray,
    magnitude: np.ndarray,
    change_mask: np.ndarray,
    regions: Sequence[ChangedRegion],
    text_changes: Sequence[ReferenceTextChange],
) -> dict[str, np.ndarray]:
    """Build heatmap, overlay and side-by-side comparison images (RGB)."""

    reference = ensure_rgb_uint8(aligned_reference)
    suspect = ensure_rgb_uint8(suspect_rgb)
    heatmap = colorize_change_map(magnitude)

    overlay = suspect.copy()
    mask = np.asarray(change_mask) > 0
    if np.any(mask):
        tint = np.zeros_like(overlay)
        tint[..., 0] = 255
        overlay[mask] = (
            overlay[mask].astype(np.float32) * 0.55 + tint[mask].astype(np.float32) * 0.45
        ).astype(np.uint8)
    annotated_suspect = overlay.copy()
    for region in regions:
        _box(annotated_suspect, region.bbox, (255, 40, 40), f"P{region.region_id}")

    annotated_reference = reference.copy()
    for change in text_changes:
        if change.reference_bbox:
            _box(annotated_reference, change.reference_bbox, (255, 140, 0), f"D{change.change_id}")
        if change.suspect_bbox:
            _box(annotated_suspect, change.suspect_bbox, (255, 140, 0), f"D{change.change_id}")

    gap = 12
    height = max(reference.shape[0], suspect.shape[0])
    width = reference.shape[1] + suspect.shape[1] + gap
    side_by_side = np.full((height, width, 3), 255, dtype=np.uint8)
    side_by_side[: reference.shape[0], : reference.shape[1]] = annotated_reference
    offset = reference.shape[1] + gap
    side_by_side[: suspect.shape[0], offset : offset + suspect.shape[1]] = annotated_suspect

    change_mask_u8 = (np.asarray(change_mask) > 0).astype(np.uint8) * 255
    return {
        "heatmap": heatmap,
        "overlay": annotated_suspect,
        "side_by_side": side_by_side,
        "change_mask": change_mask_u8,
    }


# ---------------------------------------------------------------------------
# Top-level comparison
# ---------------------------------------------------------------------------


def _reference_limitations() -> list[str]:
    return [
        "A reference comparison shows that pixels or recognised text differ; it cannot "
        "establish which screenshot is the original, who edited it, or when.",
        "The reference must be the genuine original. Comparing against an edited copy "
        "produces misleading differences.",
        "Re-rendering, theme changes, font substitution, resizing and recompression can "
        "produce differences that are not tampering.",
        "OCR text differences may be recognition errors; always inspect the highlighted "
        "pixel regions alongside the transcripts.",
        "An aligned comparison is a strong review aid, not a forensic proof of intent.",
    ]


@dataclass
class ReferenceComparison:
    status: str
    finding: str
    elapsed_seconds: float
    alignment: AlignmentResult | None = None
    changed_pixel_ratio: float = 0.0
    changed_pixels: int = 0
    raw_pixels: int = 0
    regions: list[ChangedRegion] = field(default_factory=list)
    text_changes: list[ReferenceTextChange] = field(default_factory=list)
    reference_ocr: OCRResult | None = None
    suspect_ocr: OCRResult | None = None
    heatmap: np.ndarray | None = None
    overlay: np.ndarray | None = None
    side_by_side: np.ndarray | None = None
    change_mask: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    error_message: str | None = None

    def as_dict(self, include_text: bool = False) -> dict:
        payload = {
            "status": self.status,
            "finding": self.finding,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "alignment": self.alignment.as_dict() if self.alignment else None,
            "changed_pixel_ratio": round(float(self.changed_pixel_ratio), 6),
            "changed_pixels": int(self.changed_pixels),
            "raw_pixels": int(self.raw_pixels),
            "changed_regions": [region.as_dict() for region in self.regions],
            "text_change_count": len(self.text_changes),
            "text_changes": [
                change.as_dict(include_text=include_text) for change in self.text_changes
            ],
            "change_kinds": sorted({change.kind for change in self.text_changes}),
            "ocr": {
                "reference": self.reference_ocr.as_dict(include_text=False)
                if self.reference_ocr
                else None,
                "suspect": self.suspect_ocr.as_dict(include_text=False)
                if self.suspect_ocr
                else None,
            },
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "error": self.error_message,
        }
        return payload


def compare_chat_screenshot(
    suspect_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    *,
    pixel_threshold: float = 25.0,
    min_region_area: int = 20,
    run_ocr: bool = True,
    suspect_ocr: OCRResult | None = None,
    languages: tuple[str, ...] | None = None,
    max_regions: int = 30,
) -> ReferenceComparison:
    """Compare a suspect screenshot with a supplied reference screenshot.

    Parameters
    ----------
    suspect_rgb, reference_rgb:
        RGB ``uint8`` images.
    pixel_threshold:
        Per-pixel maximum-channel difference (0-255) required to count a pixel
        as changed.  Fixed before evaluation; never tuned on test data.
    suspect_ocr:
        Optional pre-computed OCR of the suspect, to avoid running OCR twice.
    """

    started = time.perf_counter()
    limitations = _reference_limitations()

    try:
        suspect = ensure_rgb_uint8(suspect_rgb)
        reference = ensure_rgb_uint8(reference_rgb)
    except Exception as exc:  # noqa: BLE001 - report the real decode error
        return ReferenceComparison(
            status=STATUS_INVALID,
            finding=FINDING_MISMATCH,
            elapsed_seconds=time.perf_counter() - started,
            warnings=["The reference screenshot could not be read."],
            limitations=limitations,
            error_message=f"Invalid image: {exc}",
        )

    try:
        aligned_reference, alignment = align_reference(reference, suspect)
    except Exception as exc:  # noqa: BLE001
        return ReferenceComparison(
            status=STATUS_INVALID,
            finding=FINDING_MISMATCH,
            elapsed_seconds=time.perf_counter() - started,
            warnings=["The reference screenshot could not be aligned."],
            limitations=limitations,
            error_message=f"Alignment failed: {exc}",
        )

    magnitude, mask = compute_change_map(
        aligned_reference, suspect, pixel_threshold=pixel_threshold
    )
    changed_pixels = int(mask.sum())
    raw_pixels = int(mask.size)
    changed_ratio = changed_pixels / float(max(1, raw_pixels))

    regions = extract_changed_regions(
        mask, magnitude, min_area=min_region_area, max_regions=max_regions
    )

    reference_ocr = extract_text(reference, languages=languages) if run_ocr else None
    if run_ocr and suspect_ocr is None:
        suspect_ocr = extract_text(suspect, languages=languages)
    suspect_ocr = suspect_ocr if run_ocr else None

    text_changes = detect_text_changes(
        reference_ocr,
        suspect_ocr,
        image_size=(suspect.shape[1], suspect.shape[0]),
    )

    visuals = build_comparison_visuals(
        aligned_reference, suspect, magnitude, mask, regions, text_changes
    )

    warnings = [
        "Reference comparison is only meaningful when the reference is the genuine, "
        "unedited screenshot.",
        "Differences can come from re-rendering, compression, resizing or theme changes, "
        "not only from tampering.",
    ]
    if alignment.reason:
        warnings.append(alignment.reason)
    if alignment.resized_reference:
        warnings.append(
            "The reference had a different resolution and was resized before comparison."
        )
    if text_changes:
        warnings.append(
            "OCR differences are review cues. Inspect the highlighted pixel regions and "
            "both transcripts before drawing a conclusion."
        )
    if reference_ocr is not None and not reference_ocr.lines:
        warnings.append("No readable text was recognised in the reference screenshot.")

    status = STATUS_OK
    if not alignment.aligned or changed_ratio > 0.60:
        status = STATUS_MISMATCH
        warnings.append(
            "The global difference is too large or the alignment too weak to attribute "
            "individual changes. The reference may show a different screen, theme, "
            "resolution or chat state."
        )

    if status == STATUS_MISMATCH:
        finding = FINDING_MISMATCH
    elif regions or text_changes:
        finding = FINDING_DIFFERENCES
    else:
        finding = FINDING_NO_DIFFERENCES

    return ReferenceComparison(
        status=status,
        finding=finding,
        elapsed_seconds=time.perf_counter() - started,
        alignment=alignment,
        changed_pixel_ratio=changed_ratio,
        changed_pixels=changed_pixels,
        raw_pixels=raw_pixels,
        regions=regions,
        text_changes=text_changes,
        reference_ocr=reference_ocr,
        suspect_ocr=suspect_ocr,
        heatmap=visuals["heatmap"],
        overlay=visuals["overlay"],
        side_by_side=visuals["side_by_side"],
        change_mask=visuals["change_mask"],
        warnings=warnings,
        limitations=limitations,
    )
