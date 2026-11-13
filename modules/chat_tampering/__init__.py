"""Module 3 - chat screenshot tampering analysis with OCR and reference comparison."""

from .inference import ChatTamperingResult, RegionTextMatch, detect_chat_tampering
from .ocr import OCRBlock, OCRLine, OCRResult, extract_text, ocr_status
from .reference import (
    FINDING_DIFFERENCES,
    FINDING_MISMATCH,
    FINDING_NO_DIFFERENCES,
    ReferenceComparison,
    align_reference,
    compare_chat_screenshot,
    detect_text_changes,
)

__all__ = [
    "detect_chat_tampering",
    "ChatTamperingResult",
    "RegionTextMatch",
    "OCRResult",
    "OCRLine",
    "OCRBlock",
    "extract_text",
    "ocr_status",
    "compare_chat_screenshot",
    "ReferenceComparison",
    "align_reference",
    "detect_text_changes",
    "FINDING_DIFFERENCES",
    "FINDING_NO_DIFFERENCES",
    "FINDING_MISMATCH",
]
