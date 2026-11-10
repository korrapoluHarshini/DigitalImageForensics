"""Module 3 - chat screenshot tampering analysis with OCR."""

from .inference import ChatTamperingResult, RegionTextMatch, detect_chat_tampering
from .ocr import OCRBlock, OCRLine, OCRResult, extract_text, ocr_status

__all__ = [
    "detect_chat_tampering",
    "ChatTamperingResult",
    "RegionTextMatch",
    "OCRResult",
    "OCRLine",
    "OCRBlock",
    "extract_text",
    "ocr_status",
]
