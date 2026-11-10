"""OCR for chat screenshots.

OCR is used for *readability only*.  It cannot establish whether a message was
edited, and recognised text is never persisted or logged by the app.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.config import settings
from utils.image_utils import ensure_rgb_uint8

from .preprocessing import prepare_for_ocr


@dataclass
class OCRLine:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    block_id: int = 0

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "block": self.block_id,
        }


@dataclass
class OCRBlock:
    block_id: int
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    lines: list[OCRLine] = field(default_factory=list)


@dataclass
class OCRResult:
    available: bool
    backend: str | None
    lines: list[OCRLine] = field(default_factory=list)
    blocks: list[OCRBlock] = field(default_factory=list)
    full_text: str = ""
    elapsed_seconds: float = 0.0
    reason: str | None = None
    warning: str | None = None

    def as_dict(self, include_text: bool = False) -> dict:
        payload: dict[str, Any] = {
            "available": self.available,
            "backend": self.backend,
            "line_count": len(self.lines),
            "block_count": len(self.blocks),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "reason": self.reason,
        }
        if include_text:
            payload["lines"] = [line.as_dict() for line in self.lines]
            payload["blocks"] = [
                {"block": block.block_id, "text": block.text, "bbox": list(block.bbox)}
                for block in self.blocks
            ]
        return payload


def ocr_status() -> dict:
    """Cheap availability probe; does not construct an OCR reader."""
    try:
        import easyocr  # noqa: F401

        return {
            "available": True,
            "backend": "easyocr",
            "reason": "EasyOCR import succeeded (first run may download model files).",
        }
    except Exception as easyocr_error:
        try:
            import pytesseract  # noqa: F401

            pytesseract.get_tesseract_version()
            return {"available": True, "backend": "tesseract", "reason": "Tesseract is available."}
        except Exception:
            return {
                "available": False,
                "backend": None,
                "reason": (
                    "No OCR backend is installed. Install easyocr "
                    "(pip install easyocr) or Tesseract + pytesseract."
                ),
            }


def _group_lines(lines: list[OCRLine], gap_factor: float = 1.8) -> list[OCRBlock]:
    """Group nearby OCR lines into visual text blocks (e.g. chat bubbles)."""
    if not lines:
        return []
    ordered = sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0]))
    heights = [max(1, line.bbox[3] - line.bbox[1]) for line in ordered]
    median_height = statistics.median(heights) if heights else 16.0
    gap_limit = max(6.0, median_height * gap_factor)

    blocks: list[list[OCRLine]] = []
    current: list[OCRLine] = [ordered[0]]
    for line in ordered[1:]:
        previous = current[-1]
        gap = line.bbox[1] - previous.bbox[3]
        if gap <= gap_limit:
            current.append(line)
        else:
            blocks.append(current)
            current = [line]
    blocks.append(current)

    result: list[OCRBlock] = []
    for block_index, block_lines in enumerate(blocks, start=1):
        for line in block_lines:
            line.block_id = block_index
        x1 = min(line.bbox[0] for line in block_lines)
        y1 = min(line.bbox[1] for line in block_lines)
        x2 = max(line.bbox[2] for line in block_lines)
        y2 = max(line.bbox[3] for line in block_lines)
        confidences = [line.confidence for line in block_lines]
        result.append(
            OCRBlock(
                block_id=block_index,
                text="\n".join(line.text for line in block_lines),
                confidence=float(sum(confidences) / len(confidences)) if confidences else 0.0,
                bbox=(x1, y1, x2, y2),
                lines=block_lines,
            )
        )
    return result


def _run_easyocr(image_rgb: np.ndarray, languages: tuple[str, ...]) -> OCRResult:
    import easyocr
    import torch

    started = time.perf_counter()
    # The reader is intentionally constructed per call only after the user asks
    # for OCR; EasyOCR caches model files on disk after the first download.
    reader = easyocr.Reader(
        list(languages or ("en",)),
        gpu=torch.cuda.is_available(),
        verbose=False,
    )
    prepared, scale = prepare_for_ocr(image_rgb)
    raw = reader.readtext(prepared, detail=1, paragraph=False)

    lines: list[OCRLine] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        points, text, confidence = item[0], str(item[1]).strip(), float(item[2])
        if not text or confidence < 0.20:
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        bbox = (
            int(round(min(xs) / scale)),
            int(round(min(ys) / scale)),
            int(round(max(xs) / scale)),
            int(round(max(ys) / scale)),
        )
        lines.append(OCRLine(text=text, confidence=confidence, bbox=bbox))

    lines.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
    blocks = _group_lines(lines)
    return OCRResult(
        available=True,
        backend="easyocr",
        lines=lines,
        blocks=blocks,
        full_text="\n".join(line.text for line in lines),
        elapsed_seconds=time.perf_counter() - started,
        warning=(
            "OCR only transcribes visible pixels. It does not verify message "
            "authenticity and may misread stylised or low-contrast text."
        ),
    )


def _run_tesseract(image_rgb: np.ndarray) -> OCRResult:
    import pytesseract
    from pytesseract import Output

    started = time.perf_counter()
    prepared, scale = prepare_for_ocr(image_rgb)
    data = pytesseract.image_to_data(
        prepared,
        output_type=Output.DICT,
        config="--psm 6",
    )

    grouped: dict[tuple[int, int, int], list[OCRLine]] = {}
    for index, text in enumerate(data.get("text", [])):
        text = str(text).strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][index])
        except (KeyError, ValueError):
            confidence = -1.0
        if confidence < 20:
            continue
        x = int(data["left"][index] / scale)
        y = int(data["top"][index] / scale)
        w = int(data["width"][index] / scale)
        h = int(data["height"][index] / scale)
        key = (
            int(data.get("block_num", [0] * len(data["text"]))[index]),
            int(data.get("par_num", [0] * len(data["text"]))[index]),
            int(data.get("line_num", [0] * len(data["text"]))[index]),
        )
        grouped.setdefault(key, []).append(
            OCRLine(text=text, confidence=confidence / 100.0, bbox=(x, y, x + w, y + h))
        )

    lines: list[OCRLine] = []
    for group_index, (_key, words) in enumerate(sorted(grouped.items()), start=1):
        words.sort(key=lambda line: line.bbox[0])
        x1 = min(line.bbox[0] for line in words)
        y1 = min(line.bbox[1] for line in words)
        x2 = max(line.bbox[2] for line in words)
        y2 = max(line.bbox[3] for line in words)
        text = " ".join(line.text for line in words)
        confidence = sum(line.confidence for line in words) / max(1, len(words))
        lines.append(OCRLine(text=text, confidence=confidence, bbox=(x1, y1, x2, y2), block_id=group_index))

    lines.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
    blocks = _group_lines(lines)
    return OCRResult(
        available=True,
        backend="tesseract",
        lines=lines,
        blocks=blocks,
        full_text="\n".join(line.text for line in lines),
        elapsed_seconds=time.perf_counter() - started,
        warning="OCR output is a transcription, not evidence of tampering.",
    )


def extract_text(
    image_rgb: np.ndarray,
    languages: tuple[str, ...] | None = None,
) -> OCRResult:
    """Extract visible text, degrading gracefully when no backend is present."""
    started = time.perf_counter()
    try:
        rgb = ensure_rgb_uint8(image_rgb)
    except Exception as exc:
        return OCRResult(
            available=False,
            backend=None,
            reason=f"Invalid image: {exc}",
            elapsed_seconds=time.perf_counter() - started,
        )

    try:
        return _run_easyocr(rgb, languages or settings.ocr_languages)
    except Exception as easyocr_error:
        try:
            return _run_tesseract(rgb)
        except Exception as tesseract_error:
            return OCRResult(
                available=False,
                backend=None,
                reason=(
                    "OCR is unavailable. EasyOCR failed "
                    f"({type(easyocr_error).__name__}) and Tesseract failed "
                    f"({type(tesseract_error).__name__})."
                ),
                elapsed_seconds=time.perf_counter() - started,
                warning="Visual analysis still ran; only text transcription is missing.",
            )
