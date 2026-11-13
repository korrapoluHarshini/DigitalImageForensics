"""Analysis history and downloadable report generation.

Privacy rules enforced here:

* uploaded screenshots are never written to disk by the app,
* OCR text is never persisted,
* history entries contain a content fingerprint and aggregate scores only,
* no absolute filesystem paths are stored.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import settings


@dataclass
class AnalysisRecord:
    timestamp: str
    module: str
    finding: str
    score: float | None
    threshold: float | None
    width: int
    height: int
    fingerprint: str
    elapsed_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def record_from_manipulation(result, fingerprint: str) -> AnalysisRecord:
    return AnalysisRecord(
        timestamp=_now_iso(),
        module="Image Manipulation",
        finding=result.finding,
        score=_round(result.image_score),
        threshold=_round(result.threshold),
        width=int(result.image_size[0]),
        height=int(result.image_size[1]),
        fingerprint=fingerprint,
        elapsed_seconds=_round(result.elapsed_seconds, 2),
        extra={
            "regions": len(result.regions),
            "suspicious_area_ratio": _round(result.suspicious_area_ratio, 8),
            "processed_suspicious_area_ratio": _round(
                result.processed_suspicious_area_ratio, 8
            ),
            "raw_suspicious_area_ratio": _round(
                result.raw_suspicious_area_ratio, 8
            ),
            "raw_suspicious_pixels": int(result.raw_suspicious_pixels),
            "processed_suspicious_pixels": int(result.processed_suspicious_pixels),
            "total_pixels": int(result.total_pixels),
            "raw_component_count": int(result.raw_component_count),
            "dropped_component_count": int(result.dropped_component_count),
            "mode": result.mode,
            "image_decision": result.image_decision,
            "image_decision_label": result.image_decision_label,
            "localization_label": result.localization_label,
            "image_threshold": _round(result.image_threshold),
            "image_uncertainty_margin": _round(result.image_uncertainty_margin),
            "image_decision_margin": _round(result.image_decision_margin),
            "localization_threshold": _round(result.threshold),
            "uncertain": bool(result.uncertain),
        },
    )


def record_from_ai(result, fingerprint: str) -> AnalysisRecord:
    return AnalysisRecord(
        timestamp=_now_iso(),
        module="AI-Generated Image",
        finding=result.status if result.status != "ok" else result.predicted_class,
        score=_round(result.ai_confidence),
        threshold=None,
        width=int(result.image_size[0]) if result.image_size else 0,
        height=int(result.image_size[1]) if result.image_size else 0,
        fingerprint=fingerprint,
        elapsed_seconds=_round(result.elapsed_seconds, 2),
        extra={"backend": result.backend or "not integrated", "uncertain": bool(result.uncertain)},
    )


def record_from_chat(result, fingerprint: str) -> AnalysisRecord:
    manipulation = result.manipulation
    return AnalysisRecord(
        timestamp=_now_iso(),
        module="Chat Screenshot",
        finding=result.finding,
        score=_round(result.visual_score),
        threshold=_round(result.threshold),
        width=int(result.image_size[0]),
        height=int(result.image_size[1]),
        fingerprint=fingerprint,
        elapsed_seconds=_round(result.elapsed_seconds, 2),
        extra={
            "regions": len(result.regions),
            "ocr_backend": result.ocr.backend if result.ocr else "unavailable",
            "ocr_lines": len(result.ocr.lines) if result.ocr else 0,
            "detailed_mode": bool(manipulation.mode == "tiled"),
            "localization_label": (
                manipulation.localization_label
                if manipulation is not None
                else {
                    "possible_visual_manipulation": (
                        "Pixel-level signal: Possible manipulation"
                    ),
                    "no_strong_visual_evidence": (
                        "Pixel-level signal: No strong evidence"
                    ),
                }.get(result.finding, result.finding.replace("_", " ").title())
            ),
            "raw_suspicious_area_ratio": (
                _round(manipulation.raw_suspicious_area_ratio, 8)
                if manipulation is not None
                else None
            ),
            "processed_suspicious_area_ratio": (
                _round(manipulation.processed_suspicious_area_ratio, 8)
                if manipulation is not None
                else None
            ),
            "reference_supplied": result.reference_comparison is not None,
            "reference_finding": (
                result.reference_comparison.finding
                if result.reference_comparison is not None
                else None
            ),
            "reference_changed_regions": (
                len(result.reference_comparison.regions)
                if result.reference_comparison is not None
                else 0
            ),
            "reference_text_changes": (
                len(result.reference_comparison.text_changes)
                if result.reference_comparison is not None
                else 0
            ),
            "reference_change_kinds": (
                sorted({change.kind for change in result.reference_comparison.text_changes})
                if result.reference_comparison is not None
                else []
            ),
            "reference_alignment_correlation": (
                _round(result.reference_comparison.alignment.correlation)
                if result.reference_comparison is not None
                and result.reference_comparison.alignment is not None
                else None
            ),
            "reference_changed_pixel_ratio": (
                _round(result.reference_comparison.changed_pixel_ratio, 8)
                if result.reference_comparison is not None
                else None
            ),
        },
    )


def append_history(record: AnalysisRecord, path: Path | None = None) -> None:
    """Append a metadata-only record to the JSONL history file."""
    target = path or settings.history_file
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")


def load_history(path: Path | None = None, limit: int = 200) -> list[dict]:
    """Load history entries newest first. Corrupt lines are skipped."""
    target = path or settings.history_file
    if not target.exists():
        return []
    records: list[dict] = []
    try:
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return list(reversed(records[-limit:]))


def clear_history(path: Path | None = None) -> None:
    target = path or settings.history_file
    if target.exists():
        target.unlink()


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_manipulation_report(result, fingerprint: str) -> str:
    lines = [
        "Digital Image Forensics - Image Manipulation Analysis",
        "=" * 58,
        f"Generated (UTC): {_now_iso()}",
        f"Image fingerprint: {fingerprint}",
        f"Image size: {result.image_size[0]} x {result.image_size[1]}",
        f"Model: {result.model_info.get('name', 'MVSS-Net')} "
        f"({result.model_info.get('checkpoint', 'checkpoint')})",
        f"Inference mode: {result.mode}",
        "",
        "IMAGE-LEVEL DECISION",
        f"  {result.image_decision_label}",
        f"  Image score (maximum pixel probability): {_fmt(result.image_score)}",
        f"  Image decision threshold: {_fmt(result.image_threshold)}",
        f"  Uncertainty band: +/-{result.image_uncertainty_margin:.2f}",
        f"  Signed margin from threshold: {result.image_decision_margin:+.4f}",
        f"  Uncertain: {'yes' if result.uncertain else 'no'}",
        "  The decision is a threshold on the existing MVSS-Net probability map;",
        "  it is not a separately trained authenticity classifier.",
        "",
        "  Why this prediction:",
        f"  {result.decision_explanation or 'No explanation available.'}",
        "",
        "PIXEL-LEVEL LOCALIZATION SIGNAL",
        f"  {result.localization_label}",
        f"  Localization threshold: {_fmt(result.threshold)}",
        f"  Raw thresholded pixels: {result.raw_suspicious_pixels:,} of "
        f"{result.total_pixels:,} ({result.raw_suspicious_area_ratio * 100:.6f}%)",
        f"  Retained highlighted pixels: {result.processed_suspicious_pixels:,} of "
        f"{result.total_pixels:,} ({result.processed_suspicious_area_ratio * 100:.6f}%)",
        f"  Connected components: {result.raw_component_count} raw, "
        f"{len(result.regions)} retained, {result.dropped_component_count} not retained",
        "  A pixel-level signal does not prove that the whole image was manipulated.",
        "",
        "REGIONS",
    ]
    if result.regions:
        for region in result.regions:
            lines.append(
                f"  Region {region.region_id}: bbox={region.bbox}, "
                f"area={region.area_px}px ({region.area_ratio * 100:.6f}%), "
                f"max_score={region.max_score:.4f}"
            )
    elif result.raw_suspicious_pixels:
        lines.append(
            f"  {result.raw_suspicious_pixels:,} raw pixel(s) crossed the localization "
            "threshold, but no connected component survived the minimum-area filter."
        )
    else:
        lines.append("  No pixel crossed the localization threshold.")

    lines += ["", "WARNINGS"]
    lines += [f"  - {warning}" for warning in result.warnings]
    lines += ["", "LIMITATIONS"]
    lines += [f"  - {item}" for item in result.limitations]
    lines += [
        "",
        "This report is a model-assisted screening result, not proof of",
        "authenticity or manipulation. Scores are not calibrated.",
    ]
    return "\n".join(lines)


def build_ai_report(result, fingerprint: str) -> str:
    lines = [
        "Digital Image Forensics - AI-Generated Image Analysis",
        "=" * 58,
        f"Generated (UTC): {_now_iso()}",
        f"Image fingerprint: {fingerprint}",
        f"Status: {result.status}",
        f"Predicted class: {result.predicted_class or 'n/a'}",
        f"Real confidence: {_fmt(result.real_confidence)}",
        f"AI-generated confidence: {_fmt(result.ai_confidence)}",
        f"Uncertain: {result.uncertain}",
        f"Backend: {result.backend or 'not integrated'}",
        "",
        "WARNINGS",
    ]
    lines += [f"  - {warning}" for warning in result.warnings]
    lines += ["", "LIMITATIONS"]
    lines += [f"  - {item}" for item in result.limitations]
    return "\n".join(lines)


def build_chat_report(result, fingerprint: str) -> str:
    lines = [
        "Digital Image Forensics - Chat Screenshot Analysis",
        "=" * 58,
        f"Generated (UTC): {_now_iso()}",
        f"Image fingerprint: {fingerprint}",
        f"Image size: {result.image_size[0]} x {result.image_size[1]}",
        f"Pixel-level signal: {result.finding.replace('_', ' ').title()}",
        f"Maximum visual score: {_fmt(result.visual_score)}",
        f"Localization threshold: {_fmt(result.threshold)}",
        f"Suspicious regions: {len(result.regions)}",
        f"OCR backend: {result.ocr.backend if result.ocr else 'unavailable'}",
        f"OCR lines: {len(result.ocr.lines) if result.ocr else 0}",
    ]
    if result.manipulation is not None:
        lines += [
            f"Raw thresholded area: "
            f"{result.manipulation.raw_suspicious_area_ratio * 100:.6f}% "
            f"({result.manipulation.raw_suspicious_pixels:,} px)",
            f"Retained highlighted area: "
            f"{result.manipulation.processed_suspicious_area_ratio * 100:.6f}% "
            f"({result.manipulation.processed_suspicious_pixels:,} px)",
        ]
    lines += [
        "",
        "REGIONS",
    ]
    if result.regions:
        for region in result.regions:
            lines.append(
                f"  Region {region.region_id}: bbox={region.bbox}, "
                f"area={region.area_px}px, max_score={region.max_score:.4f}"
            )
    else:
        lines.append("  No connected region crossed the threshold.")

    lines += ["", "WARNINGS"]
    lines += [f"  - {warning}" for warning in result.warnings]
    if result.reference_comparison is not None:
        comparison = result.reference_comparison
        alignment = comparison.alignment
        lines += ["", "REFERENCE COMPARISON"]
        if alignment is not None:
            lines += [
                f"  Alignment: {alignment.method} "
                f"(correlation {alignment.correlation:.3f}, "
                f"reference resized: {'yes' if alignment.resized_reference else 'no'})",
            ]
        lines += [
            f"  Result: {comparison.finding.replace('_', ' ').title()}",
            f"  Changed pixels: {comparison.changed_pixels:,} "
            f"({comparison.changed_pixel_ratio * 100:.4f}% of the screen)",
            f"  Changed regions: {len(comparison.regions)}",
            f"  Text differences: {len(comparison.text_changes)}",
        ]
        if comparison.text_changes:
            kinds = sorted({change.kind.replace("_", " ") for change in comparison.text_changes})
            lines.append(f"  Change categories: {', '.join(kinds)}")
        lines.append(
            "  Recognised text is intentionally omitted from this report; "
            "review transcripts in the application."
        )
    lines += ["", "LIMITATIONS"]
    lines += [f"  - {item}" for item in result.limitations]
    lines += [
        "",
        "OCR text is shown for readability only. It is not evidence of",
        "tampering, and the original message cannot be recovered from a",
        "single screenshot without a trusted reference.",
    ]
    return "\n".join(lines)
