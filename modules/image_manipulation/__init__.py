"""Module 1 - MVSS-Net image manipulation detection and localization."""

from .inference import (
    FINDING_MANIPULATED,
    FINDING_NO_EVIDENCE,
    IMAGE_DECISION_AUTHENTIC,
    IMAGE_DECISION_LABELS,
    IMAGE_DECISION_TAMPERED,
    IMAGE_DECISION_UNCERTAIN,
    LOCALIZATION_LABELS,
    ManipulationResult,
    SuspiciousRegion,
    build_decision_explanation,
    classify_image_score,
    detect_image_manipulation,
)
from .model import MvssNetPredictor, ModelUnavailableError, get_default_predictor, predictor_status

__all__ = [
    "detect_image_manipulation",
    "classify_image_score",
    "build_decision_explanation",
    "ManipulationResult",
    "SuspiciousRegion",
    "MvssNetPredictor",
    "ModelUnavailableError",
    "get_default_predictor",
    "predictor_status",
    "FINDING_MANIPULATED",
    "FINDING_NO_EVIDENCE",
    "IMAGE_DECISION_AUTHENTIC",
    "IMAGE_DECISION_TAMPERED",
    "IMAGE_DECISION_UNCERTAIN",
    "IMAGE_DECISION_LABELS",
    "LOCALIZATION_LABELS",
]
