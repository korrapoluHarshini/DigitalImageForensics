"""Module 2 - AI-generated image detection."""

from .inference import AIDetectionResult, detect_ai_generated_image, get_ai_status
from .model import (
    AIDetectorUnavailable,
    AIIntegrationStatus,
    BaseAIBackend,
    build_backend,
    classify_probabilities,
    integration_status,
    load_backend,
    load_manifest,
)

__all__ = [
    "detect_ai_generated_image",
    "get_ai_status",
    "AIDetectionResult",
    "AIDetectorUnavailable",
    "AIIntegrationStatus",
    "BaseAIBackend",
    "build_backend",
    "classify_probabilities",
    "integration_status",
    "load_backend",
    "load_manifest",
]
