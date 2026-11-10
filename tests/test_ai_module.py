from __future__ import annotations

import json

import numpy as np
import pytest

from modules.ai_generated_detection import (
    classify_probabilities,
    detect_ai_generated_image,
    get_ai_status,
    load_backend,
)
from modules.ai_generated_detection.model import integration_status


class FixedBackend:
    name = "fixed"

    def __init__(self, probabilities, class_names=("real", "ai_generated")):
        self._probabilities = np.asarray(probabilities, dtype=np.float64)
        self.manifest = {"class_names": list(class_names)}
        self.class_names = list(class_names)

    def predict(self, _image_rgb):
        return self._probabilities.copy()

    def describe(self):
        return {"backend": self.name}


def test_status_is_unavailable_without_manifest(tmp_path):
    status = get_ai_status(tmp_path / "missing.json")
    assert status["available"] is False
    assert "No AI-generated-image detector" in status["reason"]


def test_detect_returns_placeholder_without_model(rgb_image, tmp_path):
    result = detect_ai_generated_image(rgb_image, manifest_path=tmp_path / "missing.json")
    assert result.status == "unavailable"
    assert result.predicted_class is None
    assert result.real_confidence is None
    assert result.ai_confidence is None
    assert result.uncertain is None
    assert any("placeholder" in warning.lower() for warning in result.warnings)


def test_detect_ai_generated_with_fake_backend(rgb_image):
    result = detect_ai_generated_image(rgb_image, backend=FixedBackend([0.1, 0.9]))
    assert result.status == "ok"
    assert result.predicted_class == "ai_generated"
    assert result.ai_confidence == pytest.approx(0.9)
    assert result.real_confidence == pytest.approx(0.1)
    assert result.uncertain is False


def test_detect_real_with_fake_backend(rgb_image):
    result = detect_ai_generated_image(rgb_image, backend=FixedBackend([0.95, 0.05]))
    assert result.predicted_class == "real"
    assert result.uncertain is False


def test_detect_abstains_when_margin_is_small(rgb_image):
    result = detect_ai_generated_image(
        rgb_image,
        backend=FixedBackend([0.52, 0.48]),
        uncertainty_margin=0.2,
        min_confidence=0.6,
    )
    assert result.predicted_class == "uncertain"
    assert result.uncertain is True
    assert any("abstained" in warning.lower() for warning in result.warnings)


def test_detector_failure_is_contained(rgb_image):
    class BrokenBackend(FixedBackend):
        def predict(self, _image_rgb):
            raise RuntimeError("simulated detector failure")

    result = detect_ai_generated_image(rgb_image, backend=BrokenBackend([0.5, 0.5]))
    assert result.status == "error"
    assert "simulated detector failure" in result.error_message


def test_classify_probabilities_infers_ai_class_from_name():
    class_names = ["real", "ai_generated"]
    result = classify_probabilities(np.array([0.2, 0.8]), class_names)
    assert result["ai_class_index"] == 1
    assert result["predicted_class"] == "ai_generated"

    reversed_names = ["fake", "real"]
    reversed_result = classify_probabilities(np.array([0.8, 0.2]), reversed_names)
    assert reversed_result["ai_class_index"] == 0
    assert reversed_result["predicted_class"] == "ai_generated"


def test_classify_probabilities_accepts_logits():
    result = classify_probabilities(np.array([0.0, 4.0]), ["real", "ai_generated"])
    assert result["predicted_class"] == "ai_generated"
    assert result["ai_confidence"] > 0.9


def test_manifest_validation_rejects_bad_kind(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"kind": "magic", "class_names": ["real", "ai"]}), encoding="utf-8"
    )
    status = integration_status(manifest)
    assert status.available is False
    assert "Unsupported detector kind" in status.reason


def test_torchscript_backend_end_to_end(rgb_image, tmp_path):
    torch = pytest.importorskip("torch")

    class TinyDetector(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.15, 0.85]], dtype=x.dtype, device=x.device)

    model = TinyDetector().eval()
    example = torch.zeros(1, 3, 32, 32)
    try:
        scripted = torch.jit.trace(model, example, check_trace=False)
        model_path = tmp_path / "tiny.ts"
        scripted.save(str(model_path))
    except Exception as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"TorchScript tracing unavailable: {exc}")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "torchscript",
                "path": "tiny.ts",
                "input_size": 32,
                "class_names": ["real", "ai_generated"],
                "ai_class_index": 1,
            }
        ),
        encoding="utf-8",
    )

    status = integration_status(manifest_path)
    assert status.available is True
    backend = load_backend(manifest_path)
    probabilities = backend.predict(rgb_image)
    assert probabilities.shape == (2,)
    assert probabilities[1] > probabilities[0]

    result = detect_ai_generated_image(rgb_image, backend=backend)
    assert result.status == "ok"
    assert result.predicted_class == "ai_generated"
