"""Pluggable AI-generated-image detector integration.

No detector weights ship with this repository.  Until a model is supplied, the
module reports ``unavailable`` and never fabricates a prediction.

Integration contract
--------------------
Create ``models/ai_detector/manifest.json`` describing the model.  Three
backend kinds are supported:

``torchscript``
    A TorchScript file exported from PyTorch.  Input: ``1x3xHxW`` float tensor
    (RGB, normalised with the declared mean/std).  Output: logits or
    probabilities with at least two classes.

``onnx``
    An ONNX model (requires the optional ``onnxruntime`` package).  Same input
    and output contract as above.

``python_adapter``
    ``"adapter": "package.module:ClassNameOrFunction"``.  The callable receives
    the RGB ``uint8`` NumPy array and returns a sequence of class probabilities
    (or logits).  Optional keyword arguments come from ``"adapter_kwargs"``.

Example manifest::

    {
      "kind": "torchscript",
      "path": "models/ai_detector/detector.ts",
      "input_size": 224,
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225],
      "class_names": ["real", "ai_generated"],
      "ai_class_index": 1
    }
"""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils.config import settings
from utils.image_utils import ensure_rgb_uint8

from .preprocessing import (
    DEFAULT_INPUT_SIZE,
    DEFAULT_MEAN,
    DEFAULT_STD,
    preprocess_ai_image,
    probabilities_from_output,
)


VALID_KINDS = {"torchscript", "onnx", "python_adapter"}
REQUIRED_KEYS = {"kind", "class_names"}


class AIDetectorUnavailable(RuntimeError):
    """Raised when no usable AI detector has been integrated."""


@dataclass
class AIIntegrationStatus:
    available: bool
    reason: str
    kind: str | None = None
    model_file: str | None = None
    class_names: list[str] | None = None
    manifest_path: str | None = None

    def as_dict(self) -> dict:
        return {
            "available": self.available,
            "reason": self.reason,
            "kind": self.kind,
            "model_file": self.model_file,
            "class_names": self.class_names,
            "manifest_path": self.manifest_path,
        }


def _resolve_model_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    # Paths in the manifest are relative to the manifest directory.
    return (manifest_path.parent / path).resolve()


def load_manifest(manifest_path: str | Path | None = None) -> dict:
    path = Path(manifest_path or settings.ai_manifest_path)
    if not path.exists():
        raise AIDetectorUnavailable(
            "No AI-generated-image detector has been integrated yet. "
            f"Expected a manifest at {path.name}."
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AIDetectorUnavailable(f"The AI detector manifest is not valid JSON: {exc}") from exc

    if not isinstance(manifest, dict):
        raise AIDetectorUnavailable("The AI detector manifest must be a JSON object.")
    missing = REQUIRED_KEYS - set(manifest)
    if missing:
        raise AIDetectorUnavailable(
            "The AI detector manifest is missing required keys: " + ", ".join(sorted(missing))
        )
    kind = str(manifest.get("kind", "")).lower()
    if kind not in VALID_KINDS:
        raise AIDetectorUnavailable(
            f"Unsupported detector kind '{kind}'. Use one of: {', '.join(sorted(VALID_KINDS))}."
        )
    class_names = manifest.get("class_names")
    if not isinstance(class_names, list) or len(class_names) < 2:
        raise AIDetectorUnavailable("'class_names' must list at least two classes.")
    manifest["class_names"] = [str(name) for name in class_names]
    manifest["_manifest_path"] = str(path)
    manifest["_manifest_dir"] = str(path.parent)
    return manifest


def integration_status(manifest_path: str | Path | None = None) -> AIIntegrationStatus:
    """Check integration state without loading any weights."""
    try:
        manifest = load_manifest(manifest_path)
    except AIDetectorUnavailable as exc:
        return AIIntegrationStatus(available=False, reason=str(exc))

    kind = str(manifest["kind"])
    model_name: str | None = None
    if kind in {"torchscript", "onnx"}:
        raw_path = manifest.get("path")
        if not raw_path:
            return AIIntegrationStatus(
                available=False,
                reason="The manifest must define 'path' for torchscript/onnx backends.",
                kind=kind,
                class_names=manifest["class_names"],
                manifest_path=manifest["_manifest_path"],
            )
        resolved = _resolve_model_path(Path(manifest["_manifest_path"]), str(raw_path))
        if not resolved.exists():
            return AIIntegrationStatus(
                available=False,
                reason=f"Detector file not found next to the manifest: {resolved.name}",
                kind=kind,
                class_names=manifest["class_names"],
                manifest_path=manifest["_manifest_path"],
            )
        model_name = resolved.name

    return AIIntegrationStatus(
        available=True,
        reason="Detector integration manifest found.",
        kind=kind,
        model_file=model_name,
        class_names=manifest["class_names"],
        manifest_path=manifest["_manifest_path"],
    )


class BaseAIBackend:
    """Minimal interface every backend implements."""

    name = "base"

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> dict:
        return {"backend": self.name}


class TorchScriptBackend(BaseAIBackend):
    name = "torchscript"

    def __init__(self, manifest: dict, device: str = "cpu"):
        import torch

        self.manifest = manifest
        self.device = torch.device(device)
        path = _resolve_model_path(Path(manifest["_manifest_path"]), str(manifest["path"]))
        self.path = path
        try:
            self.model = torch.jit.load(str(path), map_location=self.device)
        except Exception as exc:
            raise AIDetectorUnavailable(f"TorchScript detector could not be loaded: {exc}") from exc
        self.model.eval()
        self.input_size = manifest.get("input_size", DEFAULT_INPUT_SIZE)
        self.mean = tuple(manifest.get("mean", DEFAULT_MEAN))
        self.std = tuple(manifest.get("std", DEFAULT_STD))

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        import torch

        tensor = preprocess_ai_image(image_rgb, self.input_size, self.mean, self.std).to(self.device)
        with torch.inference_mode():
            output = self.model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return probabilities_from_output(output.detach().cpu().numpy())

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "model_file": self.path.name,
            "input_size": self.input_size,
        }


class OnnxBackend(BaseAIBackend):
    name = "onnx"

    def __init__(self, manifest: dict, device: str = "cpu"):
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise AIDetectorUnavailable(
                "The ONNX backend requires onnxruntime (pip install onnxruntime)."
            ) from exc
        self.manifest = manifest
        path = _resolve_model_path(Path(manifest["_manifest_path"]), str(manifest["path"]))
        self.path = path
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(str(path), providers=providers)
        except Exception as exc:
            raise AIDetectorUnavailable(f"ONNX detector could not be loaded: {exc}") from exc
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = manifest.get("input_size", DEFAULT_INPUT_SIZE)
        self.mean = tuple(manifest.get("mean", DEFAULT_MEAN))
        self.std = tuple(manifest.get("std", DEFAULT_STD))

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        tensor = preprocess_ai_image(image_rgb, self.input_size, self.mean, self.std)
        outputs = self.session.run(None, {self.input_name: tensor.numpy()})
        return probabilities_from_output(outputs[0])

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "model_file": self.path.name,
            "input_size": self.input_size,
        }


class PythonAdapterBackend(BaseAIBackend):
    name = "python_adapter"

    def __init__(self, manifest: dict, device: str = "cpu"):
        self.manifest = manifest
        target = str(manifest.get("adapter", "")).strip()
        if ":" not in target:
            raise AIDetectorUnavailable(
                "A python_adapter manifest must define 'adapter' as 'module:callable'."
            )
        module_name, attribute = target.split(":", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise AIDetectorUnavailable(f"Could not import adapter module '{module_name}': {exc}") from exc
        if not hasattr(module, attribute):
            raise AIDetectorUnavailable(f"Adapter '{target}' was not found.")

        adapter = getattr(module, attribute)
        kwargs: dict[str, Any] = dict(manifest.get("adapter_kwargs", {}))
        checkpoint = manifest.get("checkpoint")
        if checkpoint:
            kwargs.setdefault(
                "checkpoint",
                str(_resolve_model_path(Path(manifest["_manifest_path"]), str(checkpoint))),
            )
        self.object = None
        if isinstance(adapter, type):
            last_error = None
            for call_kwargs in (dict(kwargs, device=device), kwargs, {}):
                try:
                    self.object = adapter(**call_kwargs)
                    break
                except TypeError as exc:  # try a simpler constructor
                    last_error = exc
            if self.object is None:
                raise AIDetectorUnavailable(
                    f"Could not construct adapter '{target}': {last_error}"
                )
        else:
            self.object = adapter

        self.device = device

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        rgb = ensure_rgb_uint8(image_rgb)
        callable_object = getattr(self.object, "predict", None) or self.object
        output = callable_object(rgb)
        return probabilities_from_output(output)

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "adapter": self.manifest.get("adapter"),
        }


def build_backend(manifest: dict, device: str = "cpu") -> BaseAIBackend:
    kind = str(manifest["kind"]).lower()
    if kind == "torchscript":
        return TorchScriptBackend(manifest, device=device)
    if kind == "onnx":
        return OnnxBackend(manifest, device=device)
    if kind == "python_adapter":
        return PythonAdapterBackend(manifest, device=device)
    raise AIDetectorUnavailable(f"Unsupported detector kind: {kind}")


def load_backend(
    manifest_path: str | Path | None = None,
    device: str = "cpu",
) -> BaseAIBackend:
    manifest = load_manifest(manifest_path)
    return build_backend(manifest, device=device)


def classify_probabilities(
    probabilities: np.ndarray,
    class_names: list[str],
    *,
    ai_class_index: int | None = None,
    uncertainty_margin: float | None = None,
    min_confidence: float | None = None,
) -> dict:
    """Turn a probability vector into an honest three-way decision."""
    import torch

    probs = probabilities_from_output(probabilities)
    if len(probs) != len(class_names):
        raise ValueError(
            f"Detector returned {len(probs)} scores for {len(class_names)} class names."
        )

    if ai_class_index is None:
        ai_index = len(class_names) - 1
        for index, name in enumerate(class_names):
            lowered = name.lower()
            if any(token in lowered for token in ("ai", "fake", "generated", "synthetic")):
                ai_index = index
                break
    else:
        ai_index = int(ai_class_index)
    ai_index = min(max(ai_index, 0), len(class_names) - 1)
    real_index = next((index for index in range(len(class_names)) if index != ai_index), ai_index)

    real_confidence = float(probs[real_index])
    ai_confidence = float(probs[ai_index])
    top_confidence = max(real_confidence, ai_confidence)
    margin = settings.ai_uncertainty_margin if uncertainty_margin is None else float(uncertainty_margin)
    threshold = settings.ai_min_confidence if min_confidence is None else float(min_confidence)

    uncertain = top_confidence < threshold or abs(real_confidence - ai_confidence) < margin
    if uncertain:
        predicted = "uncertain"
    elif ai_confidence > real_confidence:
        predicted = "ai_generated"
    else:
        predicted = "real"

    return {
        "predicted_class": predicted,
        "real_confidence": real_confidence,
        "ai_confidence": ai_confidence,
        "uncertain": bool(uncertain),
        "top_confidence": float(top_confidence),
        "margin": float(abs(real_confidence - ai_confidence)),
        "probabilities": probs.astype(float).tolist(),
        "class_names": list(class_names),
        "ai_class_index": int(ai_index),
    }


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"
