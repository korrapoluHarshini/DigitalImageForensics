"""MVSS-Net model loading and probability-map inference.

Reuses the original architecture in ``models/mvssnet.py`` and the pretrained
CASIA checkpoint.  No weights are created or modified here.
"""

from __future__ import annotations

import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import cv2

from models.mvssnet import get_mvss
from utils.config import settings
from utils.image_utils import bgr_to_rgb, ensure_rgb_uint8, rgb_to_bgr

from .preprocessing import (
    DEFAULT_INPUT_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_PATCH_SIZE,
    downscale_for_tile_budget,
    gaussian_window,
    get_patch_positions,
    preprocess_bgr,
    resize_probability_map,
)


class ModelUnavailableError(RuntimeError):
    """Raised when the MVSS-Net checkpoint or architecture cannot be loaded."""


def resolve_device(requested: str | None = None) -> torch.device:
    value = (requested or settings.device or "auto").lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cpu") if value not in {"cpu", "cuda"} else torch.device(value)


class MvssNetPredictor:
    """Thread-safe, lazily loaded MVSS-Net wrapper.

    The checkpoint is not touched until the first prediction, so the dashboard
    stays responsive and a missing checkpoint only affects Module 1.
    """

    def __init__(self, checkpoint_path: str | Path | None = None, device: str | None = None):
        self.checkpoint_path = Path(checkpoint_path or settings.checkpoint_path)
        self.device = resolve_device(device)
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self.load_seconds: float | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def available(self) -> tuple[bool, str]:
        if not self.checkpoint_path.exists():
            return False, (
                f"Checkpoint not found: {self.checkpoint_path.name}. "
                "Place mvssnet_casia.pt in the ckpt/ directory."
            )
        try:
            import models.mvssnet  # noqa: F401
        except Exception as exc:  # pragma: no cover - import failure
            return False, f"MVSS-Net architecture could not be imported: {exc}"
        return True, "ready"

    def load(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            if not self.checkpoint_path.exists():
                raise ModelUnavailableError(
                    f"Checkpoint not found: {self.checkpoint_path}. "
                    "Download the pretrained model and place it in ckpt/."
                )
            started = time.perf_counter()
            try:
                model = get_mvss(
                    backbone="resnet50",
                    pretrained_base=False,
                    nclass=1,
                    sobel=True,
                    constrain=True,
                    n_input=3,
                )
                checkpoint = torch.load(
                    self.checkpoint_path,
                    map_location=self.device,
                    weights_only=False,
                )
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
                # Common DataParallel prefix from multi-GPU training.
                if isinstance(checkpoint, dict) and checkpoint and all(
                    str(key).startswith("module.") for key in checkpoint
                ):
                    checkpoint = {str(key)[7:]: value for key, value in checkpoint.items()}
                model.load_state_dict(checkpoint, strict=True)
                model.to(self.device)
                model.eval()
            except Exception as exc:
                raise ModelUnavailableError(
                    f"MVSS-Net could not be loaded from {self.checkpoint_path.name}: {exc}"
                ) from exc
            self._model = model
            self.load_seconds = time.perf_counter() - started
            return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _forward_probability(self, bgr_patch: np.ndarray) -> np.ndarray:
        model = self.load()
        tensor = preprocess_bgr(bgr_patch, size=DEFAULT_INPUT_SIZE).to(self.device)
        with torch.inference_mode():
            _, logits = model(tensor)
            probability = torch.sigmoid(logits)[0, 0]
        return probability.detach().cpu().numpy().astype(np.float32)

    def predict_probability(
        self,
        rgb_image: np.ndarray,
        *,
        tiled: bool = False,
        patch_size: int = DEFAULT_PATCH_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        max_tiles: int = 16,
    ) -> tuple[np.ndarray, dict]:
        """Return a full-resolution probability map and metadata.

        ``tiled=False`` reproduces the official single-scale evaluation: the
        whole image is resized to 512x512.  ``tiled=True`` blends overlapping
        512x512 patches (useful for tall screenshots); the number of patches is
        capped by ``max_tiles`` to bound CPU memory and latency.
        """
        rgb = ensure_rgb_uint8(rgb_image)
        bgr = rgb_to_bgr(rgb)
        original_height, original_width = bgr.shape[:2]
        metadata: dict = {
            "tiled": bool(tiled),
            "input_size": DEFAULT_INPUT_SIZE,
            "device": str(self.device),
        }

        with self._infer_lock:
            if not tiled:
                probability = self._forward_probability(bgr)
                metadata["tiles"] = 1
            else:
                working, scale = downscale_for_tile_budget(
                    bgr, patch_size, overlap, max_tiles=max_tiles
                )
                # Pad tiny dimensions so every patch is exactly patch_size.
                original_work_height, original_work_width = working.shape[:2]
                pad_height = max(0, patch_size - original_work_height)
                pad_width = max(0, patch_size - original_work_width)
                if pad_height or pad_width:
                    working = cv2.copyMakeBorder(
                        working,
                        0,
                        pad_height,
                        0,
                        pad_width,
                        borderType=cv2.BORDER_REPLICATE,
                    )
                height, width = working.shape[:2]
                accumulator = np.zeros((height, width), dtype=np.float32)
                weight_sum = np.zeros((height, width), dtype=np.float32)
                window = gaussian_window(patch_size)
                y_positions = get_patch_positions(height, patch_size, overlap)
                x_positions = get_patch_positions(width, patch_size, overlap)

                for y in y_positions:
                    for x in x_positions:
                        patch = working[y : y + patch_size, x : x + patch_size]
                        patch_probability = self._forward_probability(patch)
                        # Each patch is resized to 512 inside the model; map back
                        # to the patch's actual size for blending.
                        if patch_probability.shape != patch_shape(patch):
                            patch_probability = resize_probability_map(
                                patch_probability, patch.shape[0], patch.shape[1]
                            )
                        accumulator[y : y + patch_size, x : x + patch_size] += (
                            patch_probability * window
                        )
                        weight_sum[y : y + patch_size, x : x + patch_size] += window

                probability = accumulator / np.maximum(weight_sum, 1e-6)
                probability = probability[:original_work_height, :original_work_width]
                metadata.update(
                    {
                        "tiles": int(len(y_positions) * len(x_positions)),
                        "tile_grid": [len(y_positions), len(x_positions)],
                        "working_size": [int(original_work_width), int(original_work_height)],
                        "downscale": round(float(scale), 4),
                    }
                )

        probability = resize_probability_map(probability, original_height, original_width)
        return probability, metadata

    # ------------------------------------------------------------------
    # Display-safe description
    # ------------------------------------------------------------------
    def describe(self) -> dict:
        return {
            "name": "MVSS-Net (ResNet-50 backbone)",
            "checkpoint": self.checkpoint_path.name,
            "checkpoint_present": self.checkpoint_path.exists(),
            "device": str(self.device),
            "input_size": DEFAULT_INPUT_SIZE,
            "task": "Pixel-level manipulation localization (CASIA2.0 trained)",
            "loaded": self.loaded,
        }


def patch_shape(patch: np.ndarray) -> tuple[int, int]:
    return int(patch.shape[0]), int(patch.shape[1])


@lru_cache(maxsize=1)
def get_default_predictor() -> MvssNetPredictor:
    """Process-wide lazy singleton used by the Streamlit app."""
    predictor = MvssNetPredictor()
    return predictor


def predictor_status() -> dict:
    """Cheap status probe (does not load the 560 MB checkpoint)."""
    predictor = get_default_predictor()
    available, reason = predictor.available()
    info = {
        "available": available,
        "reason": reason,
        "checkpoint": predictor.checkpoint_path.name,
        "device": str(predictor.device),
        "loaded": predictor.loaded,
    }
    if predictor.load_seconds is not None:
        info["load_seconds"] = round(predictor.load_seconds, 2)
    return info
