"""Central configuration for the Digital Image Forensics application.

All paths can be overridden with environment variables so that the app does not
have to be moved next to the datasets it evaluates against.

Environment variables
---------------------
DIGITAL_FORENSICS_DATA_ROOT
    Folder that contains ``datasets/`` and ``experiments/``.  Defaults to the
    legacy project folder when it exists, otherwise this repository.
MVSS_CHECKPOINT
    Path to the MVSS-Net checkpoint (default: ``ckpt/mvssnet_casia.pt``).
MVSS_THRESHOLD
    Pixel-level localization threshold (default: 0.4, selected on the
    validation split).
MVSS_IMAGE_THRESHOLD
    Image-level Authentic/Tampered decision threshold applied to the maximum
    pixel probability (default: 0.42, selected on 767 tampered + 398 authentic
    validation images; see results/evaluation/module1_image_level_threshold.json).
MVSS_IMAGE_UNCERTAINTY_MARGIN
    Half-width of the uncertainty band around the image-level threshold
    (default: 0.13).  Scores inside the band are reported as uncertain.
MVSS_DEVICE
    ``auto``, ``cpu`` or ``cuda`` (default: ``auto``).
FORENSICS_MAX_UPLOAD_MB
    Maximum accepted upload size in megabytes (default: 25).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

_LEGACY_DATA_ROOT = Path(r"C:\Users\harsh\Documents\DigitalImageForensics")


def _as_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    return Path(value).expanduser().resolve()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _default_data_root() -> Path:
    configured = os.environ.get("DIGITAL_FORENSICS_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    if _LEGACY_DATA_ROOT.exists():
        return _LEGACY_DATA_ROOT
    return PROJECT_ROOT


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings shared by every module."""

    project_root: Path
    data_root: Path
    checkpoint_path: Path

    dataset_dir: Path
    groundtruth_dir: Path
    split_csv: Path
    pairs_csv: Path

    results_dir: Path
    evaluation_dir: Path
    history_file: Path
    app_artifacts_dir: Path

    ai_model_dir: Path
    ai_manifest_path: Path
    chat_model_dir: Path

    manipulation_threshold: float
    image_level_threshold: float
    image_uncertainty_margin: float
    device: str

    max_upload_mb: int
    max_image_pixels: int
    max_analysis_side: int

    ocr_languages: tuple[str, ...]
    ai_uncertainty_margin: float
    ai_min_confidence: float

    @classmethod
    def from_env(cls) -> "Settings":
        data_root = _default_data_root()
        results_dir = PROJECT_ROOT / "results" / "app"
        model_dir = PROJECT_ROOT / "models"
        return cls(
            project_root=PROJECT_ROOT,
            data_root=data_root,
            checkpoint_path=_as_path(
                os.environ.get("MVSS_CHECKPOINT"),
                PROJECT_ROOT / "ckpt" / "mvssnet_casia.pt",
            ),
            dataset_dir=data_root / "datasets" / "CASIA2.0_revised",
            groundtruth_dir=data_root / "datasets" / "CASIA2.0_Groundtruth",
            split_csv=data_root / "experiments" / "dataset_split.csv",
            pairs_csv=data_root / "experiments" / "localization_pairs.csv",
            results_dir=results_dir,
            evaluation_dir=PROJECT_ROOT / "results" / "evaluation",
            history_file=results_dir / "history.jsonl",
            app_artifacts_dir=results_dir / "artifacts",
            ai_model_dir=model_dir / "ai_detector",
            ai_manifest_path=model_dir / "ai_detector" / "manifest.json",
            chat_model_dir=model_dir / "chat_detector",
            manipulation_threshold=_env_float("MVSS_THRESHOLD", 0.4),
            image_level_threshold=_env_float("MVSS_IMAGE_THRESHOLD", 0.42),
            image_uncertainty_margin=_env_float("MVSS_IMAGE_UNCERTAINTY_MARGIN", 0.13),
            device=os.environ.get("MVSS_DEVICE", "auto").strip().lower() or "auto",
            max_upload_mb=max(1, _env_int("FORENSICS_MAX_UPLOAD_MB", 25)),
            max_image_pixels=max(1, _env_int("FORENSICS_MAX_IMAGE_PIXELS", 60_000_000)),
            max_analysis_side=max(512, _env_int("FORENSICS_MAX_ANALYSIS_SIDE", 4096)),
            ocr_languages=("en",),
            ai_uncertainty_margin=_env_float("AI_UNCERTAINTY_MARGIN", 0.20),
            ai_min_confidence=_env_float("AI_MIN_CONFIDENCE", 0.60),
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def manipulation_threshold_source(self) -> str:
        """Human readable provenance for the pixel-level threshold."""
        return (
            "0.4 selected on the CASIA2.0 validation split "
            "(see results/evaluation/validation_threshold_selection.json)"
        )

    @property
    def image_level_threshold_source(self) -> str:
        """Human readable provenance for the image-level decision threshold."""
        return (
            "0.42 selected on 767 tampered + 398 authentic validation images "
            "(maximum recall subject to validation FPR <= 5%); uncertainty band "
            "+/-0.13 chosen with >= 90% coverage. The test split was not used for "
            "selection. See results/evaluation/module1_image_level_threshold.json."
        )

    def ensure_output_dirs(self) -> None:
        for path in (
            self.results_dir,
            self.evaluation_dir,
            self.app_artifacts_dir,
            self.ai_model_dir,
            self.chat_model_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings.from_env()
