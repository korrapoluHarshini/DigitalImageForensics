"""Testable helpers for the one-cell-per-stage Colab workflow.

Nothing here trains or evaluates a model.  The helpers only inspect files,
validate the preserved inputs, plan whether a run should start/resume/stop and
package small artefacts.  That keeps the notebook short and lets the same logic
be unit-tested on a laptop without a GPU.

Conventions
-----------
* Manifest ``image_path``/``mask_path`` values are relative to a *data root*
  that contains a ``datasets/`` folder.
* The original checkpoint and the group-aware split are treated as immutable.
  Their SHA-256 digests are checked, never rewritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Preserved inputs (verified before any training starts)
# ---------------------------------------------------------------------------

CHECKPOINT_SHA256 = "080BC6C3AAE59F748B547DBF090786FE9D31A6E50749DAAA40871E298D6A7E50"
GROUP_AWARE_SPLIT_SHA256 = "BE5ACAB205E294A563397C8E6756A6CA429DAD01E3F4FEE5EEBDC84E2593E49C"

EXPECTED_THRESHOLDS = {
    "pixel": 0.40,
    "image": 0.42,
    "uncertainty": 0.13,
}

EXPECTED_SPLIT_COUNTS = {"train": 8831, "validation": 1694, "test": 2089}

# Config fields that must match when resuming an existing run.  ``epochs`` is
# intentionally absent: a run may be extended.  ``batch_size`` and
# ``learning_rate`` are soft fields: changing them (for example after a CUDA
# out-of-memory error) is allowed and only reported, because the optimizer state
# does not depend on them.
COMPATIBLE_CONFIG_KEYS = (
    "image_size",
    "seed",
    "limit_train",
    "limit_val",
)
SOFT_CONFIG_KEYS = ("batch_size", "learning_rate", "num_workers")

RUN_ARTIFACTS = (
    "last.pt",
    "best.pt",
    "mvssnet_groupaware.pt",
    "training_log.csv",
    "run.json",
    "split_used.csv",
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the upper-case SHA-256 of ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _iter_existing(root: Path, name: str, max_depth: int = 3) -> list[Path]:
    found: list[Path] = []
    direct = root / name
    if direct.is_dir():
        found.append(direct)
    for depth in range(1, max_depth + 1):
        pattern = "/".join(["*"] * depth) + f"/{name}"
        for candidate in sorted(root.glob(pattern)):
            if candidate.is_dir() and candidate not in found:
                found.append(candidate)
    return found


def _first_existing(root: Path, name: str, max_depth: int = 3) -> Path | None:
    candidates = _iter_existing(root, name, max_depth)
    if name == "CASIA2.0_revised":
        # Zips sometimes extract as CASIA2.0_revised/CASIA2.0_revised/...
        # Prefer the copy that actually contains the Au/Tp image folders.
        for candidate in candidates:
            if (candidate / "Au").is_dir() or (candidate / "Tp").is_dir():
                return candidate
    if name == "CASIA2.0_Groundtruth":
        for candidate in candidates:
            if any(
                entry.is_file() and entry.suffix.lower() == ".png"
                for entry in candidate.iterdir()
            ):
                return candidate
    return candidates[0] if candidates else None


def detect_dataset_root(casia_dir: Path) -> dict:
    """Locate ``CASIA2.0_revised``/``CASIA2.0_Groundtruth`` and build a data root.

    The manifest paths look like ``datasets/CASIA2.0_revised/...``.  If the
    extracted dataset already has that shape, its parent is the data root.
    Otherwise a small staging directory with symlinks is created so that the
    same manifest works without copying ~9 GB of images.
    """
    casia_dir = Path(casia_dir).expanduser().resolve()
    if not casia_dir.is_dir():
        raise FileNotFoundError(f"CASIA folder not found: {casia_dir}")

    revised = _first_existing(casia_dir, "CASIA2.0_revised")
    truth = _first_existing(casia_dir, "CASIA2.0_Groundtruth")
    if revised is None or truth is None:
        raise FileNotFoundError(
            "Could not find CASIA2.0_revised and CASIA2.0_Groundtruth under "
            f"{casia_dir}. Expected an extracted CASIA 2.0 folder."
        )

    if revised.parent.name == "datasets":
        data_root = revised.parent.parent
        staged = False
    else:
        staging = casia_dir.parent / "mvss_staging"
        datasets = staging / "datasets"
        datasets.mkdir(parents=True, exist_ok=True)
        for name, source in (
            ("CASIA2.0_revised", revised),
            ("CASIA2.0_Groundtruth", truth),
        ):
            link = datasets / name
            if link.is_symlink() or link.exists():
                if link.is_symlink() and link.resolve() != source.resolve():
                    link.unlink()
                    link.symlink_to(source, target_is_directory=True)
            else:
                try:
                    link.symlink_to(source, target_is_directory=True)
                except OSError as exc:  # pragma: no cover - platform specific
                    raise OSError(
                        "Could not create a dataset symlink. Pass a CASIA folder "
                        "that already contains a 'datasets/' directory instead."
                    ) from exc
        data_root = staging
        staged = True

    resolved_revised = data_root / "datasets" / "CASIA2.0_revised"
    resolved_truth = data_root / "datasets" / "CASIA2.0_Groundtruth"
    if not resolved_revised.is_dir() or not resolved_truth.is_dir():
        raise FileNotFoundError(
            f"Staged data root does not resolve: {data_root}. "
            "Manifest-relative paths cannot be found."
        )

    return {
        "data_root": str(data_root),
        "revised": str(revised),
        "groundtruth": str(truth),
        "staged": staged,
        "datasets_link": str(data_root / "datasets"),
    }


# ---------------------------------------------------------------------------
# Manifest / split audit (pure function + file wrapper)
# ---------------------------------------------------------------------------


def audit_manifest_frames(
    manifest: pd.DataFrame,
    split: pd.DataFrame,
    path_exists: Callable[[str], bool],
) -> dict:
    """Audit manifest + group-aware split without touching the filesystem logic.

    ``path_exists`` receives a manifest-relative POSIX path and must return
    whether the resolved file exists.
    """
    required_manifest = {"sample_id", "class_name", "image_path", "mask_path"}
    required_split = {"sample_id", "experimental_split"}
    missing_manifest = required_manifest - set(manifest.columns)
    missing_split = required_split - set(split.columns)
    if missing_manifest or missing_split:
        raise ValueError(
            "Manifest/split columns missing: "
            f"manifest={sorted(missing_manifest)} split={sorted(missing_split)}"
        )
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate sample_id values.")
    if split["sample_id"].duplicated().any():
        raise ValueError("Split contains duplicate sample_id values.")
    invalid_splits = set(split["experimental_split"]) - {"train", "validation", "test"}
    if invalid_splits:
        raise ValueError(f"Unexpected experimental_split values: {sorted(invalid_splits)}")

    merged = manifest.merge(
        split[["sample_id", "experimental_split"]], on="sample_id", how="inner"
    )

    split_counts = {
        name: int(count)
        for name, count in split["experimental_split"].value_counts().items()
    }
    class_counts: dict[str, dict[str, int]] = {}
    for class_name, frame in merged.groupby("class_name"):
        class_counts[str(class_name)] = {
            str(split_name): int(count)
            for split_name, count in frame["experimental_split"].value_counts().items()
        }

    spanning_components = 0
    components = 0
    if "duplicate_component" in split.columns:
        grouped = split.groupby("duplicate_component")["experimental_split"].nunique()
        components = int(grouped.size)
        spanning_components = int((grouped > 1).sum())

    missing_images = [
        str(row.image_path)
        for row in merged.itertuples(index=False)
        if not path_exists(str(row.image_path))
    ]
    tampered = merged[merged["class_name"] == "Tampered"]
    missing_masks = [
        str(row.sample_id)
        for row in tampered.itertuples(index=False)
        if not (isinstance(row.mask_path, str) and row.mask_path.strip())
        or not path_exists(str(row.mask_path))
    ]

    return {
        "manifest_rows": int(len(manifest)),
        "split_rows": int(len(split)),
        "matched_rows": int(len(merged)),
        "unmatched_manifest_rows": int(len(manifest) - len(merged)),
        "unmatched_split_rows": int(len(split) - len(merged)),
        "split_counts": split_counts,
        "class_counts": class_counts,
        "components": components,
        "components_spanning_splits": spanning_components,
        "missing_images": len(missing_images),
        "missing_image_examples": missing_images[:5],
        "tampered_rows": int(len(tampered)),
        "tampered_without_mask": len(missing_masks),
        "tampered_without_mask_examples": missing_masks[:5],
    }


def validate_inputs(repo: Path, data_root: Path) -> dict:
    """Check the preserved checkpoint, split hash, counts, layout and files."""
    repo = Path(repo).resolve()
    data_root = Path(data_root).resolve()

    manifest_path = repo / "results" / "manifests" / "casia2_manifest.csv"
    split_path = repo / "results" / "manifests" / "casia2_groupaware_split.csv"
    checkpoint_path = repo / "ckpt" / "mvssnet_casia.pt"
    for path in (manifest_path, split_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required preserved input is missing: {path}")

    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != CHECKPOINT_SHA256:
        raise ValueError(
            "The original checkpoint hash changed. Refusing to train on it. "
            f"Expected {CHECKPOINT_SHA256}, got {checkpoint_sha}."
        )
    split_sha = sha256_file(split_path)
    if split_sha != GROUP_AWARE_SPLIT_SHA256:
        raise ValueError(
            "The group-aware split hash changed. It must not be edited. "
            f"Expected {GROUP_AWARE_SPLIT_SHA256}, got {split_sha}."
        )

    manifest = pd.read_csv(manifest_path)
    split = pd.read_csv(split_path)

    def path_exists(relative: str) -> bool:
        candidate = Path(str(relative).replace("\\", "/"))
        resolved = candidate if candidate.is_absolute() else data_root / candidate
        return resolved.is_file()

    audit = audit_manifest_frames(manifest, split, path_exists)
    audit.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "split": str(split_path),
            "split_sha256": split_sha,
            "data_root": str(data_root),
            "expected_split_counts": EXPECTED_SPLIT_COUNTS,
            "split_counts_match": audit["split_counts"] == EXPECTED_SPLIT_COUNTS,
        }
    )
    if audit["missing_images"]:
        raise FileNotFoundError(
            f"{audit['missing_images']} manifest image(s) do not exist under {data_root}. "
            f"First: {audit['missing_image_examples'][0]}"
        )
    return audit


# ---------------------------------------------------------------------------
# Thresholds and model contract
# ---------------------------------------------------------------------------


def threshold_report(repo: Path) -> dict:
    """Read the deployed thresholds and compare them with the expected values."""
    repo = Path(repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from utils.config import settings  # imported lazily so tests stay light

    actual = {
        "pixel": float(settings.manipulation_threshold),
        "image": float(settings.image_level_threshold),
        "uncertainty": float(settings.image_uncertainty_margin),
    }
    consistent = all(
        abs(actual[key] - EXPECTED_THRESHOLDS[key]) < 1e-9 for key in EXPECTED_THRESHOLDS
    )
    return {
        "actual": actual,
        "expected": dict(EXPECTED_THRESHOLDS),
        "consistent": consistent,
        "pixel_source": settings.manipulation_threshold_source,
        "image_source": settings.image_level_threshold_source,
        "checkpoint": str(settings.checkpoint_path),
        "env_overrides": {
            key: os.environ[key]
            for key in (
                "MVSS_THRESHOLD",
                "MVSS_IMAGE_THRESHOLD",
                "MVSS_IMAGE_UNCERTAINTY_MARGIN",
                "MVSS_CHECKPOINT",
            )
            if key in os.environ
        },
    }


def _resolve_data_file(data_root: Path, relative: str) -> Path:
    candidate = Path(str(relative).replace("\\", "/"))
    return candidate if candidate.is_absolute() else Path(data_root) / candidate


def verify_model_contract(
    repo: Path,
    data_root: Path,
    checkpoint: Path,
    image_size: int = 512,
    device: str = "cuda",
    split: str = "validation",
) -> dict:
    """Load a checkpoint through the app's preprocessing and check output shapes.

    This is the consistency gate between training, evaluation and Streamlit:
    BGR input, 512x512, ImageNet normalisation, mask output at input resolution
    and edge output at one quarter resolution.
    """
    from pathlib import Path as _Path

    import cv2
    import numpy as np
    import torch

    repo = Path(repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from models.mvssnet import get_mvss
    from modules.image_manipulation.preprocessing import preprocess_bgr

    manifest = pd.read_csv(repo / "results" / "manifests" / "casia2_manifest.csv")
    split_frame = pd.read_csv(repo / "results" / "manifests" / "casia2_groupaware_split.csv")
    merged = manifest.merge(
        split_frame[["sample_id", "experimental_split"]], on="sample_id", how="inner"
    )
    candidates = merged[
        (merged["experimental_split"] == split) & (merged["class_name"] == "Tampered")
    ]
    if candidates.empty:
        raise ValueError(f"No tampered {split} image available for the contract check.")
    row = candidates.iloc[0]
    image_path = _resolve_data_file(Path(data_root), str(row.image_path))
    mask_path = _resolve_data_file(Path(data_root), str(row.mask_path))

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Image could not be decoded: {image_path}")
    tensor = preprocess_bgr(bgr, size=image_size)

    model = get_mvss(
        backbone="resnet50",
        pretrained_base=False,
        nclass=1,
        sobel=True,
        constrain=True,
        n_input=3,
    ).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if isinstance(state, dict) and state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.inference_mode():
        edge_logits, mask_logits = model(tensor.to(device))
        probabilities = torch.sigmoid(mask_logits)
        edge_probabilities = torch.sigmoid(edge_logits)

    expected_mask = (1, 1, image_size, image_size)
    expected_edge = (1, 1, image_size // 4, image_size // 4)
    if tuple(mask_logits.shape) != expected_mask:
        raise AssertionError(f"Mask output {tuple(mask_logits.shape)} != {expected_mask}")
    if tuple(edge_logits.shape) != expected_edge:
        raise AssertionError(f"Edge output {tuple(edge_logits.shape)} != {expected_edge}")
    if not torch.isfinite(probabilities).all():
        raise AssertionError("Model produced non-finite probabilities.")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask could not be decoded: {mask_path}")
    resized = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    foreground = resized > 127

    return {
        "sample_id": str(row.sample_id),
        "image": str(image_path),
        "input_shape": tuple(tensor.shape),
        "mask_output_shape": tuple(mask_logits.shape),
        "edge_output_shape": tuple(edge_logits.shape),
        "mask_probability_range": [
            round(float(probabilities.min()), 6),
            round(float(probabilities.max()), 6),
        ],
        "edge_probability_range": [
            round(float(edge_probabilities.min()), 6),
            round(float(edge_probabilities.max()), 6),
        ],
        "ground_truth_foreground_fraction": round(float(foreground.mean()), 6),
        "contract_ok": True,
    }


def app_integration_check(
    repo: Path,
    checkpoint: Path,
    data_root: Path,
    image_relative: str | None = None,
    timeout: int = 900,
) -> dict:
    """Run the Streamlit app's own predictor against ``checkpoint`` in a subprocess.

    Uses ``MVSS_CHECKPOINT``/``DIGITAL_FORENSICS_DATA_ROOT`` so the check exercises
    the same configuration path a user would use to demo the retrained model.
    """
    repo = Path(repo).resolve()
    manifest = pd.read_csv(repo / "results" / "manifests" / "casia2_manifest.csv")
    split_frame = pd.read_csv(repo / "results" / "manifests" / "casia2_groupaware_split.csv")
    merged = manifest.merge(
        split_frame[["sample_id", "experimental_split"]], on="sample_id", how="inner"
    )
    if image_relative is None:
        tampered = merged[
            (merged["experimental_split"] == "validation") & (merged["class_name"] == "Tampered")
        ]
        image_relative = str(tampered.iloc[0]["image_path"])
    image_path = _resolve_data_file(Path(data_root), image_relative)

    script = r"""
import json, os, sys
sys.path.insert(0, os.environ["MVSS_REPO"])
import cv2
from modules.image_manipulation import get_default_predictor, predictor_status
status = predictor_status()
result = {"predictor_status": status}
if not status.get("available"):
    print("RESULT=" + json.dumps(result))
    raise SystemExit(3)
image = cv2.imread(os.environ["MVSS_IMAGE"], cv2.IMREAD_COLOR)
if image is None:
    print("RESULT=" + json.dumps({"error": "image not decodable"}))
    raise SystemExit(4)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
probability, info = get_default_predictor().predict_probability(rgb)
result.update(
    {
        "source_image_shape": list(image.shape[:2]),
        "probability_shape": list(probability.shape),
        "probability_max": float(probability.max()),
        "probability_mean": float(probability.mean()),
        "info_keys": sorted(info.keys()),
    }
)
print("RESULT=" + json.dumps(result))
"""

    env = dict(os.environ)
    env.update(
        {
            "MVSS_REPO": str(repo),
            "MVSS_CHECKPOINT": str(Path(checkpoint).resolve()),
            "DIGITAL_FORENSICS_DATA_ROOT": str(Path(data_root).resolve()),
            "MVSS_IMAGE": str(image_path),
            "MVSS_DEVICE": env.get("MVSS_DEVICE", "cuda"),
            "PYTHONPATH": str(repo),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    payload: dict = {}
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT="):
            payload = json.loads(line[len("RESULT=") :])
            break
    if completed.returncode != 0 or not payload:
        raise RuntimeError(
            "Streamlit predictor integration check failed "
            f"(exit {completed.returncode}).\nSTDOUT:\n{completed.stdout[-2000:]}\n"
            f"STDERR:\n{completed.stderr[-2000:]}"
        )
    payload["image"] = str(image_path)
    payload["used_checkpoint"] = str(Path(checkpoint).resolve())
    return payload


# ---------------------------------------------------------------------------
# Resume planning
# ---------------------------------------------------------------------------


def read_training_state(run_dir: Path) -> dict | None:
    """Load ``last.pt`` metadata, or return ``None`` when there is no run."""
    import torch

    last = Path(run_dir) / "last.pt"
    if not last.is_file():
        return None
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or {}
    return {
        "path": str(last),
        "epoch": int(checkpoint.get("epoch", -1)),
        "completed_epochs": int(checkpoint.get("epoch", -1)) + 1,
        "best_iou": float(checkpoint.get("best_iou", -1.0)),
        "history_length": len(checkpoint.get("history") or []),
        "config": config,
        "size_mb": round(last.stat().st_size / (1024 * 1024), 1),
    }


def _config_mismatches(saved: dict, desired: dict) -> list[str]:
    mismatches: list[str] = []
    for key in COMPATIBLE_CONFIG_KEYS:
        if key in desired and str(saved.get(key)) != str(desired[key]):
            mismatches.append(f"{key}: saved={saved.get(key)!r} desired={desired[key]!r}")
    for key in ("manifest", "split_csv"):
        if key in desired:
            saved_name = Path(str(saved.get(key, ""))).name
            desired_name = Path(str(desired[key])).name
            if saved_name != desired_name:
                mismatches.append(f"{key}: saved={saved_name!r} desired={desired_name!r}")
    return mismatches


def _soft_changes(saved: dict, desired: dict) -> list[str]:
    changes: list[str] = []
    for key in SOFT_CONFIG_KEYS:
        if key in desired and str(saved.get(key)) != str(desired[key]):
            changes.append(f"{key}: saved={saved.get(key)!r} new={desired[key]!r}")
    return changes


def plan_training_run(run_dir: Path, desired: dict) -> dict:
    """Decide whether to start, resume, export only, archive or stop.

    Actions
    -------
    ``start``        empty/missing directory, no checkpoint
    ``resume``       compatible ``last.pt`` with epochs remaining
    ``export_only``  training finished but the app checkpoint is missing
    ``complete``     finished and app checkpoint present; nothing to do
    ``incompatible`` existing checkpoint was produced by a different config
    ``archive``      non-empty directory without a usable checkpoint
    """
    run_dir = Path(run_dir)
    state = read_training_state(run_dir)
    if state is not None:
        mismatches = _config_mismatches(state["config"], desired)
        changed = _soft_changes(state["config"], desired)
        if mismatches:
            return {
                "action": "incompatible",
                "state": state,
                "reasons": mismatches,
                "changed": changed,
            }
        requested_epochs = int(desired.get("epochs", 0))
        if state["completed_epochs"] >= requested_epochs > 0:
            app_checkpoint = run_dir / "mvssnet_groupaware.pt"
            if app_checkpoint.is_file():
                return {"action": "complete", "state": state, "changed": changed}
            return {
                "action": "export_only",
                "state": state,
                "resume": state["path"],
                "changed": changed,
                "reason": "training finished earlier but the app checkpoint is missing",
            }
        return {
            "action": "resume",
            "state": state,
            "resume": state["path"],
            "next_epoch": state["completed_epochs"] + 1,
            "changed": changed,
        }
    if run_dir.exists() and any(run_dir.iterdir()):
        return {
            "action": "archive",
            "reason": "output directory is non-empty but has no readable last.pt",
        }
    return {"action": "start"}


def archive_stale_dir(run_dir: Path) -> Path:
    """Rename (never delete) a stale run directory and return the new path."""
    run_dir = Path(run_dir)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = run_dir.with_name(f"{run_dir.name}.stale-{stamp}")
    if target.exists():
        raise FileExistsError(f"Archive target already exists: {target}")
    run_dir.rename(target)
    return target


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------


def package_checkpoint(
    run_dir: Path,
    export_dir: Path,
    *,
    repo: Path,
    thresholds: dict | None = None,
    evaluation: dict | None = None,
) -> dict:
    """Copy the app checkpoint and small run artifacts into ``export_dir``."""
    run_dir = Path(run_dir)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    app_checkpoint = run_dir / "mvssnet_groupaware.pt"
    if not app_checkpoint.is_file():
        raise FileNotFoundError(f"App checkpoint not found: {app_checkpoint}")

    destination = export_dir / "mvssnet_groupaware.pt"
    shutil.copy2(app_checkpoint, destination)
    for name in ("run.json", "training_log.csv", "split_used.csv"):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, export_dir / name)

    state = read_training_state(run_dir) or {}
    git_commit = None
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip() or None
    except OSError:  # pragma: no cover - git may be absent
        git_commit = None

    manifest = {
        "artifact": "mvssnet_groupaware.pt",
        "file": destination.name,
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_run": str(run_dir),
        "resume_epoch": state.get("epoch"),
        "completed_epochs": state.get("completed_epochs"),
        "best_validation_mean_image_iou": state.get("best_iou"),
        "thresholds": thresholds or dict(EXPECTED_THRESHOLDS),
        "evaluation": evaluation or {},
        "git_commit": git_commit,
        "preprocessing": "BGR + ImageNet statistics, 512x512, identical to the Streamlit app",
        "checkpoint_kind": "plain state_dict (torch.save), loadable by MvssNetPredictor",
        "historical_checkpoint_untouched": CHECKPOINT_SHA256,
    }
    (export_dir / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def zip_artifacts(
    export_dir: Path,
    zip_path: Path,
    *,
    exclude_suffixes: Iterable[str] = (".pt", ".pth", ".npy"),
) -> dict:
    """Zip small artifacts, excluding weights by default."""
    export_dir = Path(export_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    excluded = tuple(exclude_suffixes)
    added = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(export_dir.rglob("*")):
            if not path.is_file() or path == zip_path:
                continue
            if excluded and path.suffix.lower() in excluded:
                continue
            archive.write(path, path.relative_to(export_dir).as_posix())
            added += 1
    return {
        "zip": str(zip_path),
        "files": added,
        "size_mb": round(zip_path.stat().st_size / (1024 * 1024), 2),
    }


def collect_run_summary(run_dir: Path, evaluation_path: Path | None = None) -> dict:
    """Read training and optional evaluation JSON into one comparison-friendly dict."""
    run_dir = Path(run_dir)
    summary: dict = {"run_dir": str(run_dir)}
    run_json = run_dir / "run.json"
    if run_json.is_file():
        payload = json.loads(run_json.read_text(encoding="utf-8"))
        summary.update(
            {
                "mode": payload.get("mode"),
                "samples": payload.get("samples"),
                "best_validation_mean_image_iou": payload.get("best_validation_mean_image_iou"),
                "final_validation": payload.get("final_validation"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "config": payload.get("config"),
            }
        )
    if evaluation_path is not None and Path(evaluation_path).is_file():
        evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
        summary["evaluation"] = {
            "split": evaluation.get("split"),
            "mode": evaluation.get("mode"),
            "pixel": evaluation.get("pixel_metrics_at_fixed_threshold"),
            "image": evaluation.get("image_level_metrics_at_fixed_threshold"),
            "samples": evaluation.get("samples"),
            "warnings": evaluation.get("warnings"),
        }
    return summary
