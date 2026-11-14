"""Tests for the Colab pipeline helpers.

These tests never train and never touch real datasets or checkpoints.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest
import torch

from notebooks.colab_pipeline import (
    EXPECTED_THRESHOLDS,
    _first_existing,
    archive_stale_dir,
    audit_manifest_frames,
    collect_run_summary,
    detect_dataset_root,
    package_checkpoint,
    plan_training_run,
    sha256_file,
    threshold_report,
    zip_artifacts,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def test_sha256_file_matches_known_digest(tmp_path: Path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"mvss")
    assert sha256_file(path) == "7847FEABD36C3FA75E15A7AE336BBC15EA69CD2B3CF5AC2CEBBDF574ACD1B35E"
    assert len(sha256_file(path)) == 64


# ---------------------------------------------------------------------------
# Dataset layout
# ---------------------------------------------------------------------------


def _touch_dataset(root: Path) -> None:
    (root / "datasets" / "CASIA2.0_revised" / "Tp").mkdir(parents=True)
    (root / "datasets" / "CASIA2.0_Groundtruth").mkdir(parents=True)


def test_detect_dataset_root_uses_existing_datasets_folder(tmp_path: Path):
    casia = tmp_path / "casia_data"
    _touch_dataset(casia)

    layout = detect_dataset_root(casia)

    assert layout["staged"] is False
    assert Path(layout["data_root"]) == casia.resolve()
    assert Path(layout["datasets_link"]).is_dir()


def test_detect_dataset_root_stages_flat_extraction(tmp_path: Path):
    casia = tmp_path / "casia_data"
    (casia / "CASIA2.0_revised" / "Tp").mkdir(parents=True)
    (casia / "CASIA2.0_Groundtruth").mkdir(parents=True)
    try:
        (tmp_path / "linktest").symlink_to(casia, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not permitted on this platform")

    layout = detect_dataset_root(casia)

    assert layout["staged"] is True
    assert (Path(layout["data_root"]) / "datasets" / "CASIA2.0_revised").is_dir()
    assert (Path(layout["data_root"]) / "datasets" / "CASIA2.0_Groundtruth").is_dir()


def test_detect_dataset_root_rejects_missing_folders(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        detect_dataset_root(tmp_path)


def test_detect_dataset_root_prefers_real_folders_inside_wrapper_extraction(tmp_path: Path):
    casia = tmp_path / "casia_data"
    nested_revised = casia / "CASIA2.0_revised" / "CASIA2.0_revised"
    (nested_revised / "Tp").mkdir(parents=True)
    (nested_revised / "Au").mkdir()
    nested_truth = casia / "CASIA2.0_Groundtruth" / "CASIA2.0_Groundtruth"
    nested_truth.mkdir(parents=True)
    (nested_truth / "example_gt.png").write_bytes(b"png")
    try:
        (tmp_path / "linktest").symlink_to(casia, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not permitted on this platform")

    layout = detect_dataset_root(casia)

    data_root = Path(layout["data_root"])
    assert (data_root / "datasets" / "CASIA2.0_revised" / "Tp").is_dir()
    assert (data_root / "datasets" / "CASIA2.0_Groundtruth" / "example_gt.png").is_file()


def test_first_existing_skips_a_name_only_wrapper(tmp_path: Path):
    outer = tmp_path / "CASIA2.0_revised"
    (outer / "docs").mkdir(parents=True)
    inner = outer / "CASIA2.0_revised"
    (inner / "Tp").mkdir(parents=True)
    (inner / "Au").mkdir()

    assert _first_existing(tmp_path, "CASIA2.0_revised") == inner


def test_first_existing_accepts_a_flat_revised_layout(tmp_path: Path):
    flat = tmp_path / "CASIA2.0_revised"
    (flat / "Tp").mkdir(parents=True)
    (flat / "Au").mkdir()
    assert _first_existing(tmp_path, "CASIA2.0_revised") == flat


def test_first_existing_finds_ground_truth_with_masks(tmp_path: Path):
    outer = tmp_path / "CASIA2.0_Groundtruth"
    outer.mkdir()
    inner = outer / "CASIA2.0_Groundtruth"
    inner.mkdir()
    (inner / "mask.png").write_bytes(b"png")

    assert _first_existing(tmp_path, "CASIA2.0_Groundtruth") == inner


# ---------------------------------------------------------------------------
# Manifest audit
# ---------------------------------------------------------------------------


def _frames():
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "t1",
                "class_name": "Tampered",
                "image_path": "datasets/img/t1.tif",
                "mask_path": "datasets/mask/t1_gt.png",
            },
            {
                "sample_id": "t2",
                "class_name": "Tampered",
                "image_path": "datasets/img/t2.tif",
                "mask_path": "datasets/mask/t2_gt.png",
            },
            {
                "sample_id": "a1",
                "class_name": "Authentic",
                "image_path": "datasets/img/a1.jpg",
                "mask_path": "",
            },
        ]
    )
    split = pd.DataFrame(
        [
            {
                "sample_id": "t1",
                "experimental_split": "train",
                "duplicate_component": "c1",
            },
            {
                "sample_id": "t2",
                "experimental_split": "test",
                "duplicate_component": "c1",
            },
            {
                "sample_id": "a1",
                "experimental_split": "validation",
                "duplicate_component": "c2",
            },
        ]
    )
    return manifest, split


def test_audit_manifest_frames_reports_counts_and_spanning_components():
    manifest, split = _frames()
    existing = {"datasets/img/t1.tif", "datasets/img/t2.tif", "datasets/img/a1.jpg"}

    audit = audit_manifest_frames(manifest, split, lambda rel: rel in existing)

    assert audit["matched_rows"] == 3
    assert audit["split_counts"] == {"train": 1, "test": 1, "validation": 1}
    assert audit["components"] == 2
    assert audit["components_spanning_splits"] == 1
    assert audit["missing_images"] == 0
    assert audit["tampered_without_mask"] == 2  # both masks absent
    assert audit["unmatched_split_rows"] == 0


def test_audit_manifest_frames_counts_missing_images():
    manifest, split = _frames()
    audit = audit_manifest_frames(manifest, split, lambda rel: rel.endswith("a1.jpg"))
    assert audit["missing_images"] == 2
    assert len(audit["missing_image_examples"]) == 2


def test_audit_manifest_frames_rejects_duplicate_ids():
    manifest, split = _frames()
    duplicated = pd.concat([manifest, manifest.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        audit_manifest_frames(duplicated, split, lambda rel: True)


def test_audit_manifest_frames_rejects_unknown_split():
    manifest, split = _frames()
    split.loc[0, "experimental_split"] = "holdout"
    with pytest.raises(ValueError, match="Unexpected experimental_split"):
        audit_manifest_frames(manifest, split, lambda rel: True)


# ---------------------------------------------------------------------------
# Resume planning
# ---------------------------------------------------------------------------


def _desired(epochs: int = 2) -> dict:
    return {
        "image_size": 512,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "seed": 42,
        "limit_train": 0,
        "limit_val": 0,
        "manifest": "results/manifests/casia2_manifest.csv",
        "split_csv": "results/manifests/casia2_groupaware_split.csv",
        "epochs": epochs,
    }


def _write_state(run_dir: Path, *, epoch: int, config: dict, app_checkpoint: bool = False) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {"weight": torch.zeros(1)},
            "optimizer_state": {},
            "scaler_state": None,
            "epoch": epoch,
            "best_iou": 0.5,
            "history": [{"epoch": 1}],
            "config": config,
        },
        run_dir / "last.pt",
    )
    (run_dir / "best.pt").write_bytes(b"placeholder")
    if app_checkpoint:
        (run_dir / "mvssnet_groupaware.pt").write_bytes(b"placeholder")


def test_plan_training_run_starts_when_empty(tmp_path: Path):
    plan = plan_training_run(tmp_path / "run", _desired())
    assert plan["action"] == "start"


def test_plan_training_run_archives_non_empty_directory_without_checkpoint(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "notes.txt").write_text("keep me", encoding="utf-8")

    plan = plan_training_run(run, _desired())
    assert plan["action"] == "archive"

    archived = archive_stale_dir(run)
    assert archived.is_dir()
    assert (archived / "notes.txt").is_file()
    assert not run.exists()


def test_plan_training_run_resumes_with_epochs_remaining(tmp_path: Path):
    run = tmp_path / "run"
    config = {**_desired(), "manifest": "/content/DigitalImageForensics/results/manifests/casia2_manifest.csv"}
    _write_state(run, epoch=0, config=config)

    plan = plan_training_run(run, _desired(epochs=3))
    assert plan["action"] == "resume"
    assert plan["next_epoch"] == 2
    assert Path(plan["resume"]).name == "last.pt"


def test_plan_training_run_completes_or_exports_only(tmp_path: Path):
    run = tmp_path / "run"
    _write_state(run, epoch=1, config=_desired())

    plan = plan_training_run(run, _desired(epochs=2))
    assert plan["action"] == "export_only"

    (run / "mvssnet_groupaware.pt").write_bytes(b"placeholder")
    plan = plan_training_run(run, _desired(epochs=2))
    assert plan["action"] == "complete"


def test_plan_training_run_detects_incompatible_config(tmp_path: Path):
    run = tmp_path / "run"
    config = {**_desired(), "image_size": 256}
    _write_state(run, epoch=0, config=config)

    plan = plan_training_run(run, _desired(epochs=3))
    assert plan["action"] == "incompatible"
    assert any("image_size" in reason for reason in plan["reasons"])


def test_plan_training_run_allows_batch_size_change_for_oom_recovery(tmp_path: Path):
    run = tmp_path / "run"
    config = {**_desired(), "batch_size": 8}
    _write_state(run, epoch=0, config=config)

    plan = plan_training_run(run, _desired(epochs=3))
    assert plan["action"] == "resume"
    assert any("batch_size" in change for change in plan["changed"])


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------


def test_package_checkpoint_and_zip_exclude_weights(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "mvssnet_groupaware.pt").write_bytes(b"weights")
    (run / "run.json").write_text(json.dumps({"mode": "full"}), encoding="utf-8")
    (run / "training_log.csv").write_text("epoch\n1\n", encoding="utf-8")
    torch.save(
        {
            "epoch": 2,
            "best_iou": 0.61,
            "history": [],
            "config": _desired(),
            "model_state": {},
            "optimizer_state": {},
            "scaler_state": None,
        },
        run / "last.pt",
    )
    export = tmp_path / "export"

    manifest = package_checkpoint(run, export, repo=tmp_path)
    assert manifest["resume_epoch"] == 2
    assert manifest["completed_epochs"] == 3
    assert manifest["best_validation_mean_image_iou"] == pytest.approx(0.61)
    assert (export / "checkpoint_manifest.json").is_file()
    assert len(manifest["sha256"]) == 64
    assert manifest["historical_checkpoint_untouched"].startswith("080BC6C3")

    result = zip_artifacts(export, tmp_path / "artifacts.zip")
    assert result["files"] >= 3
    import zipfile

    with zipfile.ZipFile(result["zip"]) as archive:
        names = set(archive.namelist())
    assert "checkpoint_manifest.json" in names
    assert "run.json" in names
    assert "mvssnet_groupaware.pt" not in names


def test_collect_run_summary_merges_training_and_evaluation(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(
        json.dumps({"mode": "full", "best_validation_mean_image_iou": 0.6}), encoding="utf-8"
    )
    evaluation = tmp_path / "eval.json"
    evaluation.write_text(
        json.dumps(
            {
                "split": "test",
                "mode": "full",
                "pixel_metrics_at_fixed_threshold": {"micro_iou": 0.4},
                "image_level_metrics_at_fixed_threshold": {"auc_threshold_free": 0.9},
                "samples": {"evaluated": 10},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    summary = collect_run_summary(run, evaluation)
    assert summary["best_validation_mean_image_iou"] == 0.6
    assert summary["evaluation"]["pixel"]["micro_iou"] == 0.4


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


def test_threshold_report_matches_deployed_values():
    repo = Path(__file__).resolve().parents[1]
    report = threshold_report(repo)
    assert report["actual"] == {
        "pixel": EXPECTED_THRESHOLDS["pixel"],
        "image": EXPECTED_THRESHOLDS["image"],
        "uncertainty": EXPECTED_THRESHOLDS["uncertainty"],
    }
    assert report["consistent"] is True
    assert report["pixel_source"]
    assert report["image_source"]
