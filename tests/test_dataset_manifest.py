from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dataset_manifest.casia import CASIABuildConfig, build_casia_manifest
from dataset_manifest.io import load_manifest, write_manifest
from dataset_manifest.schema import (
    ALL_FIELDS,
    compute_image_hashes,
    hamming_distance,
    normalise_record,
    normalise_source_group,
    validate_manifest,
)


def _image(path: Path, value: int = 40) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((48, 64, 3), dtype=np.uint8)
    array[..., 0] = value
    array[..., 1] = np.arange(64, dtype=np.uint8)[None, :]
    array[10:30, 20:45] = (200, 80, 40)
    Image.fromarray(array).save(path)
    return path


def _mask(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((48, 64), dtype=np.uint8)
    array[10:30, 20:45] = 255
    Image.fromarray(array).save(path)
    return path


def _record(image: Path, *, sample_id: str = "s1", mask: Path | None = None, **overrides):
    payload = {
        "sample_id": sample_id,
        "dataset": "synthetic",
        "source_group": "android",
        "split": "test",
        "class_name": "Tampered" if mask else "Authentic",
        "image_path": image.name,
        "mask_path": mask.name if mask else "",
        "pair_id": f"pair-{sample_id}" if mask else "",
        "group_id": f"group-{sample_id}",
        "manipulation_type": "synthetic" if mask else "",
        "device_make": "Test",
        "device_model": "Synthetic",
        "capture_type": "smartphone",
        "width": 64,
        "height": 48,
        "format": "png",
        "mask_available": bool(mask),
        "mask_type": "binary_png" if mask else "",
        "mask_quality": "synthetic" if mask else "",
        "license": "test-only",
        "provenance_url": "",
        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "phash": "0" * 16,
        "notes": "synthetic fixture; not evidence of real-world performance",
    }
    payload.update(overrides)
    return normalise_record(payload)


def test_manifest_round_trip_preserves_types_and_paths(tmp_path):
    image = _image(tmp_path / "img.png")
    mask = _mask(tmp_path / "mask.png")
    record = _record(image, mask=mask)
    record["pixel_training_eligible"] = True

    for suffix in (".csv", ".jsonl"):
        path = tmp_path / f"manifest{suffix}"
        write_manifest([record], path)
        loaded = load_manifest(path)
        assert len(loaded) == 1
        assert loaded[0]["sample_id"] == "s1"
        assert loaded[0]["mask_available"] is True
        assert loaded[0]["pixel_training_eligible"] is True
        assert loaded[0]["width"] == 64
        assert loaded[0]["image_path"] == "img.png"


def test_validation_accepts_complete_manifest(tmp_path):
    image = _image(tmp_path / "img.png")
    mask = _mask(tmp_path / "mask.png")
    record = _record(image, mask=mask)
    result = validate_manifest([record], data_root=tmp_path, strict=False)
    assert result.ok, result.as_dict()
    assert result.summary["rows"] == 1
    assert result.summary["mask_available"] == 1


def test_validation_rejects_duplicate_sample_ids(tmp_path):
    image = _image(tmp_path / "img.png")
    result = validate_manifest(
        [_record(image), _record(image)],
        data_root=tmp_path,
    )
    assert not result.ok
    assert any(issue.code == "duplicate_sample_id" for issue in result.errors)


def test_validation_detects_group_split_leakage(tmp_path):
    image = _image(tmp_path / "img.png")
    first = _record(image, sample_id="s1")
    second = _record(image, sample_id="s2", split="train")
    second["group_id"] = first["group_id"]
    result = validate_manifest([first, second], data_root=tmp_path)
    assert not result.ok
    assert any(issue.code == "group_split_leakage" for issue in result.errors)


def test_validation_reports_missing_image_and_mask(tmp_path):
    image = _image(tmp_path / "img.png")
    mask = _mask(tmp_path / "mask.png")
    missing_image = _record(image, sample_id="missing-image")
    missing_image["image_path"] = "does-not-exist.png"
    missing_mask = _record(image, sample_id="missing-mask", mask=mask)
    missing_mask["mask_path"] = "does-not-exist.png"

    result = validate_manifest([missing_image, missing_mask], data_root=tmp_path)
    codes = {issue.code for issue in result.errors}
    assert "missing_image" in codes
    assert "missing_mask" in codes


def test_tampered_training_row_without_mask_is_an_error(tmp_path):
    image = _image(tmp_path / "img.png")
    record = _record(image, sample_id="s1")
    record["class_name"] = "Tampered"
    record["split"] = "train"
    record["mask_available"] = False
    result = validate_manifest([record], data_root=tmp_path)
    assert not result.ok
    assert any(issue.code == "missing_mask_for_pixel_training" for issue in result.errors)


def test_tampered_test_row_without_mask_is_retained_and_warned(tmp_path):
    image = _image(tmp_path / "img.png")
    record = _record(image, sample_id="s1")
    record["class_name"] = "Tampered"
    record["split"] = "test"
    record["mask_available"] = False
    result = validate_manifest([record], data_root=tmp_path)
    assert result.ok
    assert any(issue.code == "missing_mask_for_evaluation" for issue in result.warnings)


def test_source_group_aliases_are_canonicalised():
    assert normalise_source_group("DSLR") == "dslr_mirrorless"
    assert normalise_source_group("mirrorless") == "dslr_mirrorless"
    assert normalise_source_group("ios") == "iphone"
    assert normalise_source_group("smartphone other") == "other_mobile"
    assert normalise_source_group("") == "unknown"
    with pytest.raises(ValueError):
        normalise_source_group("toaster")


def test_image_hashes_are_stable_and_sensitive_to_content(tmp_path):
    first = _image(tmp_path / "a.png")
    second = _image(tmp_path / "b.png", value=41)
    hashes_a = compute_image_hashes(first)
    hashes_a_again = compute_image_hashes(first)
    hashes_b = compute_image_hashes(second)
    assert hashes_a["sha256"] == hashes_a_again["sha256"]
    assert hashes_a["decoded_sha256"] == hashes_a_again["decoded_sha256"]
    assert hashes_a["phash"] == hashes_a_again["phash"]
    assert hashes_a["sha256"] != hashes_b["sha256"]
    assert hashes_a["decoded_sha256"] != hashes_b["decoded_sha256"]
    assert hamming_distance("0000000000000000", "0000000000000003") == 2


def test_casia_adapter_preserves_rows_and_flags_missing_masks(tmp_path):
    root = tmp_path / "data"
    split_csv = root / "experiments" / "dataset_split.csv"
    pairs_csv = root / "experiments" / "localization_pairs.csv"
    dataset_dir = root / "datasets" / "CASIA2.0_revised"
    gt_dir = root / "datasets" / "CASIA2.0_Groundtruth"
    split_csv.parent.mkdir(parents=True)
    dataset_dir.joinpath("Au").mkdir(parents=True)
    dataset_dir.joinpath("Tp").mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    _image(dataset_dir / "Au" / "Au_nat_00001.jpg")
    _image(dataset_dir / "Tp" / "Tp_D_CND_M_N_ani00018_sec00096_00138.tif")
    _image(dataset_dir / "Tp" / "Tp_D_CNN_M_N_sec00011_cha00085_11227.jpg")
    _mask(gt_dir / "Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png")

    rows = [
        {"filename": "Au_nat_00001.jpg", "class": 0, "class_name": "Authentic", "split": "test"},
        {
            "filename": "Tp_D_CND_M_N_ani00018_sec00096_00138.tif",
            "class": 1,
            "class_name": "Tampered",
            "split": "validation",
        },
        {
            "filename": "Tp_D_CNN_M_N_sec00011_cha00085_11227.jpg",
            "class": 1,
            "class_name": "Tampered",
            "split": "test",
        },
    ]
    with split_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "class", "class_name", "split"])
        writer.writeheader()
        writer.writerows(rows)
    with pairs_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "mask_path"])
        writer.writeheader()
        writer.writerow(
            {
                "image_path": "datasets/CASIA2.0_revised/Tp/Tp_D_CND_M_N_ani00018_sec00096_00138.tif",
                "mask_path": "datasets/CASIA2.0_Groundtruth/Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png",
            }
        )

    before = {
        "split": hashlib.sha256(split_csv.read_bytes()).hexdigest(),
        "pairs": hashlib.sha256(pairs_csv.read_bytes()).hexdigest(),
        "image": hashlib.sha256(
            (dataset_dir / "Tp" / "Tp_D_CND_M_N_ani00018_sec00096_00138.tif").read_bytes()
        ).hexdigest(),
    }
    config = CASIABuildConfig(
        data_root=root,
        split_csv=split_csv,
        pairs_csv=pairs_csv,
        dataset_dir=dataset_dir,
        groundtruth_dir=gt_dir,
    )
    records, report = build_casia_manifest(config, compute_hashes=True)
    assert len(records) == 3
    assert len({record["sample_id"] for record in records}) == 3
    assert report.counts["rows"] == 3
    assert report.missing_masks
    assert report.missing_masks[0]["sample_id"] == records[2]["sample_id"]
    assert records[0]["class_name"] == "Authentic"
    assert records[0]["mask_available"] is False
    assert records[1]["mask_available"] is True
    assert records[1]["group_id"] == records[1]["group_id"]
    assert records[1]["pair_id"]
    assert records[1]["sha256"]
    assert records[1]["phash"]
    assert records[2]["validation_status"] == "warning"
    assert records[2]["pixel_training_eligible"] is False
    assert report.duplicate_filenames == []

    after = {
        "split": hashlib.sha256(split_csv.read_bytes()).hexdigest(),
        "pairs": hashlib.sha256(pairs_csv.read_bytes()).hexdigest(),
        "image": hashlib.sha256(
            (dataset_dir / "Tp" / "Tp_D_CND_M_N_ani00018_sec00096_00138.tif").read_bytes()
        ).hexdigest(),
    }
    assert before == after


def test_casia_adapter_never_silently_drops_duplicate_filename_rows(tmp_path):
    root = tmp_path / "data"
    split_csv = root / "experiments" / "dataset_split.csv"
    pairs_csv = root / "experiments" / "localization_pairs.csv"
    dataset_dir = root / "datasets" / "CASIA2.0_revised"
    gt_dir = root / "datasets" / "CASIA2.0_Groundtruth"
    split_csv.parent.mkdir(parents=True)
    dataset_dir.joinpath("Au").mkdir(parents=True)
    dataset_dir.joinpath("Tp").mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    _image(dataset_dir / "Au" / "same.jpg")
    with split_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "class", "class_name", "split"])
        writer.writeheader()
        writer.writerows(
            [
                {"filename": "same.jpg", "class": 0, "class_name": "Authentic", "split": "train"},
                {"filename": "same.jpg", "class": 0, "class_name": "Authentic", "split": "test"},
            ]
        )
    with pairs_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "mask_path"])
        writer.writeheader()

    records, report = build_casia_manifest(
        CASIABuildConfig(root, split_csv, pairs_csv, dataset_dir, gt_dir),
        compute_hashes=False,
    )
    assert len(records) == 2
    assert len({record["sample_id"] for record in records}) == 2
    assert report.duplicate_filenames == [{"filename": "same.jpg", "occurrences": 2}]
    validation = validate_manifest(records, data_root=root, check_paths=True)
    assert any(issue.code == "group_split_leakage" for issue in validation.errors)
