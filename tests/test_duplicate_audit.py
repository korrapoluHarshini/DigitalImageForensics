from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from dataset_manifest.duplicates import audit_manifest, geometric_match
from dataset_manifest.schema import compute_image_hashes, normalise_record


def _texture(path: Path, seed: int, size: int = 384) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    # Broad structure makes ORB and perceptual hashes stable.
    array[: size // 3, : size // 3] = (235, 40, 25)
    array[-size // 4 :, -size // 3 :] = (20, 180, 245)
    Image.fromarray(array).save(path)
    return path


def _record(path: Path, sample_id: str, *, dataset="synthetic", group=None, pair=None, class_name="Authentic"):
    hashes = compute_image_hashes(path, include_decoded=True)
    return normalise_record(
        {
            "sample_id": sample_id,
            "dataset": dataset,
            "source_group": "unknown",
            "split": "test",
            "class_name": class_name,
            "image_path": str(path),
            "mask_path": "",
            "pair_id": pair or "",
            "group_id": group or f"group-{sample_id}",
            "manipulation_type": "",
            "device_make": "",
            "device_model": "",
            "capture_type": "unknown",
            "width": hashes["width"],
            "height": hashes["height"],
            "format": hashes["format"],
            "mask_available": False,
            "mask_type": "",
            "mask_quality": "",
            "license": "test-only",
            "provenance_url": "",
            "sha256": hashes["sha256"],
            "phash": hashes["phash"],
            "notes": "synthetic fixture",
            "dhash": hashes["dhash"],
            "decoded_sha256": hashes["decoded_sha256"],
            "file_size": hashes["file_size"],
            "pixel_training_eligible": False,
        }
    )


def _hashes_before(directory: Path) -> dict[str, tuple[str, int]]:
    payload = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            payload[str(path)] = (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_mtime_ns,
            )
    return payload


def test_exact_and_decoded_duplicates_are_detected_without_modifying_files(tmp_path):
    original = _texture(tmp_path / "original.png", seed=1)
    exact = tmp_path / "exact.png"
    exact.write_bytes(original.read_bytes())
    recoded = tmp_path / "recoded.png"
    Image.open(original).save(recoded, format="PNG")

    records = [
        _record(original, "s-original"),
        _record(exact, "s-exact", group="group-exact"),
        _record(recoded, "s-recoded", group="group-exact"),
    ]
    before = _hashes_before(tmp_path)
    result = audit_manifest(records, phash_distance=6, dhash_distance=8)
    after = _hashes_before(tmp_path)

    assert before == after
    assert any({"s-original", "s-exact"} <= set(group.sample_ids) for group in result.exact_groups)
    assert any({"s-original", "s-exact", "s-recoded"} <= set(group.sample_ids) for group in result.decoded_groups)
    assert result.summary["exact_duplicate_groups"] >= 1
    assert result.summary["decoded_duplicate_groups"] >= 1
    assert any(
        relationship.relationship == "explicit_shared_group"
        for relationship in result.relationships
    )


def test_perceptual_duplicate_survives_reencoding(tmp_path):
    original = _texture(tmp_path / "original.png", seed=2)
    degraded = tmp_path / "degraded.jpg"
    Image.open(original).save(degraded, quality=60)

    records = [_record(original, "s-original"), _record(degraded, "s-degraded")]
    result = audit_manifest(records, phash_distance=6, dhash_distance=8)
    assert result.summary["exact_duplicate_groups"] == 0
    assert result.summary["decoded_duplicate_groups"] == 0
    assert result.summary["phash_groups"] >= 1
    assert result.summary["dhash_groups"] >= 1
    assert "s-original" in result.phash_groups[0].sample_ids


def test_geometric_duplicate_and_original_manipulated_relationship(tmp_path):
    original = _texture(tmp_path / "original.png", seed=3)
    rotated = tmp_path / "rotated.png"
    Image.fromarray(np.rot90(np.asarray(Image.open(original)), 1).copy()).save(rotated)

    records = [
        _record(original, "s-authentic", class_name="Authentic"),
        _record(rotated, "s-tampered", class_name="Tampered"),
    ]
    match = geometric_match(original, rotated)
    assert match is not None
    assert match["inliers"] >= 16

    result = audit_manifest(records, run_geometric=True, geometric_max_pairs=20)
    assert result.summary["geometric_groups"] == 1
    assert any(
        relationship.relationship == "geometric_original_manipulated"
        for relationship in result.relationships
    )


def test_cross_dataset_duplicate_is_reported(tmp_path):
    original = _texture(tmp_path / "dataset_a.png", seed=4)
    copied = tmp_path / "dataset_b.png"
    copied.write_bytes(original.read_bytes())
    records = [
        _record(original, "s-a", dataset="dataset-a"),
        _record(copied, "s-b", dataset="dataset-b"),
    ]
    result = audit_manifest(records)
    assert result.summary["cross_dataset_groups"] >= 1
    assert any(
        group["datasets"] == ["dataset-a", "dataset-b"]
        for group in result.cross_dataset_groups
    )


def test_duplicate_across_splits_is_reported_as_leakage(tmp_path):
    original = _texture(tmp_path / "train.png", seed=6)
    copied = tmp_path / "test.png"
    copied.write_bytes(original.read_bytes())
    train_record = _record(original, "s-train")
    test_record = _record(copied, "s-test")
    train_record["split"] = "train"
    result = audit_manifest([train_record, test_record])
    assert result.summary["split_leakage_groups"] >= 1
    assert result.split_leakage_groups[0]["splits"] == ["test", "train"]


def test_hash_candidate_generation_is_exact_for_small_sets():
    from dataset_manifest.duplicates import _candidate_pairs

    values = [0, 1, 3, 0xFFFFFFFFFFFFFFFF]
    pairs = _candidate_pairs(values, 2)
    assert pairs == {(0, 1), (0, 2), (1, 2)}


def test_audit_does_not_require_hash_fields(tmp_path):
    original = _texture(tmp_path / "original.png", seed=5)
    copy = tmp_path / "copy.png"
    copy.write_bytes(original.read_bytes())
    first = _record(original, "s-a")
    second = _record(copy, "s-b")
    for record in (first, second):
        for field in ("sha256", "phash", "dhash", "decoded_sha256"):
            record[field] = ""
    result = audit_manifest([first, second], compute_missing_hashes=True)
    assert result.summary["exact_duplicate_groups"] == 1
    assert result.summary["issues"] == 0
