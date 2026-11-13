"""Tests for the controlled synthetic chat-screenshot dataset."""

from __future__ import annotations

import json

import cv2
import numpy as np

from evaluation.build_synthetic_chat_dataset import build_dataset


def test_synthetic_dataset_is_valid_and_grounded(tmp_path):
    root = tmp_path / "chat_synthetic"
    result = build_dataset(root, conversations_count=3, seed=7, force=False)

    assert result["conversations"] == 3
    assert result["records"] == 6
    assert result["validation"]["class_counts"] == {"Authentic": 3, "Tampered": 3}
    assert result["validation"]["split_counts"] == {"test": 6}

    annotations = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    assert len(annotations["conversations"]) == 3
    assert "fictional" in annotations["privacy"].lower()

    for conversation in annotations["conversations"]:
        authentic = cv2.imread(str(root / conversation["authentic_image"]))
        edited = cv2.imread(str(root / conversation["manipulated_image"]))
        mask = cv2.imread(str(root / conversation["mask"]), cv2.IMREAD_GRAYSCALE)
        assert authentic is not None and edited is not None and mask is not None
        assert mask.shape == edited.shape[:2]
        assert int((mask > 0).sum()) > 0
        assert np.any(authentic != edited)
        assert conversation["changes"], "each edited screenshot must record its edits"
        for change in conversation["changes"]:
            assert change["kind"] in {
                "name_or_header",
                "amount_or_number",
                "message_text",
                "date_or_timestamp",
                "time",
            }
            assert change["original"] != change["edited"]
            x1, y1, x2, y2 = change["bbox"]
            assert mask[y1:y2, x1:x2].max() == 255

    manifest = (root / "manifest.csv").read_text(encoding="utf-8")
    assert "chat_synthetic_v1" in manifest
    assert "CC0-1.0" in manifest


def test_synthetic_dataset_refuses_overwrite_without_force(tmp_path):
    root = tmp_path / "chat_synthetic"
    build_dataset(root, conversations_count=1, seed=7, force=False)
    try:
        build_dataset(root, conversations_count=1, seed=7, force=False)
    except SystemExit as exc:
        assert "Refusing to overwrite" in str(exc)
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("expected SystemExit when regenerating without --force")
