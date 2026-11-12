"""CASIA 2.0 manifest adapter for the existing preserved split.

The adapter reads, but never rewrites, these preserved artefacts:

* ``experiments/dataset_split.csv``
* ``experiments/localization_pairs.csv``
* ``datasets/CASIA2.0_revised/{Au,Tp}``
* ``datasets/CASIA2.0_Groundtruth``

Every split row produces one manifest row.  Missing images and masks are
reported and retained as invalid rows; they are never silently omitted.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import pandas as pd
from PIL import Image

from .schema import (
    compute_image_hashes,
    normalise_record,
    resolve_manifest_path,
    to_posix_relative,
    validate_manifest,
)


@dataclass(frozen=True)
class CASIABuildConfig:
    data_root: Path
    split_csv: Path
    pairs_csv: Path
    dataset_dir: Path
    groundtruth_dir: Path

    @classmethod
    def from_data_root(cls, data_root: str | Path) -> "CASIABuildConfig":
        root = Path(data_root).expanduser().resolve()
        return cls(
            data_root=root,
            split_csv=root / "experiments" / "dataset_split.csv",
            pairs_csv=root / "experiments" / "localization_pairs.csv",
            dataset_dir=root / "datasets" / "CASIA2.0_revised",
            groundtruth_dir=root / "datasets" / "CASIA2.0_Groundtruth",
        )


@dataclass
class CASIABuildReport:
    dataset: str = "CASIA2.0"
    generated_utc: str = ""
    counts: dict[str, Any] = field(default_factory=dict)
    split_class_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    source_group_counts: dict[str, int] = field(default_factory=dict)
    missing_images: list[dict[str, str]] = field(default_factory=list)
    missing_masks: list[dict[str, str]] = field(default_factory=list)
    unreadable_masks: list[dict[str, str]] = field(default_factory=list)
    duplicate_filenames: list[dict[str, Any]] = field(default_factory=list)
    duplicate_sample_ids: list[str] = field(default_factory=list)
    ambiguous_mask_mappings: list[dict[str, Any]] = field(default_factory=list)
    rows_with_issues: list[dict[str, str]] = field(default_factory=list)
    validation_error_count: int = 0
    validation_warning_count: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "generated_utc": self.generated_utc,
            "counts": dict(self.counts),
            "split_class_counts": {
                split: dict(counts) for split, counts in self.split_class_counts.items()
            },
            "source_group_counts": dict(self.source_group_counts),
            "missing_images": list(self.missing_images),
            "missing_masks": list(self.missing_masks),
            "unreadable_masks": list(self.unreadable_masks),
            "duplicate_filenames": list(self.duplicate_filenames),
            "duplicate_sample_ids": list(self.duplicate_sample_ids),
            "ambiguous_mask_mappings": list(self.ambiguous_mask_mappings),
            "rows_with_issues": list(self.rows_with_issues),
            "validation_error_count": int(self.validation_error_count),
            "validation_warning_count": int(self.validation_warning_count),
            "notes": list(self.notes),
        }


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return cleaned or "sample"


def _sample_id(class_name: str, filename: str) -> str:
    digest = hashlib.sha1(f"{class_name}|{filename}".encode("utf-8")).hexdigest()[:12]
    return f"casia2-{_slug(class_name)}-{digest}"


def _load_pairs(pairs_csv: Path) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    pairs = pd.read_csv(pairs_csv)
    missing_columns = {"image_path", "mask_path"} - set(pairs.columns)
    if missing_columns:
        raise ValueError(f"{pairs_csv} is missing columns: {sorted(missing_columns)}")

    mapping: dict[str, list[str]] = defaultdict(list)
    ambiguous: list[dict[str, Any]] = []
    for row in pairs.itertuples():
        image_path = Path(str(row.image_path).replace("\\", "/"))
        mask_path = Path(str(row.mask_path).replace("\\", "/"))
        mapping[image_path.name].append(mask_path.as_posix())
    for filename, masks in mapping.items():
        unique = sorted(set(masks))
        if len(unique) > 1:
            ambiguous.append(
                {
                    "filename": filename,
                    "mask_paths": unique,
                    "reason": "one tampered filename maps to multiple masks",
                }
            )
    return {name: sorted(set(paths)) for name, paths in mapping.items()}, ambiguous


def _quick_image_info(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        width, height = image.size
        fmt = (image.format or path.suffix.lstrip(".")).lower()
    return int(width), int(height), fmt


def _mask_info(path: Path) -> tuple[bool, str, int, int]:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False, "OpenCV could not decode the mask", 0, 0
    return True, "", int(mask.shape[1]), int(mask.shape[0])


def build_casia_manifest(
    config: CASIABuildConfig,
    *,
    compute_hashes: bool = True,
    include_decoded_hash: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[dict[str, Any]], CASIABuildReport]:
    """Build a manifest-compatible representation of the preserved CASIA split."""

    if not config.split_csv.exists():
        raise FileNotFoundError(config.split_csv)
    if not config.pairs_csv.exists():
        raise FileNotFoundError(config.pairs_csv)

    split_frame = pd.read_csv(config.split_csv)
    required = {"filename", "class_name", "split"}
    missing_columns = required - set(split_frame.columns)
    if missing_columns:
        raise ValueError(f"{config.split_csv} is missing columns: {sorted(missing_columns)}")

    mask_mapping, ambiguous = _load_pairs(config.pairs_csv)
    report = CASIABuildReport(
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ambiguous_mask_mappings=ambiguous,
    )

    filename_counts = Counter(str(value) for value in split_frame["filename"])
    report.duplicate_filenames = [
        {"filename": filename, "occurrences": int(count)}
        for filename, count in sorted(filename_counts.items())
        if count > 1
    ]

    records: list[dict[str, Any]] = []
    used_sample_ids: dict[str, int] = {}
    split_class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_counts: Counter[str] = Counter()

    total = len(split_frame)
    for index, row in enumerate(split_frame.itertuples(index=False), start=1):
        filename = str(getattr(row, "filename", "")).strip()
        class_name = str(getattr(row, "class_name", "")).strip()
        split = str(getattr(row, "split", "")).strip()
        notes: list[str] = []
        row_errors: list[str] = []

        if not filename:
            row_errors.append("split row has an empty filename")
        if class_name not in {"Authentic", "Tampered", "Unknown"}:
            row_errors.append(f"unsupported class_name {class_name!r}")
        if split not in {"train", "validation", "test", "unassigned"}:
            row_errors.append(f"unsupported split {split!r}")

        base_sample_id = _sample_id(class_name or "Unknown", filename or f"row-{index}")
        sample_id = base_sample_id
        if sample_id in used_sample_ids:
            used_sample_ids[sample_id] += 1
            sample_id = f"{base_sample_id}-{used_sample_ids[base_sample_id]:02d}"
            report.duplicate_sample_ids.append(sample_id)
            row_errors.append(f"duplicate filename/class generated a suffixed sample_id: {sample_id}")
        else:
            used_sample_ids[sample_id] = 1

        if class_name == "Authentic":
            image_path = config.dataset_dir / "Au" / filename
            manifest_image_path = to_posix_relative(image_path, config.data_root)
            mask_abs: Path | None = None
            mask_manifest_path = ""
            pair_id = ""
            group_id = f"casia2:au:{base_sample_id}"
            notes.append("Authentic CASIA2.0 image; no manipulation mask.")
        elif class_name == "Tampered":
            image_path = config.dataset_dir / "Tp" / filename
            manifest_image_path = to_posix_relative(image_path, config.data_root)
            candidates = mask_mapping.get(filename, [])
            mask_manifest_path = ""
            mask_abs = None
            if len(candidates) == 1:
                mask_rel = candidates[0]
                mask_abs = resolve_manifest_path(mask_rel, config.data_root)
                mask_manifest_path = to_posix_relative(
                    mask_abs if mask_abs is not None else Path(mask_rel), config.data_root
                )
            elif len(candidates) > 1:
                row_errors.append(
                    "ambiguous mask mapping; no mask selected: " + ", ".join(candidates)
                )
            else:
                row_errors.append("no ground-truth mask mapping found in localization_pairs.csv")
            pair_id = f"casia2:pair:{base_sample_id}"
            group_id = f"casia2:tp:{base_sample_id}"
            notes.append(
                "Tampered CASIA2.0 composite; mask is linked by pair_id and group_id."
            )
        else:
            image_path = config.dataset_dir / "Unknown" / filename
            manifest_image_path = to_posix_relative(image_path, config.data_root)
            mask_abs = None
            mask_manifest_path = ""
            pair_id = ""
            group_id = f"casia2:unknown:{base_sample_id}"
            notes.append("Unknown class; retained for traceability, excluded from training/evaluation.")

        if class_name == "Tampered":
            code_parts = Path(filename).stem.split("_")
            if len(code_parts) >= 5:
                notes.append("CASIA filename family code: " + "_".join(code_parts[1:5]))
            notes.append("Source camera/device is not documented in the split metadata.")

        width = height = 0
        image_format = image_path.suffix.lower().lstrip(".")
        hashes: dict[str, Any] = {}
        image_ok = image_path.exists()
        if not image_ok:
            row_errors.append(f"image not found: {manifest_image_path}")
            report.missing_images.append(
                {"sample_id": sample_id, "image_path": manifest_image_path}
            )
        else:
            try:
                if compute_hashes:
                    hashes = compute_image_hashes(image_path, include_decoded=include_decoded_hash)
                    width = int(hashes.pop("width"))
                    height = int(hashes.pop("height"))
                    image_format = str(hashes.pop("format", image_format) or image_format)
                    hashes.pop("readable", None)
                else:
                    width, height, image_format = _quick_image_info(image_path)
            except Exception as exc:  # noqa: BLE001 - retain row with explicit error
                image_ok = False
                row_errors.append(f"image could not be decoded: {exc}")
                report.missing_images.append(
                    {
                        "sample_id": sample_id,
                        "image_path": manifest_image_path,
                        "reason": f"unreadable: {exc}",
                    }
                )

        mask_valid = False
        mask_type = ""
        mask_quality = ""
        mask_width = mask_height = 0
        if class_name == "Tampered" and mask_abs is not None:
            if not mask_abs.exists():
                row_errors.append(f"mask not found: {mask_manifest_path}")
                report.missing_masks.append(
                    {"sample_id": sample_id, "mask_path": mask_manifest_path}
                )
            else:
                mask_valid, detail, mask_width, mask_height = _mask_info(mask_abs)
                if mask_valid:
                    mask_type = "binary_png" if mask_abs.suffix.lower() == ".png" else (
                        mask_abs.suffix.lower().lstrip(".") or "unknown"
                    )
                    mask_quality = "groundtruth"
                else:
                    row_errors.append(f"mask could not be decoded: {detail}")
                    report.unreadable_masks.append(
                        {
                            "sample_id": sample_id,
                            "mask_path": mask_manifest_path,
                            "reason": detail,
                        }
                    )
        elif class_name == "Tampered":
            report.missing_masks.append(
                {"sample_id": sample_id, "mask_path": "", "reason": "; ".join(row_errors)}
            )

        if class_name == "Tampered" and not mask_valid:
            notes.append("No valid mask: this row cannot provide pixel-level supervision.")
        if mask_valid and width and height and (mask_width, mask_height) != (width, height):
            notes.append(
                f"Mask size {mask_width}x{mask_height} differs from image size {width}x{height}; evaluation resizes explicitly."
            )

        pixel_training_eligible = bool(
            class_name == "Tampered" and mask_valid and split == "train"
        )
        validation_errors = "; ".join(dict.fromkeys(row_errors))
        if not image_ok or (
            class_name == "Tampered" and not mask_valid and split == "train"
        ):
            validation_status = "invalid"
        elif row_errors:
            validation_status = "warning"
        else:
            validation_status = "ok"

        record = normalise_record(
            {
                "sample_id": sample_id,
                "dataset": "CASIA2.0",
                "source_group": "unknown",
                "split": split,
                "class_name": class_name,
                "image_path": manifest_image_path,
                "mask_path": mask_manifest_path if mask_valid else "",
                "pair_id": pair_id,
                "group_id": group_id,
                "manipulation_type": "unspecified" if class_name == "Tampered" else "",
                "device_make": "",
                "device_model": "",
                "capture_type": "unknown",
                "width": width,
                "height": height,
                "format": image_format,
                "mask_available": mask_valid,
                "mask_type": mask_type,
                "mask_quality": mask_quality,
                "license": "CASIA 2.0 research terms; verify before redistribution",
                "provenance_url": "",
                "sha256": hashes.get("sha256", ""),
                "phash": hashes.get("phash", ""),
                "notes": " ".join(dict.fromkeys(notes)),
                "dhash": hashes.get("dhash", ""),
                "decoded_sha256": hashes.get("decoded_sha256", ""),
                "file_size": hashes.get("file_size", ""),
                "validation_status": validation_status,
                "validation_errors": validation_errors,
                "pixel_training_eligible": pixel_training_eligible,
            }
        )
        records.append(record)

        split_class_counts[split][class_name] += 1
        source_counts[record["source_group"]] += 1
        if validation_errors:
            report.rows_with_issues.append(
                {"sample_id": sample_id, "status": validation_status, "issues": validation_errors}
            )
        if progress is not None:
            progress(index, total, sample_id)

    validation = validate_manifest(
        records,
        data_root=config.data_root,
        strict=False,
        check_paths=True,
    )
    report.validation_error_count = len(validation.errors)
    report.validation_warning_count = len(validation.warnings)
    report.counts = dict(validation.summary)
    report.split_class_counts = {
        split: dict(sorted(counts.items()))
        for split, counts in sorted(split_class_counts.items())
    }
    report.source_group_counts = dict(sorted(source_counts.items()))
    report.notes = [
        "Original split CSV, pair CSV, images and masks were read only.",
        "Sample IDs are globally unique; tampered image and mask share one group.",
        "Source camera metadata is not available in the preserved CASIA metadata, so source_group is unknown.",
        "The BGR/RGB preprocessing inconsistency is out of scope for Phase 0.",
    ]
    return records, report
