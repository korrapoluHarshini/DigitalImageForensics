"""Unified dataset-manifest schema, path checks and image fingerprints.

The manifest is intentionally a plain table (CSV or JSON Lines) so that new
datasets can be integrated without changing the application.  The schema is
additive: all fields required by the Phase 0 specification are present, plus a
small number of derived diagnostic fields.

Definitions
-----------
sample_id
    Globally unique identifier for one row.  Never derive an evaluation label
    from a filename alone.
group_id
    Identifier shared by an original, its manipulated versions, masks and
    derived variants.  Rows in one group must not cross train/validation/test
    boundaries.
pair_id
    Optional identifier linking a tampered image to its mask or source pair.
source_group
    One of ``dslr_mirrorless``, ``android``, ``iphone``, ``other_mobile`` or
    ``unknown``.  Combined camera/mobile reports are derived at evaluation time,
    not stored as source groups.
class_name
    ``Authentic``, ``Tampered`` or ``Unknown``.  Phone photographs without
    independently established authenticity must be ``Unknown``.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps


MANIFEST_FIELDS: tuple[str, ...] = (
    "sample_id",
    "dataset",
    "source_group",
    "split",
    "class_name",
    "image_path",
    "mask_path",
    "pair_id",
    "group_id",
    "manipulation_type",
    "device_make",
    "device_model",
    "capture_type",
    "width",
    "height",
    "format",
    "mask_available",
    "mask_type",
    "mask_quality",
    "license",
    "provenance_url",
    "sha256",
    "phash",
    "notes",
)

OPTIONAL_FIELDS: tuple[str, ...] = (
    "dhash",
    "decoded_sha256",
    "file_size",
    "validation_status",
    "validation_errors",
    "pixel_training_eligible",
)

ALL_FIELDS: tuple[str, ...] = MANIFEST_FIELDS + OPTIONAL_FIELDS

BASE_SOURCE_GROUPS: tuple[str, ...] = (
    "dslr_mirrorless",
    "android",
    "iphone",
    "other_mobile",
    "unknown",
)

VALID_SPLITS: tuple[str, ...] = ("train", "validation", "test", "unassigned")
VALID_CLASSES: tuple[str, ...] = ("Authentic", "Tampered", "Unknown")

_BOOL_FIELDS = ("mask_available", "pixel_training_eligible")
_INT_FIELDS = ("width", "height", "file_size")

_SOURCE_ALIASES = {
    "dslr": "dslr_mirrorless",
    "mirrorless": "dslr_mirrorless",
    "dslr/mirrorless": "dslr_mirrorless",
    "dslr_mirrorless": "dslr_mirrorless",
    "camera": "dslr_mirrorless",
    "cameras": "dslr_mirrorless",
    "android": "android",
    "android_smartphone": "android",
    "iphone": "iphone",
    "ios": "iphone",
    "apple_iphone": "iphone",
    "other_mobile": "other_mobile",
    "other mobile": "other_mobile",
    "mobile_other": "other_mobile",
    "mobile": "other_mobile",
    "smartphone_other": "other_mobile",
    "smartphone other": "other_mobile",
    "smartphone-other": "other_mobile",
    "unknown": "unknown",
    "other": "unknown",
    "unlabelled": "unknown",
    "unlabeled": "unknown",
    "": "unknown",
}

_TRUE = {"1", "true", "yes", "y", "available", "present"}
_FALSE = {"0", "false", "no", "n", "missing", "absent", ""}


def normalise_source_group(value: Any) -> str:
    """Map common source labels to one canonical manifest group.

    Unknown aliases are rejected rather than silently treated as legacy data.
    """

    text = str(value if value is not None else "").strip().lower()
    collapsed = " ".join(text.replace("-", "_").split())
    if collapsed in _SOURCE_ALIASES:
        return _SOURCE_ALIASES[collapsed]
    raise ValueError(
        f"Unsupported source_group {value!r}; expected one of {BASE_SOURCE_GROUPS}"
    )


def _parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    text = str(value if value is not None else "").strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"{field_name} must be boolean, got {value!r}")


def _parse_int(value: Any, field_name: str) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc


def _clean_text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def normalise_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with stable fields and typed bool/int values."""

    output: dict[str, Any] = {}
    for name in ALL_FIELDS:
        output[name] = record.get(name, "")

    for name in _BOOL_FIELDS:
        output[name] = _parse_bool(output[name], name)
    for name in _INT_FIELDS:
        parsed = _parse_int(output[name], name)
        output[name] = "" if parsed is None else parsed

    for name in MANIFEST_FIELDS + OPTIONAL_FIELDS:
        if name in _BOOL_FIELDS or name in _INT_FIELDS:
            continue
        output[name] = _clean_text(output[name])

    return output


def resolve_manifest_path(value: str | Path | None, data_root: str | Path | None) -> Path | None:
    """Resolve a manifest path against the data root without changing it."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    if data_root is None:
        return path
    return Path(data_root) / path


def to_posix_relative(path: str | Path, root: str | Path) -> str:
    """Store paths relative to the data root using forward slashes."""

    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path.as_posix()
    return relative.as_posix()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_image_rgb(path: str | Path) -> np.ndarray:
    """Decode an image to EXIF-oriented RGB uint8.

    The duplicate audit uses the visible orientation.  This function is not used
    by the deployed inference pipeline, so it cannot change Module 1 behaviour.
    """

    with Image.open(path) as image:
        oriented = ImageOps.exif_transpose(image)
        rgb = oriented.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8).copy()


def decoded_image_sha256(image_rgb: np.ndarray) -> str:
    array = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(struct.pack(">II", int(array.shape[0]), int(array.shape[1])))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _bits_to_hex(bits: np.ndarray) -> str:
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bool(bit))
    width = max(1, (bits.size + 3) // 4)
    return f"{value:0{width}x}"


def phash_bits(image_rgb: np.ndarray) -> np.ndarray:
    """64-bit DCT perceptual hash (no external imagehash dependency)."""

    gray = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(resized.astype(np.float32))
    low = coefficients[:8, :8].copy()
    low[0, 0] = 0.0
    median = float(np.median(low.ravel()[1:]))
    return (low > median).reshape(-1)


def dhash_bits(image_rgb: np.ndarray) -> np.ndarray:
    """64-bit difference hash."""

    gray = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (resized[:, 1:] > resized[:, :-1]).reshape(-1)


def phash(image_rgb: np.ndarray) -> str:
    return _bits_to_hex(phash_bits(image_rgb))


def dhash(image_rgb: np.ndarray) -> str:
    return _bits_to_hex(dhash_bits(image_rgb))


def hamming_distance(left: str, right: str) -> int:
    """Hamming distance between two hexadecimal perceptual hashes."""

    if not left or not right:
        raise ValueError("Both hashes must be non-empty")
    return popcount(int(left, 16) ^ int(right, 16))


def popcount(value: int) -> int:
    try:
        return int(value).bit_count()
    except AttributeError:  # pragma: no cover - Python < 3.10 compatibility
        return bin(int(value)).count("1")


def compute_image_hashes(
    path: str | Path,
    *,
    include_decoded: bool = True,
) -> dict[str, Any]:
    """Read one image and return metadata plus file/content/perceptual hashes."""

    path = Path(path)
    image = read_image_rgb(path)
    height, width = image.shape[:2]
    payload: dict[str, Any] = {
        "width": int(width),
        "height": int(height),
        "format": path.suffix.lower().lstrip("."),
        "sha256": sha256_file(path),
        "phash": phash(image),
        "dhash": dhash(image),
        "file_size": int(path.stat().st_size),
        "readable": True,
    }
    if include_decoded:
        payload["decoded_sha256"] = decoded_image_sha256(image)
    return payload


@dataclass
class ManifestValidationIssue:
    severity: str
    code: str
    message: str
    sample_id: str = ""
    field: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "sample_id": self.sample_id,
            "field": self.field,
            "message": self.message,
        }


@dataclass
class ManifestValidationResult:
    errors: list[ManifestValidationIssue] = field(default_factory=list)
    warnings: list[ManifestValidationIssue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            details = "\n".join(
                f"- {issue.code}: {issue.sample_id or '<manifest>'} "
                f"{issue.field + ': ' if issue.field else ''}{issue.message}"
                for issue in self.errors
            )
            raise ManifestValidationError(
                f"Manifest validation found {len(self.errors)} error(s):\n{details}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "summary": dict(self.summary),
            "errors": [issue.as_dict() for issue in self.errors],
            "warnings": [issue.as_dict() for issue in self.warnings],
        }


class ManifestValidationError(ValueError):
    """Raised when a manifest fails strict validation."""


def _check_readable_image(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            image.verify()
        return True, ""
    except Exception as exc:  # noqa: BLE001 - report the real decoder error
        return False, str(exc)


def _check_readable_mask(path: Path) -> tuple[bool, str]:
    try:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False, "OpenCV could not decode the mask"
        if not np.any(mask > 0):
            return True, ""
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def validate_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    data_root: str | Path | None = None,
    strict: bool = False,
    check_paths: bool = True,
) -> ManifestValidationResult:
    """Validate schema, uniqueness, paths, masks and split isolation.

    Missing or invalid image/mask paths are reported, never silently dropped.
    A tampered *training* row without a valid mask is an error, because it cannot
    be used for pixel-level supervision.  The CASIA split contains one known
    mask-less tampered row; it remains visible in the manifest and is marked as
    not pixel-training eligible.
    """

    result = ManifestValidationResult()
    seen_sample_ids: dict[str, int] = {}
    group_splits: dict[str, dict[str, list[str]]] = {}
    group_datasets: dict[str, set[str]] = {}
    image_paths: dict[str, list[str]] = {}
    mask_paths: dict[str, list[str]] = {}
    split_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    mask_available_count = 0
    pixel_training_eligible_count = 0

    root = Path(data_root) if data_root is not None else None

    for index, raw in enumerate(records):
        row = normalise_record(raw)
        sample_id = row["sample_id"]
        location = f"row {index + 1}"
        if not sample_id:
            result.errors.append(
                ManifestValidationIssue(
                    "error",
                    "missing_sample_id",
                    f"{location}: sample_id is required",
                    "",
                    "sample_id",
                )
            )
        elif sample_id in seen_sample_ids:
            result.errors.append(
                ManifestValidationIssue(
                    "error",
                    "duplicate_sample_id",
                    f"{location}: sample_id already used at row {seen_sample_ids[sample_id] + 1}",
                    sample_id,
                    "sample_id",
                )
            )
        else:
            seen_sample_ids[sample_id] = index

        for field_name in ("dataset", "source_group", "split", "class_name", "image_path", "group_id"):
            if not row[field_name]:
                result.errors.append(
                    ManifestValidationIssue(
                        "error",
                        f"missing_{field_name}",
                        f"{location}: {field_name} is required",
                        sample_id,
                        field_name,
                    )
                )

        if row["split"] and row["split"] not in VALID_SPLITS:
            result.errors.append(
                ManifestValidationIssue(
                    "error",
                    "invalid_split",
                    f"split must be one of {VALID_SPLITS}",
                    sample_id,
                    "split",
                )
            )
        if row["class_name"] and row["class_name"] not in VALID_CLASSES:
            result.errors.append(
                ManifestValidationIssue(
                    "error",
                    "invalid_class_name",
                    f"class_name must be one of {VALID_CLASSES}",
                    sample_id,
                    "class_name",
                )
            )
        if row["source_group"]:
            try:
                normalised_group = normalise_source_group(row["source_group"])
            except ValueError as exc:
                result.errors.append(
                    ManifestValidationIssue(
                        "error", "invalid_source_group", str(exc), sample_id, "source_group"
                    )
                )
            else:
                if normalised_group != row["source_group"]:
                    result.errors.append(
                        ManifestValidationIssue(
                            "error",
                            "noncanonical_source_group",
                            f"use canonical value {normalised_group!r}",
                            sample_id,
                            "source_group",
                        )
                    )

        if row["class_name"] == "Authentic" and row["mask_available"]:
            result.warnings.append(
                ManifestValidationIssue(
                    "warning",
                    "authentic_has_mask",
                    "Authentic rows normally have no manipulation mask",
                    sample_id,
                    "mask_path",
                )
            )
        if row["class_name"] == "Tampered" and not row["mask_available"]:
            if row["split"] == "train":
                result.errors.append(
                    ManifestValidationIssue(
                        "error",
                        "missing_mask_for_pixel_training",
                        "Tampered training rows require a valid mask for pixel supervision",
                        sample_id,
                        "mask_available",
                    )
                )
            else:
                result.warnings.append(
                    ManifestValidationIssue(
                        "warning",
                        "missing_mask_for_evaluation",
                        "Tampered row has no mask; pixel metrics will exclude it explicitly",
                        sample_id,
                        "mask_available",
                    )
                )

        if row["pixel_training_eligible"]:
            pixel_training_eligible_count += 1
            if row["class_name"] != "Tampered" or not row["mask_available"]:
                result.errors.append(
                    ManifestValidationIssue(
                        "error",
                        "invalid_pixel_training_eligibility",
                        "pixel_training_eligible requires class_name=Tampered and a mask",
                        sample_id,
                        "pixel_training_eligible",
                    )
                )
        if row["class_name"] == "Tampered" and row["split"] == "train":
            if not row["pixel_training_eligible"]:
                result.warnings.append(
                    ManifestValidationIssue(
                        "warning",
                        "train_tampered_not_pixel_eligible",
                        "Tampered train row will not be used for pixel training",
                        sample_id,
                        "pixel_training_eligible",
                    )
                )

        if row["group_id"] and row["split"]:
            group_splits.setdefault(row["group_id"], {}).setdefault(row["split"], []).append(
                sample_id
            )
        if row["group_id"] and row["dataset"]:
            group_datasets.setdefault(row["group_id"], set()).add(row["dataset"])

        if row["image_path"]:
            image_paths.setdefault(row["image_path"], []).append(sample_id)
        if row["mask_available"]:
            mask_available_count += 1
            if not row["mask_path"]:
                result.errors.append(
                    ManifestValidationIssue(
                        "error",
                        "mask_available_without_path",
                        "mask_available is true but mask_path is empty",
                        sample_id,
                        "mask_path",
                    )
                )
            else:
                mask_paths.setdefault(row["mask_path"], []).append(sample_id)
        elif row["mask_path"]:
            result.warnings.append(
                ManifestValidationIssue(
                    "warning",
                    "mask_path_without_mask_available",
                    "mask_path is set but mask_available is false",
                    sample_id,
                    "mask_available",
                )
            )

        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        class_counts[row["class_name"]] = class_counts.get(row["class_name"], 0) + 1
        source_counts[row["source_group"]] = source_counts.get(row["source_group"], 0) + 1

        if not check_paths:
            continue

        if row["image_path"]:
            image_path = resolve_manifest_path(row["image_path"], root)
            if image_path is None or not image_path.exists():
                result.errors.append(
                    ManifestValidationIssue(
                        "error",
                        "missing_image",
                        f"image not found: {row['image_path']}",
                        sample_id,
                        "image_path",
                    )
                )
            else:
                readable, detail = _check_readable_image(image_path)
                if not readable:
                    result.errors.append(
                        ManifestValidationIssue(
                            "error",
                            "unreadable_image",
                            f"image could not be decoded: {detail}",
                            sample_id,
                            "image_path",
                        )
                    )

        if row["mask_available"] and row["mask_path"]:
            mask_path = resolve_manifest_path(row["mask_path"], root)
            if mask_path is None or not mask_path.exists():
                result.errors.append(
                    ManifestValidationIssue(
                        "error",
                        "missing_mask",
                        f"mask not found: {row['mask_path']}",
                        sample_id,
                        "mask_path",
                    )
                )
            else:
                readable, detail = _check_readable_mask(mask_path)
                if not readable:
                    result.errors.append(
                        ManifestValidationIssue(
                            "error",
                            "unreadable_mask",
                            f"mask could not be decoded: {detail}",
                            sample_id,
                            "mask_path",
                        )
                    )

    for group_id, splits in group_splits.items():
        if len(splits) > 1:
            result.errors.append(
                ManifestValidationIssue(
                    "error",
                    "group_split_leakage",
                    f"group {group_id!r} appears in multiple splits: {sorted(splits)}",
                    "",
                    "group_id",
                )
            )
    for group_id, datasets in group_datasets.items():
        if len(datasets) > 1:
            result.warnings.append(
                ManifestValidationIssue(
                    "warning",
                    "group_cross_dataset",
                    f"group {group_id!r} spans datasets: {sorted(datasets)}",
                    "",
                    "group_id",
                )
            )
    for path, sample_ids in image_paths.items():
        if len(sample_ids) > 1:
            result.warnings.append(
                ManifestValidationIssue(
                    "warning",
                    "duplicate_image_path",
                    f"image path referenced by {len(sample_ids)} samples: {sample_ids}",
                    "",
                    "image_path",
                )
            )
    for path, sample_ids in mask_paths.items():
        if len(sample_ids) > 1:
            result.warnings.append(
                ManifestValidationIssue(
                    "warning",
                    "shared_mask_path",
                    f"mask path referenced by {len(sample_ids)} samples: {sample_ids}",
                    "",
                    "mask_path",
                )
            )

    result.summary = {
        "rows": len(records),
        "unique_sample_ids": len(seen_sample_ids),
        "unique_groups": len(group_splits),
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "source_group_counts": dict(sorted(source_counts.items())),
        "mask_available": int(mask_available_count),
        "pixel_training_eligible": int(pixel_training_eligible_count),
        "images_checked": bool(check_paths),
    }
    if strict:
        result.raise_for_errors()
    return result


def manifest_json_schema() -> dict[str, Any]:
    """Return a small JSON Schema for the manifest fields."""

    properties: dict[str, Any] = {}
    for name in ALL_FIELDS:
        if name in _BOOL_FIELDS:
            properties[name] = {"type": "boolean"}
        elif name in _INT_FIELDS:
            properties[name] = {"type": ["integer", "string"]}
        elif name == "source_group":
            properties[name] = {"type": "string", "enum": list(BASE_SOURCE_GROUPS)}
        elif name == "split":
            properties[name] = {"type": "string", "enum": list(VALID_SPLITS)}
        elif name == "class_name":
            properties[name] = {"type": "string", "enum": list(VALID_CLASSES)}
        else:
            properties[name] = {"type": "string"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Digital Image Forensics unified dataset manifest",
        "type": "object",
        "required": list(MANIFEST_FIELDS),
        "properties": properties,
        "additionalProperties": False,
    }


def dump_schema(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest_json_schema(), indent=2) + "\n", encoding="utf-8")
    return path
