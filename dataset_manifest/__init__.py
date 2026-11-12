"""Additive dataset-manifest tooling for Phase 0.

This package deliberately does not import the model, the application, or the
existing CASIA loader.  It only reads dataset metadata and image files, builds
manifests, audits duplicates and computes evaluation metrics from supplied
predictions.  No original dataset file is ever modified.
"""

from __future__ import annotations

from .schema import (
    ALL_FIELDS,
    BASE_SOURCE_GROUPS,
    MANIFEST_FIELDS,
    OPTIONAL_FIELDS,
    ManifestValidationResult,
    compute_image_hashes,
    hamming_distance,
    normalise_record,
    normalise_source_group,
    resolve_manifest_path,
    validate_manifest,
)
from .io import load_manifest, write_manifest

__all__ = [
    "ALL_FIELDS",
    "BASE_SOURCE_GROUPS",
    "MANIFEST_FIELDS",
    "OPTIONAL_FIELDS",
    "ManifestValidationResult",
    "compute_image_hashes",
    "hamming_distance",
    "load_manifest",
    "normalise_record",
    "normalise_source_group",
    "resolve_manifest_path",
    "validate_manifest",
    "write_manifest",
]
