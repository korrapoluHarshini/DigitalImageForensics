"""Read-only duplicate, near-duplicate and leakage audit for manifests.

The audit detects:

* exact file duplicates using SHA-256;
* decoded-pixel duplicates using a canonical RGB-pixel hash;
* perceptual near-duplicates using pHash (DCT) and dHash;
* geometric near-duplicates using ORB + RANSAC homography when requested;
* explicit original/manipulated links through ``group_id``/``pair_id``;
* duplicate content across datasets.

Only metadata and decoded images are read.  No original file is renamed,
rewritten or deleted.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .schema import (
    compute_image_hashes,
    normalise_record,
    popcount,
    read_image_rgb,
    resolve_manifest_path,
)


@dataclass
class DuplicateGroup:
    kind: str
    key: str
    sample_ids: list[str]
    datasets: list[str]
    paths: list[str]
    splits: list[str]
    classes: list[str]
    group_ids: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "key": self.key,
            "sample_count": len(self.sample_ids),
            "sample_ids": list(self.sample_ids),
            "datasets": list(self.datasets),
            "paths": list(self.paths),
            "splits": list(self.splits),
            "classes": list(self.classes),
            "group_ids": list(self.group_ids),
        }


@dataclass
class Relationship:
    sample_a: str
    sample_b: str
    relationship: str
    evidence: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_a": self.sample_a,
            "sample_b": self.sample_b,
            "relationship": self.relationship,
            "evidence": self.evidence,
            "details": dict(self.details),
        }


@dataclass
class DuplicateAuditResult:
    exact_groups: list[DuplicateGroup] = field(default_factory=list)
    decoded_groups: list[DuplicateGroup] = field(default_factory=list)
    phash_groups: list[DuplicateGroup] = field(default_factory=list)
    dhash_groups: list[DuplicateGroup] = field(default_factory=list)
    geometric_groups: list[DuplicateGroup] = field(default_factory=list)
    cross_dataset_groups: list[dict[str, Any]] = field(default_factory=list)
    split_leakage_groups: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def all_groups(self) -> list[DuplicateGroup]:
        return (
            self.exact_groups
            + self.decoded_groups
            + self.phash_groups
            + self.dhash_groups
            + self.geometric_groups
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "exact_groups": [group.as_dict() for group in self.exact_groups],
            "decoded_groups": [group.as_dict() for group in self.decoded_groups],
            "phash_groups": [group.as_dict() for group in self.phash_groups],
            "dhash_groups": [group.as_dict() for group in self.dhash_groups],
            "geometric_groups": [group.as_dict() for group in self.geometric_groups],
            "cross_dataset_groups": list(self.cross_dataset_groups),
            "split_leakage_groups": list(self.split_leakage_groups),
            "relationships": [relationship.as_dict() for relationship in self.relationships],
            "issues": list(self.issues),
            "summary": dict(self.summary),
        }


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _hash_int(value: Any) -> int | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    try:
        return int(text, 16)
    except ValueError:
        return None


def _variants_16(value: int, radius: int) -> tuple[int, ...]:
    values = {int(value)}
    for positions in range(1, radius + 1):
        for combination in itertools.combinations(range(16), positions):
            variant = int(value)
            for position in combination:
                variant ^= 1 << position
            values.add(variant)
    return tuple(values)


def _candidate_pairs(values: Sequence[int], max_distance: int) -> set[tuple[int, int]]:
    """Return index pairs whose hashes may be within ``max_distance``.

    For 64-bit hashes split into four 16-bit chunks, a pair with distance <= 8
    must have at least one chunk with distance <= 2.  That pigeonhole property
    makes the audit practical for CASIA-scale manifests.  A brute-force path is
    used for small sets and for distances above 8.
    """

    count = len(values)
    if count < 2:
        return set()
    if max_distance <= 0:
        return {(i, j) for i in range(count) for j in range(i + 1, count) if values[i] == values[j]}
    if count <= 2500 or max_distance > 8:
        pairs: set[tuple[int, int]] = set()
        for i in range(count):
            for j in range(i + 1, count):
                if popcount(values[i] ^ values[j]) <= max_distance:
                    pairs.add((i, j))
        return pairs

    radius = max(1, max_distance // 4)
    pairs: set[tuple[int, int]] = set()
    buckets: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(4)]
    for index, value in enumerate(values):
        for chunk_index in range(4):
            chunk = (value >> (16 * chunk_index)) & 0xFFFF
            for variant in _variants_16(chunk, radius):
                for previous in buckets[chunk_index].get(variant, ()):
                    if popcount(value ^ values[previous]) <= max_distance:
                        pairs.add((previous, index))
        for chunk_index in range(4):
            chunk = (value >> (16 * chunk_index)) & 0xFFFF
            buckets[chunk_index][chunk].append(index)
    return pairs


def _groups_from_pairs(
    records: Sequence[Mapping[str, Any]],
    pairs: Iterable[tuple[int, int]],
    *,
    kind: str,
    key: str,
) -> list[DuplicateGroup]:
    union = _UnionFind(len(records))
    for left, right in pairs:
        union.union(left, right)
    buckets: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        buckets[union.find(index)].append(index)
    groups: list[DuplicateGroup] = []
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        selected = [records[index] for index in indices]
        groups.append(
            DuplicateGroup(
                kind=kind,
                key=key,
                sample_ids=[str(row["sample_id"]) for row in selected],
                datasets=sorted({str(row["dataset"]) for row in selected}),
                paths=sorted({str(row["image_path"]) for row in selected}),
                splits=sorted({str(row["split"]) for row in selected}),
                classes=sorted({str(row["class_name"]) for row in selected}),
                group_ids=sorted({str(row["group_id"]) for row in selected}),
            )
        )
    groups.sort(key=lambda group: (-len(group.sample_ids), group.sample_ids))
    return groups


def find_hash_duplicates(
    records: Sequence[Mapping[str, Any]],
    field_name: str,
    *,
    kind: str,
    max_distance: int = 0,
) -> list[DuplicateGroup]:
    """Group records by an exact hash or a Hamming-distance threshold."""

    indexed: list[tuple[int, int]] = []
    for index, record in enumerate(records):
        value = _hash_int(record.get(field_name, ""))
        if value is not None:
            indexed.append((index, value))
    if len(indexed) < 2:
        return []
    values = [value for _, value in indexed]
    if max_distance == 0:
        buckets: dict[int, list[int]] = defaultdict(list)
        for position, value in enumerate(values):
            buckets[value].append(position)
        pairs: set[tuple[int, int]] = set()
        for positions in buckets.values():
            for left, right in itertools.combinations(positions, 2):
                pairs.add((left, right))
        groups = _groups_from_pairs(
            [records[index] for index, _ in indexed],
            pairs,
            kind=kind,
            key=f"{field_name}_exact",
        )
    else:
        groups = _groups_from_pairs(
            [records[index] for index, _ in indexed],
            _candidate_pairs(values, max_distance),
            kind=kind,
            key=f"{field_name}_distance<={max_distance}",
        )
    return groups


def _load_gray(path: str | Path, max_side: int = 1024) -> np.ndarray:
    rgb = read_image_rgb(path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        scale = max_side / float(longest)
        gray = cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return gray


def geometric_match(
    path_a: str | Path,
    path_b: str | Path,
    *,
    max_side: int = 1024,
    min_inliers: int = 16,
    min_inlier_ratio: float = 0.30,
) -> dict[str, Any] | None:
    """Return ORB/RANSAC evidence for a geometric near-duplicate, or ``None``.

    ORB is not mirror-invariant, so a horizontally flipped copy of B is also
    tried.  This is intentionally a conservative check: false matches become
    more likely on flat or synthetic images.
    """

    gray_a = _load_gray(path_a, max_side=max_side)
    gray_b = _load_gray(path_b, max_side=max_side)
    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=8)
    keypoints_a, descriptors_a = orb.detectAndCompute(gray_a, None)
    if descriptors_a is None or len(keypoints_a) < min_inliers:
        return None

    best: dict[str, Any] | None = None
    for flipped, candidate_b in (
        (False, gray_b),
        (True, cv2.flip(gray_b, 1)),
    ):
        keypoints_b, descriptors_b = orb.detectAndCompute(candidate_b, None)
        if descriptors_b is None or len(keypoints_b) < min_inliers:
            continue
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        try:
            matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
        except cv2.error:  # pragma: no cover - defensive
            continue
        good = [
            first
            for first, second in matches
            if first.distance < 0.75 * max(second.distance, 1e-6)
        ]
        if len(good) < min_inliers:
            continue
        source = np.float32([keypoints_a[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
        target = np.float32([keypoints_b[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(
            source, target, cv2.RANSAC, 3.0, maxIters=3000, confidence=0.995
        )
        if homography is None or mask is None:
            continue
        inliers = int(mask.sum())
        ratio = inliers / float(max(1, len(good)))
        if inliers < min_inliers or ratio < min_inlier_ratio:
            continue
        payload = {
            "good_matches": int(len(good)),
            "inliers": int(inliers),
            "inlier_ratio": round(float(ratio), 4),
            "horizontal_flip": bool(flipped),
            "homography": [[round(float(value), 6) for value in row] for row in homography],
        }
        if best is None or inliers > best["inliers"]:
            best = payload
    return best


def _geometric_candidate_pairs(
    records: Sequence[Mapping[str, Any]],
    groups: Sequence[DuplicateGroup],
    *,
    max_pairs: int,
) -> list[tuple[int, int]]:
    by_sample = {str(record["sample_id"]): index for index, record in enumerate(records)}
    candidates: set[tuple[int, int]] = set()

    for group in groups:
        indices = [by_sample[sample_id] for sample_id in group.sample_ids if sample_id in by_sample]
        for left, right in itertools.combinations(indices, 2):
            candidates.add((min(left, right), max(left, right)))

    by_group: dict[str, list[int]] = defaultdict(list)
    by_pair: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.get("group_id"):
            by_group[str(record["group_id"])].append(index)
        if record.get("pair_id"):
            by_pair[str(record["pair_id"])].append(index)
    for bucket in list(by_group.values()) + list(by_pair.values()):
        for left, right in itertools.combinations(bucket, 2):
            candidates.add((min(left, right), max(left, right)))

    total_pairs = len(records) * (len(records) - 1) // 2
    if max_pairs > 0 and total_pairs <= max_pairs:
        return list(itertools.combinations(range(len(records)), 2))

    ordered = sorted(candidates)
    if max_pairs > 0:
        ordered = ordered[:max_pairs]
    return ordered


def _dedupe_relationships(relationships: Sequence[Relationship]) -> list[Relationship]:
    seen: set[tuple[str, str, str]] = set()
    output: list[Relationship] = []
    for relationship in relationships:
        left, right = sorted((relationship.sample_a, relationship.sample_b))
        key = (left, right, relationship.relationship)
        if key in seen:
            continue
        seen.add(key)
        output.append(relationship)
    output.sort(key=lambda item: (item.relationship, item.sample_a, item.sample_b))
    return output


def _relationship_from_group(
    records_by_id: Mapping[str, Mapping[str, Any]],
    group: DuplicateGroup,
    *,
    relationship: str,
    evidence: str,
    max_pairs: int = 50,
) -> list[Relationship]:
    output: list[Relationship] = []
    pairs = itertools.islice(itertools.combinations(sorted(group.sample_ids), 2), max_pairs)
    for left, right in pairs:
        left_record = records_by_id.get(left, {})
        right_record = records_by_id.get(right, {})
        details = {
            "left_class": left_record.get("class_name", ""),
            "right_class": right_record.get("class_name", ""),
            "left_dataset": left_record.get("dataset", ""),
            "right_dataset": right_record.get("dataset", ""),
        }
        output.append(
            Relationship(
                sample_a=left,
                sample_b=right,
                relationship=relationship,
                evidence=evidence,
                details=details,
            )
        )
    return output


def audit_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    data_root: str | Path | None = None,
    compute_missing_hashes: bool = True,
    phash_distance: int = 6,
    dhash_distance: int = 8,
    run_geometric: bool = False,
    geometric_max_pairs: int = 500,
) -> DuplicateAuditResult:
    """Audit a manifest without modifying any source file."""

    working = [normalise_record(record) for record in records]
    result = DuplicateAuditResult()
    cache: dict[str, dict[str, Any]] = {}

    if compute_missing_hashes:
        for record in working:
            needed = not all(
                record.get(name)
                for name in ("sha256", "phash", "dhash", "decoded_sha256")
            )
            if not needed:
                continue
            path = resolve_manifest_path(record.get("image_path"), data_root)
            if path is None or not path.exists():
                result.issues.append(
                    {
                        "sample_id": record.get("sample_id", ""),
                        "code": "image_unavailable_for_hash",
                        "message": f"image not found: {record.get('image_path', '')}",
                    }
                )
                continue
            key = str(path)
            if key not in cache:
                try:
                    cache[key] = compute_image_hashes(path, include_decoded=True)
                except Exception as exc:  # noqa: BLE001
                    result.issues.append(
                        {
                            "sample_id": record.get("sample_id", ""),
                            "code": "image_hash_failed",
                            "message": f"{path}: {exc}",
                        }
                    )
                    continue
            for field_name, value in cache[key].items():
                if field_name in {"width", "height", "format", "file_size", "readable"}:
                    continue
                if not record.get(field_name):
                    record[field_name] = value

    result.exact_groups = find_hash_duplicates(
        working, "sha256", kind="exact_sha256", max_distance=0
    )
    result.decoded_groups = find_hash_duplicates(
        working, "decoded_sha256", kind="decoded_pixels", max_distance=0
    )
    result.phash_groups = find_hash_duplicates(
        working, "phash", kind="phash", max_distance=phash_distance
    )
    result.dhash_groups = find_hash_duplicates(
        working, "dhash", kind="dhash", max_distance=dhash_distance
    )

    records_by_id = {str(record["sample_id"]): record for record in working}
    relationships: list[Relationship] = []

    # Explicit group and pair links are authoritative even if the pixels differ.
    by_group: dict[str, list[str]] = defaultdict(list)
    by_pair: dict[str, list[str]] = defaultdict(list)
    for record in working:
        if record.get("group_id"):
            by_group[str(record["group_id"])].append(str(record["sample_id"]))
        if record.get("pair_id"):
            by_pair[str(record["pair_id"])].append(str(record["sample_id"]))
    for group_id, sample_ids in by_group.items():
        unique = sorted(set(sample_ids))
        for left, right in itertools.islice(itertools.combinations(unique, 2), 50):
            relationships.append(
                Relationship(
                    sample_a=left,
                    sample_b=right,
                    relationship="explicit_shared_group",
                    evidence=f"group_id={group_id}",
                )
            )
    for pair_id, sample_ids in by_pair.items():
        unique = sorted(set(sample_ids))
        for left, right in itertools.islice(itertools.combinations(unique, 2), 50):
            relationships.append(
                Relationship(
                    sample_a=left,
                    sample_b=right,
                    relationship="explicit_pair",
                    evidence=f"pair_id={pair_id}",
                )
            )

    for group in result.all_groups():
        if "Authentic" in group.classes and "Tampered" in group.classes:
            relationships.extend(
                _relationship_from_group(
                    records_by_id,
                    group,
                    relationship="visual_cross_class",
                    evidence=f"{group.kind} duplicate group spans authentic/tampered rows",
                )
            )

    if run_geometric:
        candidate_pairs = _geometric_candidate_pairs(
            working,
            result.exact_groups
            + result.decoded_groups
            + result.phash_groups
            + result.dhash_groups,
            max_pairs=geometric_max_pairs,
        )
        matches: list[tuple[int, int, dict[str, Any]]] = []
        for left, right in candidate_pairs:
            left_path = resolve_manifest_path(working[left].get("image_path"), data_root)
            right_path = resolve_manifest_path(working[right].get("image_path"), data_root)
            if left_path is None or right_path is None or not left_path.exists() or not right_path.exists():
                continue
            try:
                match = geometric_match(left_path, right_path)
            except Exception as exc:  # noqa: BLE001
                result.issues.append(
                    {
                        "sample_id": f"{working[left]['sample_id']}|{working[right]['sample_id']}",
                        "code": "geometric_match_failed",
                        "message": str(exc),
                    }
                )
                continue
            if match is not None:
                matches.append((left, right, match))
        result.geometric_groups = _groups_from_pairs(
            working,
            [(left, right) for left, right, _ in matches],
            kind="geometric",
            key=f"orb_ransac_inliers>=16",
        )
        for left, right, match in matches:
            left_record = working[left]
            right_record = working[right]
            if {left_record.get("class_name"), right_record.get("class_name")} == {
                "Authentic",
                "Tampered",
            }:
                relationships.append(
                    Relationship(
                        sample_a=str(left_record["sample_id"]),
                        sample_b=str(right_record["sample_id"]),
                        relationship="geometric_original_manipulated",
                        evidence="ORB/RANSAC geometric match across authentic/tampered rows",
                        details=match,
                    )
                )

    result.relationships = _dedupe_relationships(relationships)
    if len(result.relationships) > 10000:
        result.issues.append(
            {
                "sample_id": "",
                "code": "relationships_truncated",
                "message": (
                    f"relationship list truncated from {len(result.relationships)} to 10000; "
                    "duplicate groups retain all sample IDs"
                ),
            }
        )
        result.relationships = result.relationships[:10000]

    for group in result.all_groups():
        if len(group.datasets) > 1:
            result.cross_dataset_groups.append(group.as_dict())
        if len(group.splits) > 1:
            result.split_leakage_groups.append(group.as_dict())
    result.cross_dataset_groups.sort(
        key=lambda payload: (payload["kind"], -payload["sample_count"], payload["sample_ids"])
    )
    result.split_leakage_groups.sort(
        key=lambda payload: (payload["kind"], -payload["sample_count"], payload["sample_ids"])
    )

    def _count_sample_ids(groups: Sequence[DuplicateGroup]) -> int:
        return len({sample_id for group in groups for sample_id in group.sample_ids})

    result.summary = {
        "records": len(working),
        "exact_duplicate_groups": len(result.exact_groups),
        "exact_duplicate_samples": _count_sample_ids(result.exact_groups),
        "decoded_duplicate_groups": len(result.decoded_groups),
        "decoded_duplicate_samples": _count_sample_ids(result.decoded_groups),
        "phash_groups": len(result.phash_groups),
        "phash_samples": _count_sample_ids(result.phash_groups),
        "dhash_groups": len(result.dhash_groups),
        "dhash_samples": _count_sample_ids(result.dhash_groups),
        "geometric_groups": len(result.geometric_groups),
        "geometric_samples": _count_sample_ids(result.geometric_groups),
        "relationships": len(result.relationships),
        "cross_dataset_groups": len(result.cross_dataset_groups),
        "split_leakage_groups": len(result.split_leakage_groups),
        "issues": len(result.issues),
        "parameters": {
            "phash_distance": int(phash_distance),
            "dhash_distance": int(dhash_distance),
            "run_geometric": bool(run_geometric),
            "geometric_max_pairs": int(geometric_max_pairs),
        },
    }
    return result
