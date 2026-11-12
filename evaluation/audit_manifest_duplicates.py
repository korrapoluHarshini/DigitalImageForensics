"""Audit a unified manifest for duplicates and potential leakage.

Usage:
    .\\mvss_venv\\Scripts\\python.exe evaluation/audit_manifest_duplicates.py \\
        --manifest results/manifests/casia2_manifest.csv \\
        --no-geometric
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_manifest.duplicates import audit_manifest
from dataset_manifest.io import load_manifest
from utils.config import settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=settings.data_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "manifests" / "casia2_duplicate_audit.json",
    )
    parser.add_argument("--phash-distance", type=int, default=6)
    parser.add_argument("--dhash-distance", type=int, default=8)
    parser.add_argument(
        "--geometric",
        action="store_true",
        help="Run ORB/RANSAC geometric matching (bounded by --max-geometric-pairs).",
    )
    parser.add_argument("--max-geometric-pairs", type=int, default=500)
    parser.add_argument(
        "--no-compute-missing-hashes",
        action="store_true",
        help="Do not compute hashes for manifest rows that lack them.",
    )
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    print(f"Loaded {len(records):,} manifest rows from {args.manifest}")
    result = audit_manifest(
        records,
        data_root=args.data_root,
        compute_missing_hashes=not args.no_compute_missing_hashes,
        phash_distance=args.phash_distance,
        dhash_distance=args.dhash_distance,
        run_geometric=args.geometric,
        geometric_max_pairs=args.max_geometric_pairs,
    )
    payload = {
        "artifact": "manifest_duplicate_audit",
        "manifest": str(args.manifest),
        "data_root": str(args.data_root),
        **result.as_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    summary = result.summary
    print("\n=== DUPLICATE AUDIT ===")
    print(f"Exact SHA-256 groups: {summary.get('exact_duplicate_groups', 0)}")
    print(f"Decoded-image groups: {summary.get('decoded_duplicate_groups', 0)}")
    print(f"pHash groups: {summary.get('phash_groups', 0)}")
    print(f"dHash groups: {summary.get('dhash_groups', 0)}")
    print(f"Geometric groups: {summary.get('geometric_groups', 0)}")
    print(f"Cross-dataset groups: {summary.get('cross_dataset_groups', 0)}")
    print(f"Split-leakage duplicate groups: {summary.get('split_leakage_groups', 0)}")
    print(f"Relationships: {summary.get('relationships', 0)}")
    print(f"Audit issues: {summary.get('issues', 0)}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
