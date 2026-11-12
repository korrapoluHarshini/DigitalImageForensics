"""Build a CASIA 2.0 manifest from the preserved split without changing it.

Usage:
    .\\mvss_venv\\Scripts\\python.exe evaluation/build_casia_manifest.py
    .\\mvss_venv\\Scripts\\python.exe evaluation/build_casia_manifest.py --no-hashes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_manifest.casia import CASIABuildConfig, build_casia_manifest
from dataset_manifest.io import write_manifest
from utils.config import settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=settings.data_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "manifests" / "casia2_manifest.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Defaults to <output stem>.report.json",
    )
    parser.add_argument(
        "--no-hashes",
        action="store_true",
        help="Skip SHA-256/pHash/dHash decoding (fast schema check only).",
    )
    parser.add_argument(
        "--no-decoded-hash",
        action="store_true",
        help="Skip the decoded-pixel hash while still computing SHA-256/pHash/dHash.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if validation reports errors.",
    )
    args = parser.parse_args()

    config = CASIABuildConfig.from_data_root(args.data_root)
    print(f"Data root: {config.data_root}")
    print(f"Split CSV: {config.split_csv}")
    print(f"Pairs CSV: {config.pairs_csv}")
    print(f"Computing hashes: {not args.no_hashes}")


    def progress(index: int, total: int, sample_id: str) -> None:
        if index == 1 or index % 250 == 0 or index == total:
            print(f"  {index:,}/{total:,} {sample_id}")

    records, report = build_casia_manifest(
        config,
        compute_hashes=not args.no_hashes,
        include_decoded_hash=not args.no_decoded_hash,
        progress=progress,
    )
    write_manifest(records, args.output, fmt="csv")
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")

    print("\n=== CASIA 2.0 MANIFEST ===")
    print(f"Rows: {report.counts.get('rows', 0):,}")
    print(f"Unique sample IDs: {report.counts.get('unique_sample_ids', 0):,}")
    print(f"Unique groups: {report.counts.get('unique_groups', 0):,}")
    print(f"Missing images: {len(report.missing_images)}")
    print(f"Missing masks: {len(report.missing_masks)}")
    print(f"Unreadable masks: {len(report.unreadable_masks)}")
    print(f"Duplicate filenames: {len(report.duplicate_filenames)}")
    print(f"Ambiguous mask mappings: {len(report.ambiguous_mask_mappings)}")
    print(f"Validation errors: {report.validation_error_count}")
    print(f"Validation warnings: {report.validation_warning_count}")
    print(f"Manifest: {args.output}")
    print(f"Report:   {report_path}")

    if args.strict and report.validation_error_count:
        print("Strict validation failed.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
