"""Read and write the additive unified dataset manifest.

CSV is the default interchange format.  JSON Lines is useful for records whose
notes contain commas or for programmatic hand-off.  Both readers/writers use the
same ordered schema.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import ALL_FIELDS, MANIFEST_FIELDS, normalise_record


def write_manifest(
    records: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    fmt: str | None = None,
) -> Path:
    """Write records without touching any source image."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    rows = [normalise_record(record) for record in records]

    if fmt == "csv":
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ALL_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    elif fmt in {"jsonl", "ndjson"}:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
    elif fmt == "json":
        path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:  # pragma: no cover - argument validation
        raise ValueError(f"Unsupported manifest format: {fmt!r}")
    return path


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a CSV/JSON/JSONL manifest into typed dictionaries.

    Missing required columns raise immediately.  Row-level path validation is a
    separate step so that a manifest can be inspected without decoding every
    image.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    rows: list[dict[str, Any]] = []

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Manifest has no header: {path}")
            missing = [name for name in MANIFEST_FIELDS if name not in reader.fieldnames]
            if missing:
                raise ValueError(
                    f"Manifest {path.name} is missing required columns: {missing}"
                )
            rows = [dict(row) for row in reader]
    elif suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {path.name}: {exc}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Line {line_number} of {path.name} is not a JSON object"
                    )
                rows.append(payload)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"{path.name} must contain a JSON array of objects")
        rows = payload
    else:  # pragma: no cover - argument validation
        raise ValueError(f"Unsupported manifest format: {suffix!r}")

    for index, row in enumerate(rows, start=1):
        missing = [name for name in MANIFEST_FIELDS if name not in row]
        if missing:
            raise ValueError(
                f"Row {index} in {path.name} is missing required fields: {missing}"
            )
    return [normalise_record(row) for row in rows]
