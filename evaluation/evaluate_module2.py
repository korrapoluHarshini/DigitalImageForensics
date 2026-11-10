"""Evaluate Module 2 (AI-generated image detection) on a labelled dataset.

This script deliberately has no default dataset: AI-detection datasets are
separate from the CASIA forgery dataset used by Module 1.  Provide folders of
real photographs and AI-generated images.  The script exits with instructions
when the folders or an integrated detector are missing — it never fabricates
metrics for a placeholder model.

Expected layout (subfolders group results by generator)::

    real/          *.jpg, *.png, ...
    ai/
        generator_a/*.png
        generator_b/*.png

Usage:
    python evaluation/evaluate_module2.py --real-dir path/to/real --ai-dir path/to/ai
    python evaluation/evaluate_module2.py --real-dir ... --ai-dir ... --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.ai_generated_detection import detect_ai_generated_image, get_ai_status, load_backend
from utils.config import settings
from utils.image_utils import load_image_rgb
from utils.metrics import classification_metrics


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _collect(directory: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _group_name(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return "unknown"
    return relative.parts[0] if len(relative.parts) > 1 else "root"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--ai-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 = all images")
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.evaluation_dir / "module2_metrics.json",
    )
    args = parser.parse_args()

    real_images = _collect(args.real_dir)
    ai_images = _collect(args.ai_dir)
    if not real_images or not ai_images:
        print(
            "Module 2 evaluation requires labelled folders.\n"
            f"  real images found: {len(real_images)} ({args.real_dir})\n"
            f"  AI images found:   {len(ai_images)} ({args.ai_dir})\n\n"
            "Provide a dataset with real photographs and images from one or more "
            "generators. No metrics are reported without such data.",
            file=sys.stderr,
        )
        return 2

    status = get_ai_status(args.manifest)
    if not status["available"]:
        print(
            "No AI-generated-image detector is integrated:\n"
            f"  {status['reason']}\n\n"
            "Integrate a detector first (see models/ai_detector/README.md). "
            "No metrics are reported for a placeholder.",
            file=sys.stderr,
        )
        return 2

    if args.limit:
        real_images = real_images[: args.limit]
        ai_images = ai_images[: args.limit]

    backend = load_backend(args.manifest)
    rows: list[dict] = []

    for label, paths in ((0, real_images), (1, ai_images)):
        for path in paths:
            try:
                image = load_image_rgb(path)
            except Exception as exc:
                rows.append(
                    {
                        "file": str(path.name),
                        "group": _group_name(path, args.real_dir if label == 0 else args.ai_dir),
                        "label": label,
                        "status": "unreadable",
                        "error": str(exc),
                    }
                )
                continue
            result = detect_ai_generated_image(image, backend=backend, manifest_path=args.manifest)
            rows.append(
                {
                    "file": str(path.name),
                    "group": _group_name(path, args.real_dir if label == 0 else args.ai_dir),
                    "label": label,
                    "status": result.status,
                    "predicted_class": result.predicted_class,
                    "real_confidence": result.real_confidence,
                    "ai_confidence": result.ai_confidence,
                    "uncertain": result.uncertain,
                    "error": result.error_message,
                }
            )

    frame = pd.DataFrame(rows)
    readable = frame[frame["status"] == "ok"].copy()
    if readable.empty:
        print("No image could be scored; check the integrated backend.", file=sys.stderr)
        return 2

    decided = readable[readable["predicted_class"].isin(["real", "ai_generated"])].copy()
    labels = decided["label"].to_numpy(dtype=int)
    predictions = (decided["predicted_class"] == "ai_generated").astype(int).to_numpy()

    metrics = classification_metrics(labels, predictions, num_classes=2, class_names=["real", "ai_generated"])
    uncertain_count = int((readable["predicted_class"] == "uncertain").sum())
    metrics.update(
        {
            "images_scored": int(len(readable)),
            "uncertain_count": uncertain_count,
            "uncertain_rate": float(uncertain_count / max(1, len(readable))),
            "coverage": float(len(decided) / max(1, len(readable))),
            "accuracy_all_with_uncertain_wrong": float(
                (decided["predicted_class"].map({"real": 0, "ai_generated": 1}) == decided["label"]).sum()
                / max(1, len(readable))
            ),
            "dataset": {
                "real_images": int((frame["label"] == 0).sum()),
                "ai_images": int((frame["label"] == 1).sum()),
                "ai_groups": sorted(
                    {
                        str(value)
                        for value in frame[frame["label"] == 1]["group"].dropna().unique()
                    }
                ),
            },
            "provenance": (
                "Evaluated with the integrated backend declared in the manifest. "
                "No synthetic or placeholder scores are included."
            ),
        }
    )

    group_rows = []
    for (group, label), group_frame in readable.groupby(["group", "label"]):
        subset = group_frame[group_frame["predicted_class"].isin(["real", "ai_generated"])]
        if subset.empty:
            continue
        correct = (
            subset["predicted_class"].map({"real": 0, "ai_generated": 1}) == subset["label"]
        ).mean()
        group_rows.append(
            {
                "group": group,
                "label": "ai_generated" if label == 1 else "real",
                "images": int(len(group_frame)),
                "accuracy_on_decided": float(correct),
                "uncertain_rate": float((group_frame["predicted_class"] == "uncertain").mean()),
            }
        )

    artifact = {
        "artifact": args.output.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "module": "AI-generated image detection",
        "backend_status": status,
        "metrics": metrics,
        "per_group": group_rows,
        "limitations": [
            "Results apply only to this dataset and this detector.",
            "Uncertain predictions are abstentions and are excluded from decided-prediction metrics.",
            "Post-processing such as resizing or messaging-app compression can change results.",
        ],
    }

    settings.evaluation_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = args.output.with_name(args.output.stem + "_per_image.csv")
    frame.to_csv(per_image_path, index=False)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\n=== MODULE 2 EVALUATION ===")
    print(f"Backend: {status['kind']} | scored: {len(readable)} | uncertain: {uncertain_count}")
    print(
        f"Accuracy (decided): {metrics['accuracy']:.4f} | "
        f"Macro F1: {metrics['macro_f1']:.4f} | Coverage: {metrics['coverage']:.4f}"
    )
    print(f"Confusion matrix (rows=true real/AI): {metrics['confusion_matrix']}")
    print(f"\nSaved: {args.output}")
    print(f"Saved: {per_image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
