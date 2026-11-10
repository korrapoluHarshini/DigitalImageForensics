
from pathlib import Path
import hashlib
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(
    r"C:\Users\harsh\Documents\DigitalImageForensics"
)

SPLIT_CSV = PROJECT_ROOT / "experiments/dataset_split.csv"
PAIRS_CSV = PROJECT_ROOT / "experiments/localization_pairs.csv"

ORIGINAL_DIR = Path("results/validation_score_cache")
GAUSSIAN_DIR = Path("results/validation_patch_gaussian")

OUTPUT = Path("results/validation_method_comparison.csv")

ORIGINAL_THRESHOLD = 0.4
GAUSSIAN_THRESHOLD = 0.2


def metrics(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return iou, precision, recall, f1


def main():
    splits = pd.read_csv(SPLIT_CSV)
    pairs = pd.read_csv(PAIRS_CSV)

    validation = splits[
        (splits["split"] == "validation") &
        (splits["class_name"] == "Tampered")
    ].copy()

    validation["filename"] = validation["filename"].apply(
        lambda x: Path(str(x).replace("\\", "/")).name
    )

    pairs["filename"] = pairs["image_path"].apply(
        lambda x: Path(str(x).replace("\\", "/")).name
    )

    matched = validation.merge(pairs, on="filename")

    print("Matched images:", len(matched))

    results = []

    for _, row in matched.iterrows():
        filename = row["filename"]
        stem = Path(filename).stem

        file_hash = hashlib.sha1(
            filename.encode("utf-8")
        ).hexdigest()

        original_path = ORIGINAL_DIR / f"{file_hash}.npy"
        gaussian_path = (
            GAUSSIAN_DIR / stem / "patch_probability_map.npy"
        )
        mask_path = PROJECT_ROOT / row["mask_path"]

        if not original_path.exists() or not gaussian_path.exists():
            print("Missing prediction:", filename)
            continue

        original = np.load(original_path)
        gaussian = np.load(gaussian_path)

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            print("Missing mask:", filename)
            continue

        if original.shape != mask.shape or gaussian.shape != mask.shape:
            print("Dimension mismatch:", filename)
            continue

        gt = mask > 127

        original_metrics = metrics(
            original >= ORIGINAL_THRESHOLD, gt
        )

        gaussian_metrics = metrics(
            gaussian >= GAUSSIAN_THRESHOLD, gt
        )

        results.append({
            "filename": filename,
            "Original_IoU": original_metrics[0],
            "Original_Precision": original_metrics[1],
            "Original_Recall": original_metrics[2],
            "Original_F1": original_metrics[3],
            "Gaussian_IoU": gaussian_metrics[0],
            "Gaussian_Precision": gaussian_metrics[1],
            "Gaussian_Recall": gaussian_metrics[2],
            "Gaussian_F1": gaussian_metrics[3]
        })

    if len(results) != 767:
        raise RuntimeError(
            f"Expected 767 comparisons, got {len(results)}. "
            "Check missing predictions before interpreting results."
        )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT, index=False)

    print("\n========== FINAL COMPARISON ==========")
    print("Images compared:", len(df))

    for method in ["Original", "Gaussian"]:
        print(f"\n{method} MVSS-Net")
        for metric in ["IoU", "Precision", "Recall", "F1"]:
            value = df[f"{method}_{metric}"].mean() * 100
            print(f"{metric}: {value:.2f}%")

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
