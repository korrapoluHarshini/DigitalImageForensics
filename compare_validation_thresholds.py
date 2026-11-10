
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

DATA_ROOT = Path(
    r"C:\Users\harsh\Documents\DigitalImageForensics"
)

SPLIT_CSV = DATA_ROOT / "experiments" / "dataset_split.csv"
PAIRS_CSV = DATA_ROOT / "experiments" / "localization_pairs.csv"

PREDICTION_ROOT = Path("results/validation_patch_gaussian")

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def calculate_metrics(prediction, ground_truth):

    tp = np.logical_and(prediction, ground_truth).sum()
    fp = np.logical_and(prediction, ~ground_truth).sum()
    fn = np.logical_and(~prediction, ground_truth).sum()

    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return iou, precision, recall, f1


def main():

    split_df = pd.read_csv(SPLIT_CSV)
    pairs_df = pd.read_csv(PAIRS_CSV)

    validation = split_df[
        (split_df["split"] == "validation") &
        (split_df["class_name"] == "Tampered")
    ].copy()

    validation["filename"] = validation["filename"].apply(
        lambda x: Path(str(x)).name
    )

    pairs_df["filename"] = pairs_df["image_path"].apply(
        lambda x: Path(str(x)).name
    )

    matched = validation.merge(
        pairs_df, on="filename", how="inner"
    )

    # Load all valid probability maps and ground-truth masks once
    samples = []

    print("Loading saved predictions...")

    for _, row in matched.iterrows():

        stem = Path(row["filename"]).stem

        prediction_path = (
            PREDICTION_ROOT / stem / "patch_probability_map.npy"
        )

        mask_path = DATA_ROOT / row["mask_path"]

        if not prediction_path.exists():
            continue

        probability = np.load(prediction_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None or probability.shape != mask.shape:
            continue

        samples.append((probability, mask > 127))

    print("Images loaded:", len(samples))

    results = []

    for threshold in THRESHOLDS:

        image_metrics = []

        for probability, ground_truth in samples:

            prediction = probability >= threshold

            metrics = calculate_metrics(
                prediction, ground_truth
            )

            image_metrics.append(metrics)

        values = np.array(image_metrics)

        results.append({
            "Threshold": threshold,
            "Mean_IoU": values[:, 0].mean() * 100,
            "Mean_Precision": values[:, 1].mean() * 100,
            "Mean_Recall": values[:, 2].mean() * 100,
            "Mean_F1": values[:, 3].mean() * 100
        })

    df = pd.DataFrame(results)

    output_path = (
        PREDICTION_ROOT / "threshold_comparison.csv"
    )

    df.to_csv(output_path, index=False)

    print("\n========== THRESHOLD COMPARISON ==========")
    print(df.round(2).to_string(index=False))

    best = df.loc[df["Mean_IoU"].idxmax()]

    print("\n========== BEST THRESHOLD ==========")
    print("Threshold:", best["Threshold"])
    print(f"Mean IoU: {best['Mean_IoU']:.2f}%")
    print(f"Mean F1: {best['Mean_F1']:.2f}%")

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
