
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

# Dataset paths
DATA_ROOT = Path(
    r"C:\Users\harsh\Documents\DigitalImageForensics"
)

SPLIT_CSV = DATA_ROOT / "experiments" / "dataset_split.csv"
PAIRS_CSV = DATA_ROOT / "experiments" / "localization_pairs.csv"

# Patch prediction folder
PREDICTION_ROOT = Path("results/validation_patch_gaussian")

# Evaluation settings
THRESHOLD = 0.4


def calculate_metrics(prediction, ground_truth):

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    tp = np.logical_and(prediction, ground_truth).sum()
    fp = np.logical_and(prediction, ~ground_truth).sum()
    fn = np.logical_and(~prediction, ground_truth).sum()

    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = (2 * precision * recall) / (
        precision + recall + 1e-8
    )

    return iou, precision, recall, f1


def main():

    print("Loading validation dataset...")

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
        pairs_df,
        on="filename",
        how="inner"
    )

    print("Matched image-mask pairs:", len(matched))

    results = []

    for index, row in matched.iterrows():

        filename = row["filename"]

        image_name = Path(filename).stem

        prediction_path = (
            PREDICTION_ROOT /
            image_name /
            "patch_probability_map.npy"
        )

        mask_path = DATA_ROOT / row["mask_path"]

        if not prediction_path.exists():
            print("Missing prediction:", filename)
            continue

        probability_map = np.load(prediction_path)

        ground_truth = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if ground_truth is None:
            print("Missing mask:", filename)
            continue

        if probability_map.shape != ground_truth.shape:
            print("Dimension mismatch:", filename)
            continue

        prediction = probability_map >= THRESHOLD
        ground_truth = ground_truth > 127

        iou, precision, recall, f1 = calculate_metrics(
            prediction,
            ground_truth
        )

        results.append({
            "filename": filename,
            "IoU": iou,
            "Precision": precision,
            "Recall": recall,
            "F1_score": f1
        })

        if len(results) % 100 == 0:
            print("Evaluated:", len(results))

    if not results:
        raise RuntimeError("No images were evaluated.")

    df = pd.DataFrame(results)

    # Save individual image metrics
    df.to_csv(
        PREDICTION_ROOT / "patch_validation_per_image.csv",
        index=False
    )

    # Calculate mean per-image metrics
    print("\n========== VALIDATION RESULTS ==========")

    print("Images evaluated:", len(df))
    print("Threshold:", THRESHOLD)

    print(f"Mean IoU       : {df['IoU'].mean() * 100:.2f}%")
    print(f"Mean Precision : {df['Precision'].mean() * 100:.2f}%")
    print(f"Mean Recall    : {df['Recall'].mean() * 100:.2f}%")
    print(f"Mean F1-score  : {df['F1_score'].mean() * 100:.2f}%")

    print("\nResults saved in:")
    print(PREDICTION_ROOT.resolve())


if __name__ == "__main__":
    main()
