
import pandas as pd
import numpy as np
import hashlib
import cv2
from pathlib import Path

ROOT = Path(r"C:\Users\harsh\Documents\DigitalImageForensics")
RESULTS = Path("results")

CACHE_DIR = RESULTS / "validation_score_cache"
SPLIT_FILE = ROOT / "experiments" / "dataset_split.csv"
PAIRS_FILE = ROOT / "experiments" / "localization_pairs.csv"

THRESHOLD = 0.4

split = pd.read_csv(SPLIT_FILE)
pairs = pd.read_csv(PAIRS_FILE)

# Extract filename from image_path in localization pairs
pairs["filename"] = pairs["image_path"].apply(
    lambda x: Path(str(x).replace("\\", "/")).name
)

# Select validation tampered images
validation = split[
    (split["split"] == "validation") &
    (split["class_name"].str.lower() == "tampered")
]

# Match validation images with their ground-truth masks
validation_pairs = validation.merge(
    pairs,
    on="filename",
    how="inner"
)

records = []

for _, row in validation_pairs.iterrows():

    filename = row["filename"]

    mask_path = ROOT / row["mask_path"]
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print("Mask missing:", filename)
        continue

    mask = (mask > 0).astype(np.uint8)

    file_hash = hashlib.sha1(
        filename.encode("utf-8")
    ).hexdigest()

    probability_path = CACHE_DIR / f"{file_hash}.npy"

    if not probability_path.exists():
        print("Probability map missing:", filename)
        continue

    probability = np.load(probability_path)

    if probability.shape != mask.shape:
        print("Shape mismatch:", filename)
        continue

    prediction = (probability >= THRESHOLD).astype(np.uint8)

    gt_pixels = int(mask.sum())
    pred_pixels = int(prediction.sum())

    intersection = int((mask * prediction).sum())
    union = int(((mask + prediction) > 0).sum())

    iou = intersection / union if union else 0.0

    inside_mean = (
        float(probability[mask == 1].mean())
        if gt_pixels else 0.0
    )

    outside_mean = (
        float(probability[mask == 0].mean())
        if np.any(mask == 0) else 0.0
    )

    if pred_pixels == 0:
        category = "No prediction"
    elif intersection == 0:
        category = "Wrong localization"
    elif iou < 0.5:
        category = "Partial localization"
    else:
        category = "Good localization"

    records.append({
        "filename": filename,
        "GT_area_percent": gt_pixels / mask.size * 100,
        "Predicted_area_percent": pred_pixels / prediction.size * 100,
        "Max_probability": float(probability.max()),
        "Mean_probability_inside_GT": inside_mean,
        "Mean_probability_outside_GT": outside_mean,
        "IoU": iou,
        "Failure_category": category
    })

report = pd.DataFrame(records)

output = RESULTS / "localization_diagnostic_report.csv"
report.to_csv(output, index=False)

print("\nDIAGNOSTIC COMPLETED")
print("Images analysed:", len(report))

print("\nFailure category summary:")
print(report["Failure_category"].value_counts().to_string())

print("\nAverage diagnostic metrics:")
print(
    report[
        [
            "GT_area_percent",
            "Predicted_area_percent",
            "Max_probability",
            "Mean_probability_inside_GT",
            "Mean_probability_outside_GT",
            "IoU"
        ]
    ].mean().round(4).to_string()
)

print("\nSaved report:", output)
