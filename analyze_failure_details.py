import os
import hashlib
import cv2
import numpy as np
import pandas as pd

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(
    r"C:\Users\harsh\Documents\DigitalImageForensics"
)

RESULTS_ROOT = Path("results/final_test")
CACHE_DIR = RESULTS_ROOT / "score_cache"

TEST_CSV = RESULTS_ROOT / "final_test_per_image.csv"

PAIRS_CSV = (
    PROJECT_ROOT / "experiments" / "localization_pairs.csv"
)

OUTPUT_DIR = RESULTS_ROOT / "failure_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.4

# Load CSV files
test_df = pd.read_csv(TEST_CSV)
pairs_df = pd.read_csv(PAIRS_CSV)

# Map filenames to image and mask paths
pairs_df["filename"] = pairs_df["image_path"].apply(
    lambda x: os.path.basename(x)
)

pair_lookup = pairs_df.set_index("filename").to_dict("index")

results = []

for _, row in test_df.iterrows():

    filename = row["filename"]

    if filename not in pair_lookup:
        print("Pair not found:", filename)
        continue

    image_path = PROJECT_ROOT / pair_lookup[filename]["image_path"]
    mask_path = PROJECT_ROOT / pair_lookup[filename]["mask_path"]

    # Locate cached prediction
    cache_name = hashlib.sha1(
        filename.encode("utf-8")
    ).hexdigest() + ".npy"

    cache_path = CACHE_DIR / cache_name

    if not cache_path.exists():
        print("Cache missing:", filename)
        continue

    # Load ground truth and prediction
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    scores = np.load(cache_path)

    if gt is None:
        print("Mask unreadable:", filename)
        continue

    # Resize prediction if necessary
    if scores.shape != gt.shape:
        scores = cv2.resize(
            scores,
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    gt_binary = gt > 0
    pred_binary = scores >= THRESHOLD

    gt_area = int(np.count_nonzero(gt_binary))
    pred_area = int(np.count_nonzero(pred_binary))

    results.append({
        "filename": filename,
        "IoU": row["IoU"],
        "F1": row["F1"],
        "GT_Area_Pixels": gt_area,
        "Predicted_Area_Pixels": pred_area,
        "GT_Area_Percentage": 100 * gt_area / gt.size,
        "Predicted_Area_Percentage": 100 * pred_area / pred_binary.size,
        "Max_Probability": float(np.max(scores)),
        "Mean_Probability": float(np.mean(scores)),
        "Threshold": THRESHOLD
    })

    print("Processed:", filename)

# Save detailed results
details = pd.DataFrame(results)

details.to_csv(
    OUTPUT_DIR / "detailed_failure_analysis.csv",
    index=False
)

# Compare successful and failed images
details["Result"] = np.where(
    details["IoU"] == 0,
    "Failed",
    "Successful"
)

summary = details.groupby("Result").agg({
    "filename": "count",
    "IoU": "mean",
    "F1": "mean",
    "GT_Area_Percentage": "mean",
    "Predicted_Area_Percentage": "mean",
    "Max_Probability": "mean",
    "Mean_Probability": "mean"
}).rename(columns={"filename": "Image_Count"})

summary.to_csv(
    OUTPUT_DIR / "success_vs_failure_summary.csv"
)

print("\nDETAILED ANALYSIS COMPLETED")
print("\nImages processed:", len(details))
print("\nSUCCESS VS FAILURE SUMMARY")
print(summary)

print("\nOutput folder:", OUTPUT_DIR)
# Further analyse zero-IoU failures

failed = details[details["IoU"] == 0].copy()

no_prediction = failed[
    failed["Predicted_Area_Pixels"] == 0
]

wrong_location = failed[
    failed["Predicted_Area_Pixels"] > 0
]

print("\nZERO-IOU FAILURE BREAKDOWN")

print("Total failed:", len(failed))
print("No predicted pixels:", len(no_prediction))
print("Predictions outside GT:", len(wrong_location))

print("\nNo-prediction examples:")
print(no_prediction["filename"].head(10).to_string(index=False))

print("\nWrong-location examples:")
print(wrong_location["filename"].head(10).to_string(index=False))

# Save both groups
no_prediction.to_csv(
    OUTPUT_DIR / "no_prediction_failures.csv",
    index=False
)

wrong_location.to_csv(
    OUTPUT_DIR / "wrong_location_failures.csv",
    index=False
)
# Failure type analysis by manipulation category

failed = details[details["IoU"] == 0].copy()

failed["Manipulation_Type"] = failed["filename"].apply(
    lambda x: x.split("_")[1]
)

failed["Manipulation_Method"] = failed["filename"].apply(
    lambda x: x.split("_")[2].upper()
)

failed["Failure_Type"] = np.where(
    failed["Predicted_Area_Pixels"] == 0,
    "No Prediction",
    "Wrong Location"
)

category_summary = failed.groupby(
    ["Manipulation_Type", "Manipulation_Method", "Failure_Type"]
).size().reset_index(name="Failure_Count")

print("\nFAILURE BREAKDOWN BY CATEGORY")
print(category_summary.to_string(index=False))

category_summary.to_csv(
    OUTPUT_DIR / "failure_category_breakdown.csv",
    index=False
)

print("\nSaved: failure_category_breakdown.csv")
# Complete category-wise performance analysis

all_images = details.copy()

all_images["Manipulation_Type"] = all_images["filename"].apply(
    lambda x: x.split("_")[1]
)

all_images["Manipulation_Method"] = all_images["filename"].apply(
    lambda x: x.split("_")[2].upper()
)

all_images["Failure_Type"] = np.where(
    all_images["IoU"] > 0,
    "Successful",
    np.where(
        all_images["Predicted_Area_Pixels"] == 0,
        "No Prediction",
        "Wrong Location"
    )
)

category_comparison = all_images.pivot_table(
    index=["Manipulation_Type", "Manipulation_Method"],
    columns="Failure_Type",
    values="filename",
    aggfunc="count",
    fill_value=0
).reset_index()

for col in ["Successful", "No Prediction", "Wrong Location"]:
    if col not in category_comparison.columns:
        category_comparison[col] = 0

category_comparison["Total_Images"] = (
    category_comparison["Successful"]
    + category_comparison["No Prediction"]
    + category_comparison["Wrong Location"]
)

category_comparison["Failure_Percentage"] = (
    100 * (
        category_comparison["No Prediction"]
        + category_comparison["Wrong Location"]
    ) / category_comparison["Total_Images"]
)

category_comparison = category_comparison.sort_values(
    "Failure_Percentage",
    ascending=False
)

category_comparison.to_csv(
    OUTPUT_DIR / "complete_category_comparison.csv",
    index=False
)

print("\nCOMPLETE CATEGORY COMPARISON")
print(category_comparison.to_string(index=False))

print("\nSaved: complete_category_comparison.csv")