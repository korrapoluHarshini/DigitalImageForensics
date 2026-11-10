
import cv2
import numpy as np
import pandas as pd
import hashlib
from pathlib import Path

ROOT = Path(r"C:\Users\harsh\Documents\DigitalImageForensics")
RESULTS = Path("results")

filename = "Tp_S_NNN_S_N_sec20026_sec20026_01616.tif"
THRESHOLD = 0.4

# Find corresponding ground-truth mask
pairs = pd.read_csv(
    ROOT / "experiments" / "localization_pairs.csv"
)

pairs["filename"] = pairs["image_path"].apply(
    lambda x: Path(str(x).replace("\\", "/")).name
)

pair = pairs[pairs["filename"] == filename]

if pair.empty:
    raise ValueError("Ground-truth mask not found")

mask_path = ROOT / pair.iloc[0]["mask_path"]

mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise ValueError("Cannot read ground-truth mask")

gt = (mask > 0).astype(np.uint8)

# Load original MVSS-Net probability map
file_hash = hashlib.sha1(
    filename.encode("utf-8")
).hexdigest()

original_path = (
    RESULTS / "validation_score_cache" /
    f"{file_hash}.npy"
)

original = np.load(original_path)

# Load individual and combined multi-scale maps
test_dir = RESULTS / "multiscale_tests" / "sec20026"

predictions = {
    "Original MVSS-Net": original,
    "Scale 512": np.load(test_dir / "probability_scale_512.npy"),
    "Scale 768": np.load(test_dir / "probability_scale_768.npy"),
    "Scale 1024": np.load(test_dir / "probability_scale_1024.npy"),
    "Combined Multi-scale": np.load(
        test_dir / "multiscale_probability_map.npy"
    )
}


def calculate_iou(probability):

    if probability.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: {probability.shape} vs {gt.shape}"
        )

    prediction = (
        probability >= THRESHOLD
    ).astype(np.uint8)

    intersection = np.logical_and(
        gt == 1, prediction == 1
    ).sum()

    union = np.logical_or(
        gt == 1, prediction == 1
    ).sum()

    return intersection / union if union else 0.0


print("\nINDIVIDUAL SCALE COMPARISON")
print("Image:", filename)
print("Threshold:", THRESHOLD)

for name, probability in predictions.items():

    iou = calculate_iou(probability)

    print(f"{name}: IoU = {iou:.4f}")
