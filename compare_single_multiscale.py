
import cv2
import numpy as np
import pandas as pd
import hashlib
from pathlib import Path

filename = "Tp_S_NNN_S_N_sec20026_sec20026_01616.tif"

ROOT = Path(r"C:\Users\harsh\Documents\DigitalImageForensics")
RESULTS = Path("results")

pairs = pd.read_csv(
    ROOT / "experiments" / "localization_pairs.csv"
)

pair = pairs[
    pairs["image_path"].apply(
        lambda x: Path(str(x).replace("\\", "/")).name
    ) == filename
].iloc[0]

mask_path = ROOT / pair["mask_path"]

mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
gt = (mask > 0).astype(np.uint8)

file_hash = hashlib.sha1(
    filename.encode("utf-8")
).hexdigest()

original_path = (
    RESULTS / "validation_score_cache" / f"{file_hash}.npy"
)

multiscale_path = (
    RESULTS / "multiscale_tests" / "sec20026" /
    "multiscale_probability_map.npy"
)

original = np.load(original_path)
multiscale = np.load(multiscale_path)

def calculate_iou(probability, threshold=0.4):

    prediction = (probability >= threshold).astype(np.uint8)

    intersection = np.logical_and(
        gt == 1, prediction == 1
    ).sum()

    union = np.logical_or(
        gt == 1, prediction == 1
    ).sum()

    return intersection / union if union else 0.0

print("\nSINGLE IMAGE COMPARISON")
print("Original MVSS-Net IoU:", round(calculate_iou(original), 4))
print("Multi-scale MVSS-Net IoU:", round(calculate_iou(multiscale), 4))
