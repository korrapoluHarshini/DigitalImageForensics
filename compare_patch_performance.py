
import cv2
import numpy as np
import pandas as pd
import hashlib
from pathlib import Path

PROJECT = Path(r"C:\Users\harsh\Documents\DigitalImageForensics")
ROOT = Path("results/final_test")

filename = "Tp_S_NRD_S_B_art10106_art10106_11589.jpg"
threshold = 0.4

# Find matching ground-truth mask
pairs = pd.read_csv(
    PROJECT / "experiments" / "localization_pairs.csv"
)

row = pairs[
    pairs["image_path"].str.endswith(filename)
].iloc[0]

mask_path = PROJECT / row["mask_path"]
gt = cv2.imread(str(mask_path), 0) > 0

# Load original cached prediction
cache_name = hashlib.sha1(
    filename.encode("utf-8")
).hexdigest() + ".npy"

original_scores = np.load(
    ROOT / "score_cache" / cache_name
)

# Load patch-based prediction
patch_scores = cv2.imread(
    "results/patch_inference/patch_probability_map.png",
    cv2.IMREAD_GRAYSCALE
).astype(np.float32) / 255.0

def evaluate(scores, gt):
    if scores.shape != gt.shape:
        scores = cv2.resize(
            scores,
            (gt.shape[1], gt.shape[0])
        )

    pred = scores >= threshold

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    precision = intersection / (pred.sum() + 1e-8)
    recall = intersection / (gt.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

print("\nORIGINAL INFERENCE")
print(evaluate(original_scores, gt))

print("\nPATCH-BASED INFERENCE")
print(evaluate(patch_scores, gt))
