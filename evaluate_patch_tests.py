
import cv2
import numpy as np
import pandas as pd
import hashlib
from pathlib import Path

PROJECT = Path(r"C:\Users\harsh\Documents\DigitalImageForensics")
ROOT = Path("results/final_test")
PATCH_ROOT = Path("results/patch_tests")

filenames = [
    "Tp_S_NRD_S_B_art10106_art10106_11589.jpg",
    "Tp_S_NNN_S_N_ani10157_ani10157_12465.jpg",
    "Tp_S_NND_S_N_cha20040_cha20040_02039.tif",
    "Tp_S_NRD_S_N_cha10154_cha10154_12266.jpg"
]

patch_paths = [
    Path("results/patch_inference/patch_probability_map.npy"),
    Path("results/patch_tests/ani10157/patch_probability_map.npy"),
    Path("results/patch_tests/cha20040/patch_probability_map.npy"),
    Path("results/patch_tests/cha10154/patch_probability_map.npy")
]

pairs = pd.read_csv(
    PROJECT / "experiments" / "localization_pairs.csv"
)

def evaluate(scores, gt, threshold=0.4):
    if scores.shape != gt.shape:
        scores = cv2.resize(
            scores,
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    pred = scores >= threshold

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    precision = intersection / (pred.sum() + 1e-8)
    recall = intersection / (gt.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return iou, precision, recall, f1

results = []

for filename, patch_path in zip(filenames, patch_paths):

    row = pairs[
        pairs["image_path"].str.endswith(filename)
    ]

    if row.empty:
        print("Mask pair missing:", filename)
        continue

    mask_path = PROJECT / row.iloc[0]["mask_path"]
    gt_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if gt_image is None:
        print("Mask unreadable:", filename)
        continue

    gt = gt_image > 0

    cache_name = hashlib.sha1(
        filename.encode("utf-8")
    ).hexdigest() + ".npy"

    original_path = ROOT / "score_cache" / cache_name
    

    if not original_path.exists() or not patch_path.exists():
        print("Prediction missing:", filename)
        continue

    original = np.load(original_path)
    patch = np.load(patch_path)

    original_metrics = evaluate(original, gt)
    patch_metrics = evaluate(patch, gt)

    results.append({
        "filename": filename,
        "Original_IoU": original_metrics[0],
        "Patch_IoU": patch_metrics[0],
        "Original_F1": original_metrics[3],
        "Patch_F1": patch_metrics[3],
        "Original_Recall": original_metrics[2],
        "Patch_Recall": patch_metrics[2],
        "Original_Precision": original_metrics[1],
        "Patch_Precision": patch_metrics[1]
    })

df = pd.DataFrame(results)

print("\nORIGINAL VS PATCH-BASED EVALUATION")
print(df.round(4).to_string(index=False))

if not df.empty:
    print("\nAVERAGE METRICS")
    print(
        df[[
            "Original_IoU", "Patch_IoU",
            "Original_F1", "Patch_F1",
            "Original_Precision", "Patch_Precision",
            "Original_Recall", "Patch_Recall"
        ]].mean().round(4)
    )

df.to_csv(
    PATCH_ROOT / "patch_comparison_results.csv",
    index=False
)

print("\nSaved: results/patch_tests/patch_comparison_results.csv")
