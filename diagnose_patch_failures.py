
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT = Path(r"C:\Users\harsh\Documents\DigitalImageForensics")
PATCH_ROOT = Path("results/patch_tests")

filenames = [
    "Tp_S_NRD_S_B_art10106_art10106_11589.jpg",
    "Tp_S_NNN_S_N_ani10157_ani10157_12465.jpg",
    "Tp_S_NND_S_N_cha20040_cha20040_02039.tif",
    "Tp_S_NRD_S_N_cha10154_cha10154_12266.jpg"
]

folders = [
    Path("results/patch_inference"),
    PATCH_ROOT / "ani10157",
    PATCH_ROOT / "cha20040",
    PATCH_ROOT / "cha10154"
]

pairs = pd.read_csv(
    PROJECT / "experiments" / "localization_pairs.csv"
)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

for filename, folder in zip(filenames, folders):

    row = pairs[
        pairs["image_path"].str.endswith(filename)
    ]

    if row.empty:
        print("Missing pair:", filename)
        continue

    mask_path = PROJECT / row.iloc[0]["mask_path"]
    gt_image = cv2.imread(str(mask_path), 0)

    scores = np.load(
        folder / "patch_probability_map.npy"
    )

    gt = gt_image > 0

    if scores.shape != gt.shape:
        scores = cv2.resize(
            scores,
            (gt.shape[1], gt.shape[0])
        )

    print("\nIMAGE:", filename)
    print("GT pixels:", int(gt.sum()))
    print("Maximum probability:", float(scores.max()))
    print("Mean probability:", float(scores.mean()))

    for threshold in thresholds:

        pred = scores >= threshold

        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()

        iou = intersection / (union + 1e-8)

        print(
            f"Threshold {threshold}: "
            f"Predicted pixels={int(pred.sum())}, "
            f"IoU={iou:.4f}"
        )
