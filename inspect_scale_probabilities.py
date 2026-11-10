
import numpy as np
import cv2
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\harsh\Documents\DigitalImageForensics")

filename = "Tp_S_NNN_S_N_sec20026_sec20026_01616.tif"

pairs = pd.read_csv(
    ROOT / "experiments" / "localization_pairs.csv"
)

pairs["filename"] = pairs["image_path"].apply(
    lambda x: Path(str(x).replace("\\", "/")).name
)

pair = pairs[pairs["filename"] == filename].iloc[0]

mask_path = ROOT / pair["mask_path"]

mask = cv2.imread(
    str(mask_path),
    cv2.IMREAD_GRAYSCALE
)

gt = mask > 0

test_dir = Path("results/multiscale_tests/sec20026")

scales = [512, 768, 1024]

print("\nPROBABILITY ANALYSIS")
print("Image:", filename)
print("Ground-truth pixels:", int(gt.sum()))

for scale in scales:

    probability = np.load(
        test_dir / f"probability_scale_{scale}.npy"
    )

    inside = probability[gt]
    outside = probability[~gt]

    print(f"\nScale {scale}")

    print("Maximum inside GT:", float(inside.max()))
    print("Mean inside GT:", float(inside.mean()))
    print("Maximum outside GT:", float(outside.max()))
    print("Mean outside GT:", float(outside.mean()))

    for threshold in [0.1, 0.2, 0.3, 0.4]:

        prediction = probability >= threshold

        intersection = np.logical_and(
            gt, prediction
        ).sum()

        union = np.logical_or(
            gt, prediction
        ).sum()

        iou = intersection / union if union else 0

        print(
            f"Threshold {threshold}: IoU = {iou:.4f}"
        )

