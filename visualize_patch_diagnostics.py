
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT = Path(
    r"C:\Users\harsh\Documents\DigitalImageForensics"
)

items = [
    (
        "Tp_S_NNN_S_N_ani10157_ani10157_12465.jpg",
        Path("results/patch_tests/ani10157")
    ),
    (
        "Tp_S_NND_S_N_cha20040_cha20040_02039.tif",
        Path("results/patch_tests/cha20040")
    ),
    (
        "Tp_S_NRD_S_N_cha10154_cha10154_12266.jpg",
        Path("results/patch_tests/cha10154")
    )
]

pairs = pd.read_csv(
    PROJECT / "experiments" / "localization_pairs.csv"
)

output = Path("results/patch_tests/diagnostics")
output.mkdir(parents=True, exist_ok=True)

for filename, folder in items:

    row = pairs[
        pairs["image_path"].str.endswith(filename)
    ]

    if row.empty:
        print("Missing pair:", filename)
        continue

    image_path = PROJECT / row.iloc[0]["image_path"]
    mask_path = PROJECT / row.iloc[0]["mask_path"]

    image = cv2.imread(str(image_path))
    gt_image = cv2.imread(str(mask_path), 0)

    if image is None or gt_image is None:
        print("Could not load:", filename)
        continue

    gt = gt_image > 0

    scores = np.load(
        folder / "patch_probability_map.npy"
    )

    h, w = gt.shape

    if scores.shape != (h, w):
        scores = cv2.resize(
            scores,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

    # Ground-truth overlay: RED
    gt_overlay = image.copy()

    gt_overlay[gt] = (
        0.5 * image[gt] +
        0.5 * np.array([0, 0, 255])
    ).astype(np.uint8)

    panels = [image, gt_overlay]

    # Prediction overlays: GREEN
    for threshold in [0.1, 0.4]:

        predicted = scores >= threshold
        pred_overlay = image.copy()

        if np.any(predicted):
            pred_overlay[predicted] = (
                0.5 * image[predicted] +
                0.5 * np.array([0, 255, 0])
            ).astype(np.uint8)

        panels.append(pred_overlay)

    comparison = np.hstack(panels)

    save_name = Path(filename).stem + "_diagnostic.png"

    cv2.imwrite(
        str(output / save_name),
        comparison
    )

    print("Saved:", output / save_name)

print("\nAll diagnostic visualizations generated!")
