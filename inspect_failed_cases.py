
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(
    r"C:\Users\harsh\Documents\DigitalImageForensics"
)

PAIRS_CSV = PROJECT_ROOT / "experiments" / "localization_pairs.csv"

CACHE_DIR = Path("results/final_test/score_cache")

OUTPUT_DIR = Path("results/final_test/failed_case_visualizations")

THRESHOLD = 0.4


# --------------------------------------------------
# FAILED IMAGE FILENAMES
# --------------------------------------------------

FAILED_IMAGES = [
    "Tp_S_NRN_S_N_ind20019_ind20019_01760.tif",
    "Tp_S_NRD_S_B_art10106_art10106_11589.jpg",
    "Tp_S_NRD_S_N_cha10154_cha10154_12266.jpg"
]


# --------------------------------------------------
# CACHE FILE NAME
# --------------------------------------------------

import hashlib


def get_cache_path(filename):

    file_hash = hashlib.sha1(
        filename.encode("utf-8")
    ).hexdigest()

    return CACHE_DIR / f"{file_hash}.npy"


# --------------------------------------------------
# MAIN DIAGNOSTIC
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pairs = pd.read_csv(PAIRS_CSV)

    pairs["key"] = pairs["image_path"].apply(
        lambda x: Path(str(x).replace("\\", "/")).name
    )

    for filename in FAILED_IMAGES:

        print("\nInspecting:", filename)

        matched = pairs[pairs["key"] == filename]

        if matched.empty:

            print("Image not found in localization pairs.")
            continue

        row = matched.iloc[0]

        image_path = PROJECT_ROOT / row["image_path"]
        mask_path = PROJECT_ROOT / row["mask_path"]

        image = cv2.imread(str(image_path))
        gt = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        cache_path = get_cache_path(filename)

        if image is None or gt is None:

            print("Unable to read image or mask.")
            continue

        if not cache_path.exists():

            print("Cached prediction scores not found.")
            continue

        scores = np.load(cache_path)

        if scores.shape != gt.shape:

            print("Prediction and mask dimensions differ.")
            continue

        original = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        ground_truth = gt > 127
        prediction = scores >= THRESHOLD

        # Red = ground truth
        # Green = prediction
        overlay = original.copy()

        overlay[ground_truth] = (
            0.5 * overlay[ground_truth] +
            0.5 * np.array([255, 0, 0])
        ).astype(np.uint8)

        overlay[prediction] = (
            0.5 * overlay[prediction] +
            0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        axes[0].imshow(original)
        axes[0].set_title("Original Image")

        axes[1].imshow(ground_truth, cmap="gray")
        axes[1].set_title("Ground Truth Mask")

        axes[2].imshow(prediction, cmap="gray")
        axes[2].set_title("Predicted Mask (0.4)")

        axes[3].imshow(overlay)
        axes[3].set_title("Overlay: Red=GT, Green=Prediction")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(filename, fontsize=10)
        plt.tight_layout()

        output_path = OUTPUT_DIR / (
            Path(filename).stem + "_comparison.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        print("Saved:", output_path)

    print("\nDiagnostic visualization completed!")


if __name__ == "__main__":
    main()
