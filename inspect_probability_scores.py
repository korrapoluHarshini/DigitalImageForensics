
from pathlib import Path
import hashlib

import cv2
import numpy as np
import pandas as pd


# Paths
PROJECT_ROOT = Path(
    r"C:\Users\harsh\Documents\DigitalImageForensics"
)

PAIRS_CSV = PROJECT_ROOT / "experiments" / "localization_pairs.csv"

CACHE_DIR = Path("results/final_test/score_cache")

OUTPUT_PATH = Path(
    "results/final_test/probability_score_analysis.csv"
)

THRESHOLDS = [0.1, 0.2, 0.3, 0.4]


# Failed images
FAILED_IMAGES = [
    "Tp_S_NRN_S_N_ind20019_ind20019_01760.tif",
    "Tp_S_NRD_S_B_art10106_art10106_11589.jpg",
    "Tp_S_NRD_S_N_cha10154_cha10154_12266.jpg"
]


def get_cache_path(filename):

    file_hash = hashlib.sha1(
        filename.encode("utf-8")
    ).hexdigest()

    return CACHE_DIR / f"{file_hash}.npy"


def main():

    pairs = pd.read_csv(PAIRS_CSV)

    pairs["key"] = pairs["image_path"].apply(
        lambda x: Path(str(x).replace("\\", "/")).name
    )

    results = []

    for filename in FAILED_IMAGES:

        print("\nAnalysing:", filename)

        matched = pairs[pairs["key"] == filename]

        if matched.empty:
            print("Image not found in dataset pairs.")
            continue

        row = matched.iloc[0]

        mask_path = PROJECT_ROOT / row["mask_path"]

        gt = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        cache_path = get_cache_path(filename)

        if gt is None or not cache_path.exists():
            print("Mask or cached scores unavailable.")
            continue

        scores = np.load(cache_path)

        if scores.shape != gt.shape:
            print("Dimension mismatch.")
            continue

        ground_truth = gt > 127

        print("Maximum probability:", scores.max())
        print("Mean probability:", scores.mean())
        print("Ground-truth pixels:", ground_truth.sum())

        record = {
            "filename": filename,
            "Max_Probability": scores.max(),
            "Mean_Probability": scores.mean(),
            "Ground_Truth_Pixels": ground_truth.sum()
        }

        for threshold in THRESHOLDS:

            pred = scores >= threshold

            record[f"Pixels_Above_{threshold}"] = int(pred.sum())

            overlap = np.logical_and(
                pred,
                ground_truth
            ).sum()

            record[f"Overlap_At_{threshold}"] = int(overlap)

            print(
                f"Threshold {threshold}: "
                f"Predicted pixels = {pred.sum()}, "
                f"Overlap = {overlap}"
            )

        results.append(record)

    if results:

        df = pd.DataFrame(results)

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            OUTPUT_PATH,
            index=False
        )

        print("\nProbability analysis saved:", OUTPUT_PATH)

        print("\n", df.to_string(index=False))

    print("\nProbability analysis completed!")


if __name__ == "__main__":
    main()
