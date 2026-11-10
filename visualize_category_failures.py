
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(
    r"C:\Users\harsh\Documents\DigitalImageForensics"
)

SELECTED_CSV = Path(
    "results/final_test/category_analysis/selected_failed_samples.csv"
)

PAIRS_CSV = PROJECT_ROOT / "experiments" / "localization_pairs.csv"

CACHE_DIR = Path("results/final_test/score_cache")

OUTPUT_DIR = Path(
    "results/final_test/category_analysis/visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.4


def main():

    selected = pd.read_csv(SELECTED_CSV)
    pairs = pd.read_csv(PAIRS_CSV)

    pairs["key"] = pairs["image_path"].apply(
        lambda x: Path(str(x).replace("\\", "/")).name
    )

    for _, sample in selected.iterrows():

        filename = sample["filename"]

        print("\nProcessing:", filename)

        match = pairs[pairs["key"] == filename]

        if match.empty:
            print("Image-mask pair not found.")
            continue

        row = match.iloc[0]

        image_path = PROJECT_ROOT / row["image_path"]
        mask_path = PROJECT_ROOT / row["mask_path"]

        image = cv2.imread(str(image_path))
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        import hashlib

        cache_name = hashlib.sha1(
            filename.encode("utf-8")
        ).hexdigest() + ".npy"

        score_path = CACHE_DIR / cache_name

        if image is None or gt is None or not score_path.exists():
            print("Missing image, mask, or cached prediction.")
            continue

        scores = np.load(score_path)

        if scores.shape != gt.shape:
            print("Prediction and mask dimensions differ.")
            continue

        predicted = scores >= THRESHOLD
        actual = gt > 127

        # Create side-by-side comparison
        height, width = image.shape[:2]

        original = image.copy()

        gt_overlay = image.copy()
        gt_red = np.zeros_like(image)
        gt_red[:, :, 2] = 255

        gt_overlay[actual] = cv2.addWeighted(
            image[actual], 0.5,
            gt_red[actual], 0.5, 0
        )

        pred_overlay = image.copy()

        if np.any(predicted):
            pred_overlay[predicted] = (
                0.5 * image[predicted] +
                0.5 * np.array([0, 255, 0])
            ).astype(np.uint8)

        comparison = np.hstack([
            original,
            gt_overlay,
            pred_overlay
        ])

        output_path = OUTPUT_DIR / (
            Path(filename).stem + "_comparison.png"
        )

        cv2.imwrite(str(output_path), comparison)

        print("Saved:", output_path)

    print("\nAll available comparisons generated!")
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
