
from pathlib import Path
import hashlib

import cv2
import numpy as np
import pandas as pd
from PIL import Image


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(
    r"C:\Users\harsh\Documents\DigitalImageForensics"
)

PAIRS_CSV = PROJECT_ROOT / "experiments" / "localization_pairs.csv"

CACHE_DIR = Path("results/final_test/score_cache")

TEST_RESULTS = Path(
    "results/final_test/final_test_per_image.csv"
)

OUTPUT_PATH = Path(
    "results/final_test/inference_method_comparison.csv"
)

THRESHOLD = 0.4


# --------------------------------------------------
# CACHE FILE NAME
# --------------------------------------------------

def get_cache_path(filename):

    file_hash = hashlib.sha1(
        filename.encode("utf-8")
    ).hexdigest()

    return CACHE_DIR / f"{file_hash}.npy"


# --------------------------------------------------
# OFFICIAL PIL CONVERSION
# --------------------------------------------------

def official_pil_conversion(scores):

    # Equivalent to torchvision.transforms.ToPILImage()
    pil_image = Image.fromarray(
        scores.astype(np.float32),
        mode="F"
    )

    return np.array(pil_image)


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(pred, gt):

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = (
        2 * precision * recall
        / (precision + recall + 1e-8)
    )

    return iou, precision, recall, f1


# --------------------------------------------------
# MAIN COMPARISON
# --------------------------------------------------

def main():

    pairs = pd.read_csv(PAIRS_CSV)
    existing_results = pd.read_csv(TEST_RESULTS)

    pairs["key"] = pairs["image_path"].apply(
        lambda x: Path(str(x).replace("\\", "/")).name
    )

    # Only inspect the test images already evaluated
    filenames = existing_results["filename"].tolist()

    results = []

    print("Comparing inference output methods...")
    print("Images:", len(filenames))

    for position, filename in enumerate(
        filenames,
        start=1
    ):

        print(
            f"\nProcessing {position}/{len(filenames)}: {filename}"
        )

        matched = pairs[pairs["key"] == filename]

        if matched.empty:
            print("Matching mask not found.")
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

        # Method 1: Current floating-point scores
        direct_scores = scores

        # Method 2: Official PIL conversion
        pil_scores = official_pil_conversion(scores)

        # Calculate predictions at the same threshold
        direct_pred = direct_scores >= THRESHOLD
        pil_pred = pil_scores >= THRESHOLD

        direct_metrics = calculate_metrics(
            direct_pred,
            ground_truth
        )

        pil_metrics = calculate_metrics(
            pil_pred,
            ground_truth
        )

        results.append({
            "filename": filename,

            "Direct_IoU": direct_metrics[0],
            "Direct_Precision": direct_metrics[1],
            "Direct_Recall": direct_metrics[2],
            "Direct_F1": direct_metrics[3],

            "PIL_IoU": pil_metrics[0],
            "PIL_Precision": pil_metrics[1],
            "PIL_Recall": pil_metrics[2],
            "PIL_F1": pil_metrics[3],

            "Prediction_Maps_Identical": np.array_equal(
                direct_scores,
                pil_scores
            )
        })

    if not results:
        print("No images available for comparison.")
        return

    df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n========== AVERAGE COMPARISON ==========")

    columns = [
        "Direct_IoU",
        "Direct_Precision",
        "Direct_Recall",
        "Direct_F1",
        "PIL_IoU",
        "PIL_Precision",
        "PIL_Recall",
        "PIL_F1"
    ]

    print(df[columns].mean().to_string())

    print(
        "\nIdentical prediction maps:",
        df["Prediction_Maps_Identical"].sum(),
        "/",
        len(df)
    )

    print("\nResults saved:", OUTPUT_PATH)
    print("Comparison completed!")


if __name__ == "__main__":
    main()
