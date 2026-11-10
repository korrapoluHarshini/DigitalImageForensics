
import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from models.mvssnet import get_mvss
from common.transforms import direct_val


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(
    r"C:\Users\harsh\Documents\DigitalImageForensics"
)

SPLIT_CSV = PROJECT_ROOT / "experiments" / "dataset_split.csv"
PAIRS_CSV = PROJECT_ROOT / "experiments" / "localization_pairs.csv"

CHECKPOINT = Path("ckpt/mvssnet_casia.pt")

THRESHOLD = 0.4
IMAGE_SIZE = 512

OUTPUT_DIR = Path("results/final_test")
CACHE_DIR = OUTPUT_DIR / "score_cache"

SUMMARY_PATH = OUTPUT_DIR / "final_test_summary.csv"
DETAILS_PATH = OUTPUT_DIR / "final_test_per_image.csv"
SKIPPED_PATH = OUTPUT_DIR / "skipped_images.csv"


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def load_model():

    print("\nLoading MVSS-Net...")

    model = get_mvss(
        backbone="resnet50",
        pretrained_base=False,
        nclass=1,
        sobel=True,
        constrain=True,
        n_input=3
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    print("MVSS-Net loaded successfully!")

    return model


# --------------------------------------------------
# CALCULATE METRICS
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
# CACHE FILE NAME
# --------------------------------------------------

def get_cache_path(filename):

    file_hash = hashlib.sha1(
        filename.encode("utf-8")
    ).hexdigest()

    return CACHE_DIR / f"{file_hash}.npy"


# --------------------------------------------------
# GENERATE OR LOAD PREDICTION SCORES
# --------------------------------------------------

def get_prediction_scores(model, image_path, filename):

    cache_path = get_cache_path(filename)

    if cache_path.exists():

        print("Using cached prediction scores.")

        return np.load(cache_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Unable to read image.")

    original_h, original_w = image.shape[:2]

    resized = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    batch = resized.reshape(
        1,
        *resized.shape
    )

    tensor = direct_val(batch)

    with torch.no_grad():

        _, logits = model(tensor)

        scores = torch.sigmoid(logits)[0, 0]

        scores = scores.cpu().numpy()

    scores = cv2.resize(
        scores,
        (original_w, original_h)
    )

    np.save(cache_path, scores)

    return scores


# --------------------------------------------------
# MAIN TEST EVALUATION
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=10
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nLoading dataset information...")

    split = pd.read_csv(SPLIT_CSV)
    pairs = pd.read_csv(PAIRS_CSV)

    split["key"] = split["filename"].apply(
        lambda x: Path(str(x)).name
    )

    pairs["key"] = pairs["image_path"].apply(
        lambda x: Path(str(x).replace("\\", "/")).name
    )

    # Select tampered images from TEST split
    test_tampered = split[
        (split["split"] == "test") &
        (split["class"] == 1)
    ]

    matched = test_tampered.merge(
        pairs,
        on="key",
        how="inner"
    )

    print("\n========== TEST DATASET INFORMATION ==========")

    print("Test tampered images:", len(test_tampered))
    print("Matched image-mask pairs:", len(matched))

    if args.limit > 0:
        matched = matched.head(args.limit)

    print("Images selected:", len(matched))

    if matched.empty:
        print("No matching test pairs found.")
        return

    model = load_model()

    results = []
    skipped = []

    for position, (_, row) in enumerate(
        matched.iterrows(),
        start=1
    ):

        filename = row["key"]

        print(
            f"\nProcessing {position}/{len(matched)}: "
            f"{filename}"
        )

        image_path = PROJECT_ROOT / row["image_path"]
        mask_path = PROJECT_ROOT / row["mask_path"]

        try:

            gt = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE
            )

            if gt is None:
                raise ValueError("Unable to read ground-truth mask.")

            scores = get_prediction_scores(
                model,
                image_path,
                filename
            )

            if scores.shape != gt.shape:
                raise ValueError(
                    "Prediction and ground-truth dimensions differ."
                )

            ground_truth = gt > 127
            pred = scores >= THRESHOLD

            iou, precision, recall, f1 = calculate_metrics(
                pred,
                ground_truth
            )

            results.append({
                "filename": filename,
                "Threshold": THRESHOLD,
                "IoU": iou,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            })

            print(
                f"IoU: {iou:.4f} | "
                f"Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | "
                f"F1: {f1:.4f}"
            )

        except Exception as error:

            skipped.append({
                "filename": filename,
                "error": str(error)
            })

            print("Skipped:", error)

        # Save progress after every image
        pd.DataFrame(results).to_csv(
            DETAILS_PATH,
            index=False
        )

        pd.DataFrame(skipped).to_csv(
            SKIPPED_PATH,
            index=False
        )

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    if not results:

        print("\nNo test images were successfully evaluated.")
        return

    results_df = pd.DataFrame(results)

    summary = pd.DataFrame([{
        "Split": "test",
        "Threshold": THRESHOLD,
        "Images_Evaluated": len(results_df),
        "Images_Skipped": len(skipped),
        "IoU": results_df["IoU"].mean(),
        "Precision": results_df["Precision"].mean(),
        "Recall": results_df["Recall"].mean(),
        "F1": results_df["F1"].mean()
    }])

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    print("\n========== FINAL TEST RESULTS ==========")

    print(summary.to_string(index=False))

    print("\n========== FILES SAVED ==========")

    print("Summary:", SUMMARY_PATH)
    print("Per-image results:", DETAILS_PATH)
    print("Skipped images:", SKIPPED_PATH)
    print("Prediction cache:", CACHE_DIR)

    print("\nFinal test evaluation completed!")


if __name__ == "__main__":
    main()
