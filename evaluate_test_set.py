
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
# PROJECT PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(
    r"C:\Users\harsh\Documents\DigitalImageForensics"
)

SPLIT_CSV = PROJECT_ROOT / "experiments" / "dataset_split.csv"
PAIRS_CSV = PROJECT_ROOT / "experiments" / "localization_pairs.csv"

CHECKPOINT = Path("ckpt/mvssnet_casia.pt")

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
IMAGE_SIZE = 512

OUTPUT_DIR = Path("results")
CACHE_DIR = OUTPUT_DIR / "validation_score_cache"

SUMMARY_PATH = OUTPUT_DIR / "threshold_comparison.csv"
DETAILS_PATH = OUTPUT_DIR / "threshold_comparison_per_image.csv"


# --------------------------------------------------
# LOAD PRETRAINED MVSS-NET
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
# CALCULATE PIXEL-LEVEL METRICS
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
# GET CACHE FILE NAME
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

    # Reuse previously generated scores if available
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

    # Cache scores so interrupted evaluations can resume
    np.save(cache_path, scores)

    return scores


# --------------------------------------------------
# SAVE CURRENT RESULTS
# --------------------------------------------------

def save_results(threshold_results):

    summary = []
    all_image_results = []

    for threshold in THRESHOLDS:

        records = threshold_results[threshold]

        if not records:
            continue

        results_df = pd.DataFrame(records)

        summary.append({
            "Threshold": threshold,
            "Images_Evaluated": len(records),
            "IoU": results_df["IoU"].mean(),
            "Precision": results_df["Precision"].mean(),
            "Recall": results_df["Recall"].mean(),
            "F1": results_df["F1"].mean()
        })

        for record in records:

            all_image_results.append({
                "Threshold": threshold,
                **record
            })

    if not summary:
        return

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False
    )

    pd.DataFrame(all_image_results).to_csv(
        DETAILS_PATH,
        index=False
    )


# --------------------------------------------------
# MAIN EVALUATION
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

    # Select tampered images from validation split
    validation_tampered = split[
        (split["split"] == "validation") &
        (split["class"] == 1)
    ]

    matched = validation_tampered.merge(
        pairs,
        on="key",
        how="inner"
    )

    print("\n========== DATASET INFORMATION ==========")

    print(
        "Validation tampered images:",
        len(validation_tampered)
    )

    print(
        "Matched image-mask pairs:",
        len(matched)
    )

    if args.limit > 0:

        matched = matched.head(args.limit)

    print(
        "Images selected for evaluation:",
        len(matched)
    )

    if matched.empty:

        print("No matching validation pairs found.")
        return

    model = load_model()

    threshold_results = {
        threshold: []
        for threshold in THRESHOLDS
    }

    successful = 0
    skipped = 0

    # Process each image once
    for position, (_, row) in enumerate(
        matched.iterrows(),
        start=1
    ):

        filename = row["key"]

        print(
            f"\nProcessing image {position}/{len(matched)}: "
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

            # Evaluate all thresholds using the same scores
            for threshold in THRESHOLDS:

                pred = scores >= threshold

                iou, precision, recall, f1 = calculate_metrics(
                    pred,
                    ground_truth
                )

                threshold_results[threshold].append({
                    "filename": filename,
                    "IoU": iou,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1
                })

            successful += 1

            print("Image evaluated successfully.")

        except Exception as error:

            skipped += 1

            print(
                f"Skipping {filename}: {error}"
            )

        # Save progress after every image
        save_results(threshold_results)

        print(
            f"Progress: {position}/{len(matched)} | "
            f"Successful: {successful} | "
            f"Skipped: {skipped}"
        )

    # --------------------------------------------------
    # FINAL THRESHOLD COMPARISON
    # --------------------------------------------------

    print("\n========== THRESHOLD COMPARISON ==========")

    summary_df = pd.read_csv(SUMMARY_PATH)

    print(
        summary_df.to_string(index=False)
    )

    best_row = summary_df.loc[
        summary_df["F1"].idxmax()
    ]

    print("\n========== BEST VALIDATION THRESHOLD ==========")

    print("Best threshold:", best_row["Threshold"])
    print("Mean IoU:", best_row["IoU"])
    print("Mean Precision:", best_row["Precision"])
    print("Mean Recall:", best_row["Recall"])
    print("Mean F1:", best_row["F1"])

    print("\n========== EVALUATION COMPLETED ==========")

    print("Successfully evaluated:", successful)
    print("Skipped:", skipped)

    print("Threshold summary:", SUMMARY_PATH)
    print("Per-image results:", DETAILS_PATH)
    print("Prediction cache:", CACHE_DIR)


if __name__ == "__main__":
    main()
