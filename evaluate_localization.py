
import cv2
import numpy as np
from pathlib import Path

# Paths to prediction and ground-truth mask
PREDICTION_PATH = Path("results/cpu_inference/predicted_mask.png")

GROUND_TRUTH_PATH = Path(
    r"C:\Users\harsh\Documents\DigitalImageForensics\datasets"
    r"\CASIA2.0_Groundtruth\Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png"
)


def evaluate_masks(prediction_path, ground_truth_path):

    prediction = cv2.imread(str(prediction_path), cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.imread(str(ground_truth_path), cv2.IMREAD_GRAYSCALE)

    if prediction is None:
        raise FileNotFoundError(f"Prediction not found: {prediction_path}")

    if ground_truth is None:
        raise FileNotFoundError(f"Ground truth not found: {ground_truth_path}")

    # Ensure both masks have the same dimensions
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Mask dimensions differ: prediction={prediction.shape}, "
            f"ground truth={ground_truth.shape}"
        )

    # Convert masks to binary values
    prediction = prediction > 127
    ground_truth = ground_truth > 127

    # Calculate pixel-level metrics
    true_positive = np.logical_and(prediction, ground_truth).sum()
    false_positive = np.logical_and(prediction, ~ground_truth).sum()
    false_negative = np.logical_and(~prediction, ground_truth).sum()

    iou = true_positive / (
        true_positive + false_positive + false_negative + 1e-8
    )

    precision = true_positive / (
        true_positive + false_positive + 1e-8
    )

    recall = true_positive / (
        true_positive + false_negative + 1e-8
    )

    f1_score = (2 * precision * recall) / (
        precision + recall + 1e-8
    )

    print("\nMVSS-Net Pixel-Level Evaluation")
    print("--------------------------------")
    print(f"IoU       : {iou:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1_score:.4f}")


if __name__ == "__main__":
    evaluate_masks(PREDICTION_PATH, GROUND_TRUTH_PATH)
