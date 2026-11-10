import sys
from pathlib import Path

import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Allow Python to find project modules
sys.path.append("preprocessing")
sys.path.append("models")

from dataloaders import get_dataloaders
from baseline_model import create_baseline_model


# ==================================================
# CONFIGURATION
# ==================================================

DEVICE = torch.device("cpu")

MODEL_PATH = Path("results/baseline_resnet18_best.pth")


print("=" * 60)
print("BASELINE MODEL EVALUATION")
print("=" * 60)

print(f"Device     : {DEVICE}")
print(f"Model path : {MODEL_PATH}")


# ==================================================
# LOAD DATA
# ==================================================

train_loader, validation_loader, test_loader = get_dataloaders()

print(f"\nTest batches: {len(test_loader)}")


# ==================================================
# LOAD MODEL
# ==================================================

model = create_baseline_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

print("Best baseline model loaded successfully.")


# ==================================================
# EVALUATION
# ==================================================

all_labels = []
all_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())


all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)


# ==================================================
# METRICS
# ==================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)


# ==================================================
# CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ==================================================
# RESULTS
# ==================================================

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-score  : {f1 * 100:.2f}%")


print("\nConfusion Matrix:")
print(cm)


print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["Authentic", "Tampered"],
        zero_division=0
    )
)


print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)