import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Allow Python to find project modules
sys.path.append("preprocessing")
sys.path.append("models")

from dataloaders import get_dataloaders
from baseline_model import create_baseline_model


# ==================================================
# CONFIGURATION
# ==================================================

DEVICE = torch.device("cpu")

NUM_EPOCHS = 5
LEARNING_RATE = 0.0001

MODEL_SAVE_PATH = Path("results/baseline_resnet18_best.pth")


# ==================================================
# SETUP
# ==================================================

print("=" * 60)
print("BASELINE MODEL TRAINING")
print("=" * 60)

print(f"Device        : {DEVICE}")
print(f"Epochs        : {NUM_EPOCHS}")
print(f"Learning rate : {LEARNING_RATE}")


MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)


# ==================================================
# DATA
# ==================================================

train_loader, validation_loader, test_loader = get_dataloaders()

print(f"\nTraining batches   : {len(train_loader)}")
print(f"Validation batches : {len(validation_loader)}")
print(f"Testing batches    : {len(test_loader)}")


# ==================================================
# MODEL
# ==================================================

model = create_baseline_model()
model = model.to(DEVICE)


# ==================================================
# LOSS AND OPTIMIZER
# ==================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==================================================
# TRACK BEST MODEL
# ==================================================

best_validation_accuracy = 0.0


# ==================================================
# TRAINING
# ==================================================

for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 60)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("=" * 60)

    # ----------------------------------------------
    # Training
    # ----------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    train_progress = tqdm(
        train_loader,
        desc="Training"
    )

    for images, labels in train_progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        current_accuracy = 100 * correct / total

        train_progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{current_accuracy:.2f}%"
        )

    train_loss = running_loss / total
    train_accuracy = 100 * correct / total


    # ----------------------------------------------
    # Validation
    # ----------------------------------------------

    model.eval()

    validation_loss = 0.0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            validation_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            validation_total += labels.size(0)

            validation_correct += (
                predicted == labels
            ).sum().item()

    validation_loss = validation_loss / validation_total

    validation_accuracy = (
        100 * validation_correct / validation_total
    )


    # ----------------------------------------------
    # Epoch Results
    # ----------------------------------------------

    print("\nEpoch results:")

    print(f"Training loss      : {train_loss:.4f}")
    print(f"Training accuracy  : {train_accuracy:.2f}%")

    print(f"Validation loss    : {validation_loss:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.2f}%")


    # ----------------------------------------------
    # Save Best Model
    # ----------------------------------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )

        print(
            f"\nBest model saved!"
            f" Validation accuracy: "
            f"{validation_accuracy:.2f}%"
        )


print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_validation_accuracy:.2f}%"
)

print(f"Best model saved to: {MODEL_SAVE_PATH}")