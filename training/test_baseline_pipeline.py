import sys

import torch


# Allow Python to find our project modules
sys.path.append("preprocessing")
sys.path.append("models")

from dataloaders import get_dataloaders
from baseline_model import create_baseline_model


print("=" * 50)
print("BASELINE MODEL PIPELINE TEST")
print("=" * 50)


# Use CPU
device = torch.device("cpu")

print(f"Device: {device}")


# Load DataLoader
train_loader, validation_loader, test_loader = get_dataloaders()

print(f"Training batches: {len(train_loader)}")


# Load model
model = create_baseline_model()

model = model.to(device)

# Evaluation mode for this test
model.eval()


# Get one batch
images, labels = next(iter(train_loader))

images = images.to(device)
labels = labels.to(device)


print(f"\nInput images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")


# Pass images through ResNet18
with torch.no_grad():
    outputs = model(images)


print(f"Model output shape: {outputs.shape}")

print("\nExpected output shape:")
print("(32, 2)")


print("\nPipeline test successful!")