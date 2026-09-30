import matplotlib.pyplot as plt
import torch

from casia_dataset import CASIADataset


CSV_FILE = "experiments/dataset_split.csv"

# ImageNet normalization values
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def denormalize(image):
    """
    Reverse ImageNet normalization for visualization.
    """
    image = image.clone()

    for channel in range(3):
        image[channel] = image[channel] * STD[channel] + MEAN[channel]

    return torch.clamp(image, 0, 1)


dataset = CASIADataset(CSV_FILE, "train")

# Take first 6 images
fig, axes = plt.subplots(2, 3, figsize=(10, 7))

for i, ax in enumerate(axes.flat):

    image, label = dataset[i]

    image = denormalize(image)

    # Convert from [C, H, W] to [H, W, C]
    image = image.permute(1, 2, 0)

    ax.imshow(image)
    ax.set_title(
        "Authentic" if label.item() == 0 else "Tampered"
    )
    ax.axis("off")


plt.tight_layout()

output_path = "results/preprocessing_samples.png"
plt.savefig(output_path, dpi=150)

print("=" * 50)
print("PREPROCESSING VISUALIZATION")
print("=" * 50)
print(f"Saved visualization to: {output_path}")