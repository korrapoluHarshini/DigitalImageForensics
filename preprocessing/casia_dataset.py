from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from image_preprocessing import image_transform


class CASIADataset(Dataset):

    def __init__(self, csv_file, split):
        self.data = pd.read_csv(csv_file)

        # Select only the requested split
        self.data = self.data[self.data["split"] == split].reset_index(drop=True)

        self.dataset_path = Path("datasets/CASIA2.0_revised")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        filename = row["filename"]
        label = int(row["class"])

        # Determine the correct folder
        if label == 0:
            image_path = self.dataset_path / "Au" / filename
        else:
            image_path = self.dataset_path / "Tp" / filename

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply preprocessing
        image = image_transform(image)

        return image, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":

    print("=" * 50)
    print("CASIA DATASET LOADER TEST")
    print("=" * 50)

    csv_file = "experiments/dataset_split.csv"

    # Create training dataset
    train_dataset = CASIADataset(csv_file, "train")

    print(f"Training images: {len(train_dataset)}")

    # Load one sample
    image, label = train_dataset[0]

    print(f"Image shape: {image.shape}")
    print(f"Image data type: {image.dtype}")
    print(f"Label: {label.item()}")

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    # Load one batch
    images, labels = next(iter(train_loader))

    print("\nFirst batch:")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")