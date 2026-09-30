from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms


# Image size required by the model
IMAGE_SIZE = 224


# Image preprocessing pipeline
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image_path):
    """
    Load an image and apply preprocessing.
    """

    image = Image.open(image_path).convert("RGB")

    processed_image = image_transform(image)

    return processed_image


if __name__ == "__main__":

    print("=" * 50)
    print("IMAGE PREPROCESSING TEST")
    print("=" * 50)

    dataset_path = Path("datasets/CASIA2.0_revised")

    authentic_path = dataset_path / "Au"

    # Get the first image from the authentic folder
    image_files = list(authentic_path.glob("*"))

    if not image_files:
        print("No images found.")
    else:
        sample_image = image_files[0]

        print(f"Sample image: {sample_image.name}")

        processed = preprocess_image(sample_image)

        print(f"Processed shape: {processed.shape}")
        print(f"Data type: {processed.dtype}")
        print(f"Minimum value: {processed.min():.4f}")
        print(f"Maximum value: {processed.max():.4f}")