
import cv2
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset
import torch.nn.functional as F


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(
    r"C:\Users\harsh\Documents\DigitalImageForensics"
)

SPLIT_CSV = PROJECT_ROOT / "experiments" / "dataset_split.csv"

PAIRS_CSV = PROJECT_ROOT / "experiments" / "localization_pairs.csv"

DATASET_DIR = PROJECT_ROOT / "datasets" / "CASIA2.0_revised"


# --------------------------------------------------
# EDGE GENERATOR
# Based on the GitHub MVSS-Net reproduction
# --------------------------------------------------

class EdgeGenerator(torch.nn.Module):

    def __init__(self, kernel_size=15):
        super().__init__()

        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size

    def forward(self, mask):

        mask = mask.clone().float()

        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)

        assert mask.shape[1] == 1

        k = self.kernel_size

        kernel = torch.zeros((1, 1, k, k))

        # Cross-shaped kernel
        kernel[0, 0, k // 2, :] = 1
        kernel[0, 0, :, k // 2] = 1

        # Dilation of original mask
        dilated = F.conv2d(
            mask,
            kernel,
            stride=1,
            padding=k // 2
        ) > 0

        # Dilation of inverse mask
        eroded_inverse = F.conv2d(
            1 - mask,
            kernel,
            stride=1,
            padding=k // 2
        ) > 0

        # Generate binary edge map
        edge = 1 - torch.abs(
            eroded_inverse.float() - dilated.float()
        )

        return (edge > 0).float()


# --------------------------------------------------
# CASIA DATASET
# --------------------------------------------------

class CASIADataset(Dataset):

    def __init__(
        self,
        split="train",
        image_size=512,
        augment=False
    ):

        self.image_size = image_size
        self.split = split
        self.augment = augment and split == "train"

        self.edge_generator = EdgeGenerator(
            kernel_size=15
        )

        # Read existing CSV files
        split_data = pd.read_csv(SPLIT_CSV)
        pair_data = pd.read_csv(PAIRS_CSV)

        # Map tampered filenames to their ground-truth masks
        self.mask_mapping = {
            Path(row.image_path).name: row.mask_path
            for row in pair_data.itertuples()
        }

        # Select the requested split
        split_data = split_data[
            split_data["split"] == split
        ]

        self.samples = []

        for row in split_data.itertuples():

            filename = row.filename
            class_name = row.class_name

            # --------------------------------------
            # Authentic image
            # --------------------------------------

            if class_name == "Authentic":

                label = 0

                image_path = (
                    DATASET_DIR / "Au" / filename
                )

                mask_path = None

            # --------------------------------------
            # Tampered image
            # --------------------------------------

            elif class_name == "Tampered":

                label = 1

                image_path = (
                    DATASET_DIR / "Tp" / filename
                )

                # Exclude images without valid masks
                if filename not in self.mask_mapping:
                    continue

                mask_path = (
                    PROJECT_ROOT / self.mask_mapping[filename]
                )

            else:
                raise ValueError(
                    f"Unknown class: {class_name}"
                )

            # Verify image path
            if not image_path.exists():
                raise FileNotFoundError(image_path)

            # Verify mask path
            if mask_path is not None:
                if not mask_path.exists():
                    raise FileNotFoundError(mask_path)

            self.samples.append(
                (image_path, mask_path, label)
            )

        authentic_count = sum(
            1 for _, _, label in self.samples if label == 0
        )

        tampered_count = sum(
            1 for _, _, label in self.samples if label == 1
        )

        print(f"\n{split.upper()} DATASET")
        print("---------------------------")
        print("Total images:", len(self.samples))
        print("Authentic:", authentic_count)
        print("Tampered:", tampered_count)


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):

        image_path, mask_path, label = self.samples[index]

        # ------------------------------------------
        # Read RGB image
        # ------------------------------------------

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Cannot read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ------------------------------------------
        # Read ground-truth manipulation mask
        # ------------------------------------------

        if mask_path is None:

            # Authentic images receive black masks
            mask = np.zeros(
                image.shape[:2],
                dtype=np.uint8
            )

        else:

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE
            )

            if mask is None:
                raise RuntimeError(
                    f"Cannot read mask: {mask_path}"
                )

        # ------------------------------------------
        # Resize image and mask
        # ------------------------------------------

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        # Convert mask into binary values: 0 or 1
        mask = (mask > 127).astype(np.float32)

        # ------------------------------------------
        # Data augmentation
        # ------------------------------------------

        if self.augment:

            # Horizontal flip
            if np.random.rand() < 0.5:

                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

            # Vertical flip
            if np.random.rand() < 0.5:

                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

            # Rotation: same angle for image and mask
            if np.random.rand() < 0.5:

                k = np.random.randint(1, 4)

                image = np.rot90(
                    image,
                    k=k
                ).copy()

                mask = np.rot90(
                    mask,
                    k=k
                ).copy()

        # ------------------------------------------
        # Convert image to tensor
        # ------------------------------------------

        image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(
            image.transpose(2, 0, 1).copy()
        )

        # ImageNet normalization
        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        image = (image - mean) / std

        # ------------------------------------------
        # Convert manipulation mask to tensor
        # ------------------------------------------

        mask = torch.from_numpy(
            mask.copy()
        )

        mask = mask.unsqueeze(0)

        # ------------------------------------------
        # Generate edge map
        # ------------------------------------------

        edge = self.edge_generator(mask)

        # Resize edge map to H/4 and W/4
        edge = F.interpolate(
            edge,
            size=(
                self.image_size // 4,
                self.image_size // 4
            ),
            mode="bilinear",
            align_corners=False
        )

        edge = edge.squeeze(0)

        # ------------------------------------------
        # Classification label
        # ------------------------------------------

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return image, mask, edge, label
