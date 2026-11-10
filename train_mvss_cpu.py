import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from models.mvssnet import get_mvss
from casia_dataset import CASIADataset


# -------------------------------
# 1. CONFIGURATION
# -------------------------------

DEVICE = torch.device("cpu")

IMAGE_SIZE = 512
BATCH_SIZE = 1
EPOCHS = 1

LEARNING_RATE = 1e-5

CHECKPOINT_PATH = Path("ckpt/mvssnet_casia.pt")

RESULTS_DIR = Path("results/mvssnet_finetuned")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = RESULTS_DIR / "best_model.pth"

torch.set_num_threads(4)


# -------------------------------
# 2. LOAD DATASETS
# -------------------------------

train_dataset = CASIADataset(
    split="train",
    image_size=IMAGE_SIZE,
    augment=True
)

val_dataset = CASIADataset(
    split="validation",
    image_size=IMAGE_SIZE,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Device:", DEVICE)
# -------------------------------
# 3. LOAD MVSS-NET MODEL
# -------------------------------

model = get_mvss(
    backbone="resnet50",
    pretrained_base=False,
    nclass=1,
    sobel=True,
    constrain=True,
    n_input=3
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint, strict=True)

model = model.to(DEVICE)

print("\nPretrained MVSS-Net checkpoint loaded successfully!")


# -------------------------------
# 4. LOSS FUNCTIONS
# -------------------------------

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(logits, target, smooth=1.0):

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (probabilities * target).sum()

    dice = (
        2.0 * intersection + smooth
    ) / (
        probabilities.sum() + target.sum() + smooth
    )

    return 1.0 - dice


def classification_loss(mask_logits, labels):

    # Global maximum pooling over the raw segmentation logits
    image_logits = F.adaptive_max_pool2d(
        mask_logits, output_size=1
    ).flatten(1).squeeze(1)

    return bce_loss(image_logits, labels)
# -------------------------------
# 5. OPTIMIZER
# -------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.999)
)


# -------------------------------
# 6. TRAINING FUNCTION
# -------------------------------

def train_one_epoch():

    model.train()

    total_loss = 0.0

    for batch_index, (images, masks, edges, labels) in enumerate(
        tqdm(train_loader, desc="Training")
    ):

        # Temporary safety limit: test only one batch first
        if batch_index >= 1:
            break

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        edges = edges.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        edge_logits, mask_logits = model(images)

        loss_seg = dice_loss(mask_logits, masks)
        loss_edge = dice_loss(edge_logits, edges)
        loss_cls = classification_loss(mask_logits, labels)

        loss = (
            0.16 * loss_seg
            + 0.80 * loss_edge
            + 0.04 * loss_cls
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(
            f"\nBatch {batch_index + 1} | "
            f"Seg: {loss_seg.item():.4f} | "
            f"Edge: {loss_edge.item():.4f} | "
            f"Class: {loss_cls.item():.4f} | "
            f"Total: {loss.item():.4f}"
        )

    return total_loss


# -------------------------------
# 7. VALIDATION FUNCTION
# -------------------------------

def validate():

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch_index, (images, masks, edges, labels) in enumerate(
            tqdm(val_loader, desc="Validation")
        ):

            # Temporary safety limit: test only one batch first
            if batch_index >= 1:
                break

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            edges = edges.to(DEVICE)
            labels = labels.to(DEVICE)

            edge_logits, mask_logits = model(images)

            loss_seg = dice_loss(mask_logits, masks)
            loss_edge = dice_loss(edge_logits, edges)
            loss_cls = classification_loss(mask_logits, labels)

            loss = (
                0.16 * loss_seg
                + 0.80 * loss_edge
                + 0.04 * loss_cls
            )

            total_loss += loss.item()

            print(
                f"\nValidation loss: {loss.item():.4f}"
            )

    return total_loss
# -------------------------------
# 8. MAIN TRAINING EXECUTION
# -------------------------------

if __name__ == "__main__":

    print("\nStarting MVSS-Net CPU training test...")

    for epoch in range(EPOCHS):

        print(f"\n========== EPOCH {epoch + 1} ==========")

        train_loss = train_one_epoch()

        val_loss = validate()

        print("\nTraining loss:", train_loss)
        print("Validation loss:", val_loss)

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("\nModel checkpoint saved at:", BEST_MODEL_PATH)

    print("\nCPU training test completed!")