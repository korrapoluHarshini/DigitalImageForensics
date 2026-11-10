from torch.utils.data import DataLoader

from casia_dataset import CASIADataset


CSV_FILE = "experiments/dataset_split.csv"

BATCH_SIZE = 32


def get_dataloaders():

    train_dataset = CASIADataset(CSV_FILE, "train")
    validation_dataset = CASIADataset(CSV_FILE, "validation")
    test_dataset = CASIADataset(CSV_FILE, "test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, validation_loader, test_loader


if __name__ == "__main__":

    print("=" * 50)
    print("DATALOADER TEST")
    print("=" * 50)

    train_loader, validation_loader, test_loader = get_dataloaders()

    print(f"Training batches   : {len(train_loader)}")
    print(f"Validation batches : {len(validation_loader)}")
    print(f"Testing batches    : {len(test_loader)}")

    # Test one training batch
    images, labels = next(iter(train_loader))

    print("\nTraining batch:")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    print("\nDataLoaders working correctly!")