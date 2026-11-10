from casia_dataset import CASIADataset


CSV_FILE = "experiments/dataset_split.csv"


splits = ["train", "validation", "test"]


print("=" * 50)
print("DATASET SPLIT VERIFICATION")
print("=" * 50)


for split in splits:

    dataset = CASIADataset(CSV_FILE, split)

    print(f"\n{split.capitalize()} dataset:")
    print(f"Number of images: {len(dataset)}")

    # Test loading one image
    image, label = dataset[0]

    print(f"Sample image shape: {image.shape}")
    print(f"Sample label: {label.item()}")


print("\n" + "=" * 50)
print("SPLIT VERIFICATION COMPLETE")
print("=" * 50)