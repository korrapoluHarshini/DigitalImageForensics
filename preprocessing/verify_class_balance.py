from casia_dataset import CASIADataset


CSV_FILE = "experiments/dataset_split.csv"

splits = ["train", "validation", "test"]


print("=" * 50)
print("CLASS BALANCE VERIFICATION")
print("=" * 50)


for split in splits:

    dataset = CASIADataset(CSV_FILE, split)

    authentic_count = 0
    tampered_count = 0

    for i in range(len(dataset)):

        _, label = dataset[i]

        if label.item() == 0:
            authentic_count += 1
        else:
            tampered_count += 1

    print(f"\n{split.capitalize()} dataset:")
    print(f"Authentic : {authentic_count}")
    print(f"Tampered  : {tampered_count}")
    print(f"Total     : {authentic_count + tampered_count}")


print("\n" + "=" * 50)
print("CLASS BALANCE VERIFICATION COMPLETE")
print("=" * 50)