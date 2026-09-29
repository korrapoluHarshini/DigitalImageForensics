from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Dataset location
DATASET_PATH = Path("datasets/CASIA2.0_revised")

AUTHENTIC_PATH = DATASET_PATH / "Au"
TAMPERED_PATH = DATASET_PATH / "Tp"

# Reproducibility
RANDOM_SEED = 42


def get_images(folder):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    return [
        file.name
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in extensions
    ]


# Get filenames
authentic_images = get_images(AUTHENTIC_PATH)
tampered_images = get_images(TAMPERED_PATH)

print(f"Authentic images: {len(authentic_images)}")
print(f"Tampered images : {len(tampered_images)}")

# Create dataframe
data = []

for filename in authentic_images:
    data.append({
        "filename": filename,
        "class": 0,
        "class_name": "Authentic"
    })

for filename in tampered_images:
    data.append({
        "filename": filename,
        "class": 1,
        "class_name": "Tampered"
    })

df = pd.DataFrame(data)

# First split: 70% train, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=df["class"]
)

# Second split: half validation, half test
# 15% validation + 15% test
validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_df["class"]
)

# Add split information
train_df = train_df.copy()
validation_df = validation_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
validation_df["split"] = "validation"
test_df["split"] = "test"

# Combine
final_df = pd.concat(
    [train_df, validation_df, test_df],
    ignore_index=True
)

# Shuffle final dataframe
final_df = final_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

# Create output directory
output_dir = Path("experiments")
output_dir.mkdir(exist_ok=True)

# Save CSV
output_file = output_dir / "dataset_split.csv"
final_df.to_csv(output_file, index=False)

# Display results
print("\n" + "=" * 50)
print("DATASET SPLIT")
print("=" * 50)

print(f"Total      : {len(final_df)}")
print(f"Training   : {len(train_df)}")
print(f"Validation : {len(validation_df)}")
print(f"Testing    : {len(test_df)}")

print("\nClass distribution:")
print(pd.crosstab(final_df["split"], final_df["class_name"]))

print(f"\nSaved to: {output_file}")