from pathlib import Path
from PIL import Image

# Path to the revised CASIA dataset
DATASET_PATH = Path("datasets/CASIA2.0_revised")

AUTHENTIC_PATH = DATASET_PATH / "Au"
TAMPERED_PATH = DATASET_PATH / "Tp"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_image_files(folder):
    return [
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def check_images(files):
    valid = 0
    invalid = 0

    for file in files:
        try:
            with Image.open(file) as image:
                image.verify()
            valid += 1
        except Exception:
            invalid += 1

    return valid, invalid


print("=" * 50)
print("CASIA 2.0 DATASET CHECK")
print("=" * 50)

# Check folders
if not AUTHENTIC_PATH.exists():
    print("ERROR: Au folder not found!")
    exit()

if not TAMPERED_PATH.exists():
    print("ERROR: Tp folder not found!")
    exit()

# Get images
authentic_images = get_image_files(AUTHENTIC_PATH)
tampered_images = get_image_files(TAMPERED_PATH)

print(f"\nAuthentic images found : {len(authentic_images)}")
print(f"Tampered images found  : {len(tampered_images)}")

# Check whether images can be opened
print("\nChecking authentic images...")
auth_valid, auth_invalid = check_images(authentic_images)

print("Checking tampered images...")
tam_valid, tam_invalid = check_images(tampered_images)

print("\n" + "=" * 50)
print("RESULT")
print("=" * 50)

print(f"Authentic - valid   : {auth_valid}")
print(f"Authentic - invalid : {auth_invalid}")

print(f"Tampered - valid    : {tam_valid}")
print(f"Tampered - invalid  : {tam_invalid}")

print("\nDataset check completed.")