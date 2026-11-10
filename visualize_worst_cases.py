
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import hashlib
import numpy as np
from pathlib import Path

ROOT = Path(r"C:\Users\harsh\Documents\DigitalImageForensics")

RESULTS = Path("results")
CACHE_DIR = RESULTS / "validation_score_cache"

PAIRS = pd.read_csv(
    ROOT / "experiments" / "localization_pairs.csv"
)

COMPARISON = pd.read_csv(
    RESULTS / "validation_method_comparison.csv"
)

# Extract filenames from image_path
PAIRS["filename"] = PAIRS["image_path"].apply(
    lambda x: Path(str(x).replace("\\", "/")).name
)

worst = COMPARISON.nsmallest(5, "Original_IoU")

for _, row in worst.iterrows():

    filename = row["filename"]

    pair = PAIRS[PAIRS["filename"] == filename]

    if pair.empty:
        print("Image-mask pair not found:", filename)
        continue

    image_path = ROOT / pair.iloc[0]["image_path"]
    mask_path = ROOT / pair.iloc[0]["mask_path"]

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("Could not load image or mask:", filename)
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    file_hash = hashlib.sha1(
        filename.encode("utf-8")
    ).hexdigest()

    probability = np.load(CACHE_DIR / f"{file_hash}.npy")
    prediction = (probability >= 0.4).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")

    axes[2].imshow(prediction, cmap="gray")
    axes[2].set_title("MVSS-Net Prediction")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(filename)
    plt.tight_layout()

    output = (
        RESULTS / "debug_cases" /
        f"{Path(filename).stem}_comparison.png"
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output, dpi=150)
    plt.close()

    print("Saved:", output)

print("\nVisualization completed.")
