
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(
    r"C:\Users\harsh\Documents\DigitalImageForensics"
)

PAIRS_CSV = PROJECT_ROOT / "experiments" / "localization_pairs.csv"

TEST_RESULTS = Path(
    "results/final_test/final_test_per_image.csv"
)

OUTPUT_DIR = Path("results/final_test/alignment_check")

REPORT_PATH = OUTPUT_DIR / "alignment_report.csv"

# Three images we previously investigated
SAMPLE_IMAGES = [
    "Tp_S_NRN_S_N_ind20019_ind20019_01760.tif",
    "Tp_S_NRD_S_B_art10106_art10106_11589.jpg",
    "Tp_S_NRD_S_N_cha10154_cha10154_12266.jpg"
]


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(PAIRS_CSV)
    test_results = pd.read_csv(TEST_RESULTS)

    pairs["key"] = pairs["image_path"].apply(
        lambda x: Path(str(x).replace("\\", "/")).name
    )

    test_names = set(test_results["filename"])

    # Check only the evaluated test images
    pairs = pairs[pairs["key"].isin(test_names)]

    records = []

    print("\nChecking image-mask alignment...")

    for position, (_, row) in enumerate(
        pairs.iterrows(), start=1
    ):

        filename = row["key"]

        image_path = PROJECT_ROOT / row["image_path"]
        mask_path = PROJECT_ROOT / row["mask_path"]

        image = cv2.imread(str(image_path))
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        image_exists = image is not None
        mask_exists = mask is not None

        dimensions_match = False
        mask_pixels = 0
        mask_unique_values = ""

        if image_exists and mask_exists:

            dimensions_match = (
                image.shape[:2] == mask.shape[:2]
            )

            mask_pixels = int((mask > 127).sum())

            mask_unique_values = str(
                np.unique(mask).tolist()[:20]
            )

        records.append({
            "filename": filename,
            "Image_Readable": image_exists,
            "Mask_Readable": mask_exists,
            "Dimensions_Match": dimensions_match,
            "Image_Height": image.shape[0] if image_exists else None,
            "Image_Width": image.shape[1] if image_exists else None,
            "Mask_Height": mask.shape[0] if mask_exists else None,
            "Mask_Width": mask.shape[1] if mask_exists else None,
            "Manipulated_Pixels": mask_pixels,
            "Mask_Unique_Values": mask_unique_values
        })

        if filename in SAMPLE_IMAGES and image_exists and mask_exists:

            # Red overlay showing ground-truth region
            overlay = image.copy()

            suspicious = mask > 127

            red = np.zeros_like(image)
            red[:, :, 2] = 255

            overlay[suspicious] = cv2.addWeighted(
                image[suspicious],
                0.5,
                red[suspicious],
                0.5,
                0
            )

            output_path = OUTPUT_DIR / (
                Path(filename).stem + "_alignment.png"
            )

            cv2.imwrite(str(output_path), overlay)

            print("Saved sample overlay:", output_path)

        if position % 100 == 0:
            print(f"Checked {position} pairs...")

    df = pd.DataFrame(records)

    df.to_csv(REPORT_PATH, index=False)

    print("\n========== ALIGNMENT REPORT ==========")

    print("Pairs checked:", len(df))

    print(
        "Unreadable images:",
        int((~df["Image_Readable"]).sum())
    )

    print(
        "Unreadable masks:",
        int((~df["Mask_Readable"]).sum())
    )

    print(
        "Dimension mismatches:",
        int((~df["Dimensions_Match"]).sum())
    )

    print(
        "Empty ground-truth masks:",
        int((df["Manipulated_Pixels"] == 0).sum())
    )

    print("\nReport saved:", REPORT_PATH)
    print("Alignment verification completed!")


if __name__ == "__main__":
    main()
