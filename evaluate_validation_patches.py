
import pandas as pd
from pathlib import Path
import torch
import traceback
from contextlib import redirect_stdout

from cpu_patch_inference import load_model, predict_image


# Dataset location
DATA_ROOT = Path(
    r"C:\Users\harsh\Documents\DigitalImageForensics"
)

SPLIT_CSV = DATA_ROOT / "experiments" / "dataset_split.csv"
PAIRS_CSV = DATA_ROOT / "experiments" / "localization_pairs.csv"

# MVSS-Net project output location
OUTPUT_ROOT = Path("results/validation_patch_gaussian")
CHECKPOINT = Path("ckpt/mvssnet_casia.pt")


def main():

    print("Loading dataset information...")

    split_df = pd.read_csv(SPLIT_CSV)
    pairs_df = pd.read_csv(PAIRS_CSV)

    # Select validation tampered images
    validation = split_df[
        (split_df["split"] == "validation") &
        (split_df["class_name"] == "Tampered")
    ].copy()

    # Match image filenames with ground-truth masks
    validation["filename"] = validation["filename"].apply(
        lambda x: Path(str(x)).name
    )

    pairs_df["filename"] = pairs_df["image_path"].apply(
        lambda x: Path(str(x)).name
    )

    matched = validation.merge(
        pairs_df,
        on="filename",
        how="inner"
    )

    print("Matched validation pairs:", len(matched))

    if len(matched) != 767:
        raise ValueError(
            f"Expected 767 pairs, found {len(matched)}"
        )

    # Create a separate output folder
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Verify all image and mask paths before inference
    for _, row in matched.iterrows():

        image_path = DATA_ROOT / row["image_path"]
        mask_path = DATA_ROOT / row["mask_path"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image missing: {image_path}"
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask missing: {mask_path}"
            )

    print("All image and mask files verified.")

    # Load model only once
    device = torch.device("cpu")
    model = load_model(CHECKPOINT, device)

    results = []

    # Process all 767 images
    for index, row in matched.iterrows():

        filename = row["filename"]

        image_path = DATA_ROOT / row["image_path"]
        mask_path = DATA_ROOT / row["mask_path"]

        image_output = OUTPUT_ROOT / Path(filename).stem
        log_path = OUTPUT_ROOT / f"{Path(filename).stem}_log.txt"

        print(
            f"\n[{len(results) + 1}/{len(matched)}] "
            f"Processing: {filename}",
            flush=True
        )

        try:
            # Save detailed inference output separately
            with open(log_path, "w", encoding="utf-8") as log_file:
                with redirect_stdout(log_file):
                    predict_image(
                        image_path,
                        model,
                        device,
                        image_output
                    )

            results.append({
                "filename": filename,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "output_folder": str(image_output),
                "status": "success"
            })

        except Exception as error:

            print("ERROR:", error)

            results.append({
                "filename": filename,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "output_folder": str(image_output),
                "status": f"failed: {error}"
            })

        # Save progress after every image
        pd.DataFrame(results).to_csv(
            OUTPUT_ROOT / "inference_progress.csv",
            index=False
        )

    print("\n========== INFERENCE SUMMARY ==========")

    successful = sum(
        item["status"] == "success"
        for item in results
    )

    failed = len(results) - successful

    print("Total images:", len(results))
    print("Successful:", successful)
    print("Failed:", failed)

    print("Results saved in:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
