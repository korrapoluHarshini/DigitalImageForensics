
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from models.mvssnet import get_mvss
from common.transforms import direct_val


# Input resolutions for multi-scale inference
SCALES = [512, 768, 1024]

# Keep the same threshold for comparison
THRESHOLD = 0.4


def load_model(checkpoint_path, device):

    print("Loading MVSS-Net architecture...")

    model = get_mvss(
        backbone="resnet50",
        pretrained_base=False,
        nclass=1,
        sobel=True,
        constrain=True,
        n_input=3
    )

    print("Loading pretrained checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint, strict=True)

    model.to(device)
    model.eval()

    print("MVSS-Net loaded successfully!")

    return model


def predict_at_scale(image, model, device, scale):

    resized = cv2.resize(
        image,
        (scale, scale)
    )

    input_batch = resized.reshape(
        1, *resized.shape
    )

    input_tensor = direct_val(input_batch).to(device)

    with torch.no_grad():

        _, logits = model(input_tensor)

        probabilities = torch.sigmoid(logits)[0, 0]

        probability_map = probabilities.cpu().numpy()

    return probability_map


def predict_multiscale(image_path, model, device, output_dir):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Cannot read image: {image_path}"
        )

    height, width = image.shape[:2]

    # Create output directory
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Initialize combined probability map
    combined_probability = np.zeros(
        (height, width),
        dtype=np.float32
    )

    print("\nStarting multi-scale inference...")

    # Process each scale independently
    for scale in SCALES:

        print(f"\nProcessing scale: {scale} x {scale}")

        probability = predict_at_scale(
            image,
            model,
            device,
            scale
        )

        # Resize prediction to original image dimensions
        probability = cv2.resize(
            probability,
            (width, height)
        )

        probability = probability.astype(np.float32)

        # Save individual scale probability map
        np.save(
            output_dir / f"probability_scale_{scale}.npy",
            probability
        )

        probability_png = (
            probability * 255
        ).clip(0, 255).astype(np.uint8)

        cv2.imwrite(
            str(output_dir / f"probability_scale_{scale}.png"),
            probability_png
        )

        # Add prediction to combined map
        combined_probability += probability

        print(
            f"Scale {scale}: "
            f"Max={probability.max():.4f}, "
            f"Mean={probability.mean():.4f}"
        )

    # Average predictions from all scales
    combined_probability /= len(SCALES)

    combined_probability = np.clip(
        combined_probability,
        0.0,
        1.0
    )

    # Generate binary prediction mask
    binary_mask = (
        combined_probability >= THRESHOLD
    ).astype(np.uint8) * 255

    # Generate grayscale probability map
    grayscale_mask = (
        combined_probability * 255
    ).astype(np.uint8)

    # Generate red overlay
    overlay = image.copy()

    suspicious = binary_mask > 0

    red = np.zeros_like(image)
    red[:, :, 2] = 255

    overlay[suspicious] = cv2.addWeighted(
        image[suspicious],
        0.5,
        red[suspicious],
        0.5,
        0
    )

    # Save combined probability map
    np.save(
        output_dir / "multiscale_probability_map.npy",
        combined_probability
    )

    cv2.imwrite(
        str(output_dir / "multiscale_probability_map.png"),
        grayscale_mask
    )

    cv2.imwrite(
        str(output_dir / "multiscale_predicted_mask.png"),
        binary_mask
    )

    cv2.imwrite(
        str(output_dir / "multiscale_overlay.png"),
        overlay
    )

    print("\nMULTI-SCALE INFERENCE COMPLETED")

    print("Image dimensions:", width, "x", height)

    print(
        "Combined maximum:",
        float(combined_probability.max())
    )

    print(
        "Combined mean:",
        float(combined_probability.mean())
    )

    print("Threshold:", THRESHOLD)

    print("Results saved to:", output_dir)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image"
    )

    parser.add_argument(
        "--checkpoint",
        default="ckpt/mvssnet_casia.pt"
    )

    parser.add_argument(
        "--output",
        default="results/multiscale_inference"
    )

    args = parser.parse_args()

    device = torch.device("cpu")

    model = load_model(
        args.checkpoint,
        device
    )

    predict_multiscale(
        Path(args.image),
        model,
        device,
        Path(args.output)
    )


if __name__ == "__main__":
    main()
