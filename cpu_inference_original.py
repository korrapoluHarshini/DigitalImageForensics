
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from models.mvssnet import get_mvss
from common.transforms import direct_val


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


def predict_image(image_path, model, device, output_dir):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    original_height, original_width = image.shape[:2]

    # Resize image to the model's expected input size
    resized_image = cv2.resize(image, (512, 512))

    # Apply the repository's original preprocessing
    input_batch = resized_image.reshape(1, *resized_image.shape)
    input_tensor = direct_val(input_batch)
    input_tensor = input_tensor.to(device)

    print("Generating tampering prediction...")

    with torch.no_grad():
        _, logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)[0, 0]
        probability_map = probabilities.cpu().numpy()

    # Resize prediction to original image dimensions
    probability_map = cv2.resize(
        probability_map,
        (original_width, original_height)
    )

    # Convert probability map into an 8-bit grayscale image
    grayscale_mask = np.clip(
        probability_map * 255, 0, 255
    ).astype(np.uint8)

    # Exploratory threshold; requires validation later
    binary_mask = (probability_map >= 0.5).astype(np.uint8) * 255

    # Create red overlay for predicted suspicious regions
    overlay = image.copy()
    suspicious = binary_mask > 0

    red = np.zeros_like(image)
    red[:, :, 2] = 255

    overlay[suspicious] = cv2.addWeighted(
        image[suspicious], 0.5,
        red[suspicious], 0.5,
        0
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_dir / "probability_map.png"), grayscale_mask)
    cv2.imwrite(str(output_dir / "predicted_mask.png"), binary_mask)
    cv2.imwrite(str(output_dir / "highlighted_overlay.png"), overlay)

    print("\nPrediction completed!")
    print("Maximum pixel score:", float(probability_map.max()))
    print("Mean pixel score:", float(probability_map.mean()))
    print("Saved results to:", output_dir)


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
        default="results/cpu_inference"
    )

    args = parser.parse_args()

    device = torch.device("cpu")

    model = load_model(args.checkpoint, device)

    predict_image(
        Path(args.image),
        model,
        device,
        Path(args.output)
    )


if __name__ == "__main__":
    main()
