
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from models.mvssnet import get_mvss
from common.transforms import direct_val


PATCH_SIZE = 512
OVERLAP = 128
THRESHOLD = 0.4


def load_model(checkpoint_path, device):
    print("Loading MVSS-Net...")

    model = get_mvss(
        backbone="resnet50",
        pretrained_base=False,
        nclass=1,
        sobel=True,
        constrain=True,
        n_input=3
    )

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


def get_patch_positions(length):
    if length <= PATCH_SIZE:
        return [0]

    stride = PATCH_SIZE - OVERLAP

    positions = list(
        range(0, length - PATCH_SIZE + 1, stride)
    )

    last_position = length - PATCH_SIZE

    if positions[-1] != last_position:
        positions.append(last_position)

    return positions


def predict_patch(patch, model, device):
    resized = cv2.resize(
        patch,
        (PATCH_SIZE, PATCH_SIZE)
    )

    input_batch = resized.reshape(1, *resized.shape)
    input_tensor = direct_val(input_batch).to(device)

    with torch.no_grad():
        _, logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)[0, 0]

    return probabilities.cpu().numpy()


def predict_image(image_path, model, device, output_dir):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    height, width = image.shape[:2]

    accumulated = np.zeros(
        (height, width),
        dtype=np.float32
    )

    weights = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # Create Gaussian weight map
    center = PATCH_SIZE // 2
    sigma = PATCH_SIZE / 4

    gy, gx = np.ogrid[
        :PATCH_SIZE,
        :PATCH_SIZE
    ]

    gaussian_weights = np.exp(
        -(
            (gx - center) ** 2 +
            (gy - center) ** 2
        ) / (2 * sigma ** 2)
    ).astype(np.float32)

    gaussian_weights /= gaussian_weights.max()

    y_positions = get_patch_positions(height)
    x_positions = get_patch_positions(width)

    total_patches = (
        len(y_positions) * len(x_positions)
    )

    print("Image dimensions:", width, "x", height)
    print("Total patches:", total_patches)

    patch_number = 0

    # Process all patches
    for y in y_positions:

        for x in x_positions:

            patch_number += 1

            y_end = min(y + PATCH_SIZE, height)
            x_end = min(x + PATCH_SIZE, width)

            patch = image[y:y_end, x:x_end]

            # Pad edge patches to 512 x 512
            padded = np.zeros(
                (PATCH_SIZE, PATCH_SIZE, 3),
                dtype=np.uint8
            )

            padded[
                :patch.shape[0],
                :patch.shape[1]
            ] = patch

            print(
                f"Processing patch "
                f"{patch_number}/{total_patches}"
            )

            prediction = predict_patch(
                padded,
                model,
                device
            )

            # Remove padded region
            prediction = prediction[
                :patch.shape[0],
                :patch.shape[1]
            ]

            # Get corresponding Gaussian weights
            patch_weights = gaussian_weights[
                :patch.shape[0],
                :patch.shape[1]
            ]

            # Accumulate weighted predictions
            accumulated[y:y_end, x:x_end] += (
                prediction * patch_weights
            )

            # IMPORTANT: Update weights for every patch
            weights[y:y_end, x:x_end] += patch_weights

    # Normalize accumulated predictions
    probability_map = accumulated / np.maximum(
        weights,
        1e-8
    )

    # Validate probability map
    print("\nProbability map validation:")

    print(
        "Minimum probability:",
        float(probability_map.min())
    )

    print(
        "Maximum probability:",
        float(probability_map.max())
    )

    print(
        "Mean probability:",
        float(probability_map.mean())
    )

    if (
        not np.isfinite(probability_map).all()
        or probability_map.min() < -1e-6
        or probability_map.max() > 1.000001
    ):
        raise ValueError(
            "Invalid probability map. "
            "Check patch accumulation and normalization."
        )

    # Generate binary mask
    binary_mask = (
        probability_map >= THRESHOLD
    ).astype(np.uint8) * 255

    grayscale_mask = np.clip(
        probability_map * 255,
        0,
        255
    ).astype(np.uint8)

    # Create highlighted overlay
    overlay = image.copy()

    suspicious = binary_mask > 0

    red = np.zeros_like(image)
    red[:, :, 2] = 255

    if np.any(suspicious):

        overlay[suspicious] = cv2.addWeighted(
            image[suspicious],
            0.5,
            red[suspicious],
            0.5,
            0
        )

    # Save outputs
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_dir / "patch_probability_map.npy",
        probability_map
    )

    cv2.imwrite(
        str(output_dir / "patch_probability_map.png"),
        grayscale_mask
    )

    cv2.imwrite(
        str(output_dir / "patch_predicted_mask.png"),
        binary_mask
    )

    cv2.imwrite(
        str(output_dir / "patch_highlighted_overlay.png"),
        overlay
    )

    print("\nPatch-based prediction completed!")

    print("Output folder:", output_dir)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        default="ckpt/mvssnet_casia.pt"
    )

    parser.add_argument(
        "--output",
        default="results/patch_inference"
    )

    args = parser.parse_args()

    device = torch.device("cpu")

    model = load_model(
        args.checkpoint,
        device
    )

    predict_image(
        Path(args.image),
        model,
        device,
        Path(args.output)
    )


if __name__ == "__main__":
    main()
