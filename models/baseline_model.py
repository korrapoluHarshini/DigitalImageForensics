import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_baseline_model():

    # Load ImageNet-pretrained ResNet18
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Replace the final classification layer
    num_features = model.fc.in_features

    model.fc = nn.Linear(num_features, 2)

    return model


if __name__ == "__main__":

    print("=" * 50)
    print("BASELINE MODEL TEST")
    print("=" * 50)

    model = create_baseline_model()

    print(model)

    print("\nFinal classification layer:")
    print(model.fc)

    print("\nNumber of output classes: 2")
    print("0 = Authentic")
    print("1 = Tampered")