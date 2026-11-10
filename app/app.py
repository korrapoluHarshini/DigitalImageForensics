import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="Digital Image Authentication"
)

st.title("Digital Image Authentication")
st.write("Upload an image to check if it is authentic or tampered.")

st.divider()

# -----------------------------
# Device
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 2)

    model_path = "results/baseline_resnet18_best.pth"

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model


model = load_model()

# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Check Image"):

        # Preprocess image
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)

        # Model prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)

        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100

        # Class labels
        labels = {
            0: "Authentic",
            1: "Tampered"
        }

        result = labels[predicted_class]

        st.divider()

        st.subheader("Result")

        st.write(f"**Prediction:** {result}")
        st.write(f"**Confidence:** {confidence:.2f}%")