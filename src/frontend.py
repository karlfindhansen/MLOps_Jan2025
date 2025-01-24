import os

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
import timm
import torch
from google.cloud import run_v2
from PIL import Image
from explainability import compute_gradcam, overlay_gradcam_on_image


@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/keen-defender-448412-p6/locations/europe-west10"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "backend":
            return service.uri
    name = os.environ.get("BACKEND", None)
    return name


def preprocess_image(image_path, input_size, mean, std):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(input_size, Image.BILINEAR)
    image = np.array(image).astype(np.float32) / 255.0

    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    image = (image - mean) / std
    image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
    return image.unsqueeze(0)


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/predict/"

    response = requests.post(predict_url, files=image, timeout=20)

    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    num_classes = 5

    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Bird Image Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = uploaded_file.read()

        st.image(img, caption="Uploaded Bird Image.", use_container_width=True)

        # Read the file content
        file_content = uploaded_file.getvalue()

        # Create form data dictionary
        form_data = {"image": ("image.jpg", file_content)}

        result = classify_image(form_data, backend=backend)

        if result is not None:
            prediction = result["top5_classes"]
            probabilities = result["top5_probabilities"]

            class_names = {
                1: "Black footed Albatross",
                2: "Laysan Albatross",
                3: "Sooty Albatross",
                4: "Groove billed Ani",
                5: "Crested Auklet",
                6: "Least Auklet",
                7: "Parakeet Auklet",
                8: "Rhinoceros Auklet",
                9: "Brewer Blackbird",
                10: "Red winged Blackbird",
                11: "Rusty Blackbird",
                12: "Yellow headed Blackbird",
                13: "Bobolink",
                14: "Indigo Bunting",
                15: "Lazuli Bunting",
                16: "Painted Bunting",
                17: "Cardinal",
                18: "Spotted Catbird",
                19: "Gray Catbird",
            }

            st.write("Class Prediction:", class_names[prediction[0] - 1])
            st.write("Probability:", probabilities[0])

            top5_classes = prediction
            top5_probabilities = probabilities

            st.header("Top 5 Predictions:")
            for i in range(5):
                st.write(f"{class_names[top5_classes[i]]}: {top5_probabilities[i] * 100:.2f}%")

            # Create a bar plot
            fig, ax = plt.subplots()
            ax.bar(range(5), [top5_probabilities[i] * 100 for i in range(5)], color="skyblue")
            ax.set_xticks(range(5))
            ax.set_xticklabels([class_names[top5_classes[i]] for i in range(5)], rotation=90)
            ax.set_ylabel("Probability (%)")
            ax.set_title("Top 5 Predictions")

            st.pyplot(fig)

            model = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)

            image = Image.open(uploaded_file)

            input_size = (224, 224)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            input_tensor = preprocess_image(uploaded_file, input_size, mean, std)

            # Compute Grad-CAM for the top predicted class
            gradcam = compute_gradcam(model, input_tensor, prediction[0])
            gradcam_overlay = overlay_gradcam_on_image(image, gradcam)

            st.header("Saliency Map")
            st.image(gradcam_overlay, caption="Grad-CAM Overlay", use_container_width=True)

        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
