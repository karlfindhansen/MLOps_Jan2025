import streamlit as st
import torch
import timm
from PIL import Image
import numpy as np
from config import NUM_CLASSES
import os
import matplotlib.pyplot as plt
from explainability import compute_gradcam, overlay_gradcam_on_image
# from google.cloud import run_v2


def preprocess_image(image_path, input_size, mean, std):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(input_size, Image.BILINEAR)
    image = np.array(image).astype(np.float32) / 255.0

    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    image = (image - mean) / std
    image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
    return image.unsqueeze(0)


def load_model(model_path, model_name, num_classes):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def main():
    st.title("Bird Classifier with Saliency Map")

    uploaded_file = st.file_uploader("Choose an image of your favorite bird...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Bird Image.", use_container_width=True)

        input_size = (224, 224)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_tensor = preprocess_image(uploaded_file, input_size, mean, std)

        model_name = "resnet50"
        num_classes = NUM_CLASSES
        model = load_model("../model.pth", model_name, num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)
        model = model.to(device)

        input_tensor.requires_grad_()
        with torch.no_grad():
            output = model(input_tensor)

        probabilities = torch.nn.functional.softmax(output, dim=1)
        top5_probabilities, top5_classes = torch.topk(probabilities, 5)

        class_names = os.listdir("../data")[:num_classes]
        class_names = [name[4:].replace("_", " ") for name in class_names]

        top5_classes = top5_classes.cpu().numpy().flatten()
        top5_probabilities = top5_probabilities.cpu().numpy().flatten()

        st.write("Top 5 Predictions:")
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

        # Compute Grad-CAM for the top predicted class
        gradcam = compute_gradcam(model, input_tensor, top5_classes[0])
        gradcam_overlay = overlay_gradcam_on_image(image, gradcam)

        st.image(gradcam_overlay, caption="Grad-CAM Overlay", use_container_width=True)


if __name__ == "__main__":
    main()
