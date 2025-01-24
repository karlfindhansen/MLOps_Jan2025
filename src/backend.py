from io import BytesIO

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from src.model import CustomClassifier

# Define the device (GPU or CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Load the model from cloud storage directly without downloading it locally
def load_model_from_cloud():
    model = CustomClassifier(backbone="resnet50", num_classes=10, pretrained=False).to(DEVICE)
    model.eval()
    return model


# Load the model once at startup
model = load_model_from_cloud()

# Create the FastAPI app
app = FastAPI()


@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        # Check if the uploaded file is an image (JPEG, PNG, or JPG)
        if not image.filename.endswith((".jpg", "jpeg", ".png")):
            raise HTTPException(status_code=400, detail="File must be a JPEG or PNG image.")

        # Read the uploaded image bytes and process it
        image_bytes = await image.read()
        img = Image.open(BytesIO(image_bytes))

        # Define the image transformation for the model
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Apply the transformation and move the tensor to the device (GPU/CPU)
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # Perform inference with the model
        with torch.no_grad():
            output = model(input_tensor)

        # Get the predicted class

        probs = torch.nn.functional.softmax(output, dim=1)
        top5_probs, top5_classes = torch.topk(probs, 5)

        top5_probs = top5_probs.squeeze().tolist()
        top5_classes = top5_classes.squeeze().tolist()

        # Return the predicted classes and probabilities as a JSON response
        return JSONResponse(content={"top5_probabilities": top5_probs, "top5_classes": top5_classes})

    except Exception as e:
        # Handle any errors that occur during processing
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
