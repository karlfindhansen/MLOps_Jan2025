from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import torch
from src.model import CustomClassifier
from google.cloud import storage
from torchvision import transforms
import io

# Initialize the Google Cloud Storage client
storage_client = storage.Client()

# Define the bucket and blob where the model is stored
bucket_name = "bird-calssifier-model"
blob_path = "files/md5/94/fd3ab04560561dbfdb4536c286e73a"
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(blob_path)

# Define the device (GPU or CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model from cloud storage directly without downloading it locally
def load_model_from_cloud():
    model_bytes = blob.download_as_bytes()  # Downloads model as a byte stream
    model = CustomClassifier(backbone='resnet50', num_classes=10, pretrained=False).to(DEVICE)

    # Load the state_dict directly from the byte stream
    model.load_state_dict(torch.load(io.BytesIO(model_bytes), map_location=DEVICE), strict=False)
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
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Apply the transformation and move the tensor to the device (GPU/CPU)
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # Perform inference with the model
        with torch.no_grad():
            output = model(input_tensor)

        # Get the predicted class
        predicted_class = output.argmax(dim=1)

        # Return the predicted class as a JSON response
        return JSONResponse(content={"predicted_class": predicted_class.item()})

    except Exception as e:
        # Handle any errors that occur during processing
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
