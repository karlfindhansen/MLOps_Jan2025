from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import torch
from src.model import CustomClassifier
from google.cloud import storage
from torchvision import transforms
import os

storage_client = storage.Client()

bucket_name = os.environ['BIRD_CLASSIFIER_MODEL']

bucket = storage_client.bucket(bucket_name)
blob_path = "files/md5/94/fd3ab04560561dbfdb4536c286e73a"
blob = bucket.blob(blob_path)
model_path = "model.pth"
blob.download_to_filename(model_path)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CustomClassifier(backbone = 'resnet50', num_classes=10, pretrained=False).to(DEVICE)

model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)

model.eval()

app = FastAPI()

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        if not image.filename.endswith((".jpg", "jpeg", ".png")):
            raise HTTPException(status_code=400, detail="File must be a JPEG or PNG image.")

        image_bytes = await image.read()
        img = Image.open(BytesIO(image_bytes))

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)

        predicted_class = output.argmax(dim=1)

        return JSONResponse(content={"predicted_class": predicted_class.item()})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
