from fastapi import FastAPI
from pydantic import BaseModel
import base64
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import os
import subprocess
from datetime import datetime

app = FastAPI(
    title="CheXpert X-ray classification API",
    description="API for classifying chest X-ray image diagnoses",
    version="1.0.0"
)

# Request and response models
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

class PredictionResponse(BaseModel):
    predictions: list[str]
    probabilities: list[float]

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
MODEL_PATH = "demoModel.pt"
model = torch.jit.load(MODEL_PATH)
model.to(device)
model.eval()

# Class labels
classes = np.array([
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
])

# Preprocessing
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

# Prediction endpoint
@app.post("/predict")
def predict_image(request: ImageRequest):
    try:
        # Decode image from base64
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Save image to disk with a unique filename
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = f"uploaded_image_{timestamp}.jpg"
        local_path = f"/tmp/{filename}"
        image.save(local_path)

        # Upload to object store via rclone
        RCLONE_CONTAINER = "object-persist-project29"
        remote_path = f"chi_tacc:{RCLONE_CONTAINER}/production/{filename}"

        try:
            result = subprocess.run(
                ["rclone", "copyto", local_path, remote_path],
                capture_output=True,
                text=True,
                check=True
            )
            print("Rclone upload output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("Rclone upload failed:", e.stderr)
            raise RuntimeError("Failed to upload image to object store")

        # Preprocess image and run inference
        image_tensor = preprocess_image(image).to(device)
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1).squeeze()
            prob_list = probabilities.tolist()

        # Remove temp file
        os.remove(local_path)

        return PredictionResponse(
            predictions=classes.tolist(),
            probabilities=prob_list
        )

    except Exception as e:
        return {"error": str(e)}
