from fastapi import FastAPI
from pydantic import BaseModel, Field
import base64
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np

app = FastAPI(
    title="CheXpert X-ray classification API",
    description="API for classifying chest X-ray image diagnoses",
    version="1.0.0"
)


# Define the request and response models
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image


class PredictionResponse(BaseModel):
    predictions: list[str]
    probabilities: list[float]


# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CheXpert model
MODEL_PATH = "demoModel.pt"
model = torch.jit.load(MODEL_PATH)
model.to(device)
model.eval()

# Define class labels
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


# Define the image preprocessing function
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform(img).unsqueeze(0)


@app.post("/predict")
def predict_image(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocess the image
        image = preprocess_image(image).to(device)

        # Run inference
        # with torch.no_grad():
        #     output = model(image)
        #     probabilities = F.softmax(output, dim=1)  # Apply softmax to get probabilities
        #     predicted_class = torch.argmax(probabilities, 1).item()
        #     confidence = probabilities[0, predicted_class].item()  # Get the probability
        #
        # return PredictionResponse(predictions=[classes[predicted_class]], probabilities=[confidence])
        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1).squeeze()  # Shape: (num_classes,)
            prob_list = probabilities.tolist()

        # Return all class names and their probabilities
        return PredictionResponse(
            predictions=classes.tolist(),  # All class names
            probabilities=prob_list  # Corresponding probabilities
        )



    except Exception as e:
        return {"error": str(e)}
