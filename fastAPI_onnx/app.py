from fastapi import FastAPI
from pydantic import BaseModel, Field
import base64
import onnxruntime as ort  # ONNX Runtime
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import io

app = FastAPI(
    title="CheXpert X-ray classification API ONNX version",
    description="API for classifying chest X-ray image diagnoses using ONNX runtime",
    version="1.0.0"
)


# Define the request and response models
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image


class PredictionResponse(BaseModel):
    predictions: list[str]
    probabilities: list[float]


# Load the CheXpert model
MODEL_PATH = "demoONNX.onnx"

# Set device (Use GPU if available, otherwise CPU)
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(MODEL_PATH, providers=providers)

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
    return transform(img).unsqueeze(0).numpy()


@app.post("/predict")
def predict_image(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocess the image
        image = preprocess_image(image)

        # Run inference with ONNX
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: image})

        probabilities = np.exp(result[0]) / np.sum(np.exp(result[0]))  # Softmax manually
        predicted_class_idx = np.argmax(probabilities)
        confidence = probabilities[0, predicted_class_idx]

        return PredictionResponse(prediction=classes[predicted_class_idx], probability=confidence)

    except Exception as e:
        return {"error": str(e)}
