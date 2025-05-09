from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel, Field
import base64
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from prometheus_fastapi_instrumentator import Instrumentator

# Change to environment variable later:
mlflow.set_tracking_uri("http://129.114.26.91:8000/")

client = MlflowClient()
experiment = client.get_experiment_by_name("chexpert-classifier")
runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                          order_by=["metrics.val_accuracy DESC"],
                          max_results=2)
best_run = runs[0]  # The first run is the best due to sorting
best_run_id = best_run.info.run_id
model_uri = f"runs:/{best_run_id}/model"


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
# try:
#     model = mlflow.pytorch.load_model(model_uri)
#     print("Model loaded successfully from MLflow.")
# except Exception as e:
#     print(f"Failed to load model from MLflow: {e}")
#     MODEL_PATH = "demoModel.pt"
#     model = torch.jit.load(MODEL_PATH)
#     model.to(device)
#     model.eval()
model = mlflow.pytorch.load_model(model_uri)
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
    image_size = 224
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
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

        with torch.no_grad():
            logits = model(image.unsqueeze(0))
            probabilities = torch.sigmoid(logits).squeeze()
            # probabilities = F.softmax(output, dim=1).squeeze()  # Shape: (num_classes,)
            prob_list = probabilities.tolist()

        # Return all class names and their probabilities
        return PredictionResponse(
            predictions=classes.tolist(),  # All class names
            probabilities=prob_list  # Corresponding probabilities
        )

    except base64.binascii.Error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid base64 input")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}")

Instrumentator().instrument(app).expose(app)
