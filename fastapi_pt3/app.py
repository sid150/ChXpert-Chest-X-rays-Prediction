from fastapi import FastAPI
from fastapi import HTTPException, status
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
import boto3
from botocore.exceptions import ClientError
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter
import os
import lightning

# Histogram for prediction confidence
confidence_histogram = Histogram(
    "prediction_confidence",
    "Model prediction confidence",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Count how often we predict each class
class_counter = Counter(
    "predicted_class_total",
    "Count of predictions per class",
    ['class_name']
)

experiment_to_prefix = {
    "chexpert-triage": "1",
    "chexpert-classifier": "2",
    "vit-chexpert": "3"
}

experiment_name = "vit-chexpert"
folder_prefix = experiment_to_prefix[experiment_name]
bucket_name = "mlflow-artifacts"

# Change to environment variable later:
mlflow.set_tracking_uri("http://129.114.26.91:8000/")
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.val_accuracy DESC"],
    max_results=10  # Check more runs if needed
)

s3 = boto3.client(
    's3',
    endpoint_url='http://129.114.26.91:9000',
    aws_access_key_id="myminioadmin",
    aws_secret_access_key="myminioadmin123"
)

found_path = None
found_bucket = None
selected_run_id = None

for run in runs:
    run_id = run.info.run_id
    val_acc = run.data.metrics.get("val_accuracy", None)
    print(f"\n Run ID: {run_id} | val_accuracy: {val_acc}")
    base_prefix = f"{folder_prefix}/{run_id}/artifacts"
    possible_paths = [
        f"{base_prefix}/model/data/model.pth",
        f"{base_prefix}/model/model.pth",
        f"{base_prefix}/checkpoints/latest_checkpoint.pth"
    ]

    for path in possible_paths:
        try:
            print(f"Checking: {path}")
            s3.head_object(Bucket=bucket_name, Key=path)
            # Found a valid model file
            found_path = path
            found_bucket = bucket_name
            selected_run_id = run_id
            print(f"Found model at: s3://{found_bucket}/{found_path}")
            break
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                raise  # Unexpected error
            # If 404, try next path
    if found_path:
        break  # Stop looping through runs if model found

if found_path:
    print(f"Using model from run: {selected_run_id}")
    local_file = "./downloaded_model.pth"
    s3.download_file(found_bucket, found_path, local_file)
    # Load it into PyTorch manually
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(local_file, map_location=device)
    model.eval()
    print("Model loaded and ready for inference.")
else:
    print("No valid model file found in top runs.")
    print("Getting DenseNet Model from local or demoModel")  # densenet model too large to fit on github
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_file = "./demoModel.pt"  # can copy over densenet model on local storage in place of demo model
    model = torch.load(local_file, map_location=device)
    model.eval()

# runs = client.search_runs(experiment_ids=[experiment.experiment_id],
#                           order_by=["metrics.val_accuracy DESC"],
#                           max_results=5)
# best_run = runs[0]  # The first run is the best due to sorting
# run_id = best_run.info.run_id
# print("Best run ID:", run_id)
# s3 = boto3.client(
#     's3',
#     endpoint_url='http://129.114.26.91:9000',
#     aws_access_key_id= "myminioadmin",  # os.getenv("MINIO_ROOT_USER"),
#     aws_secret_access_key= "myminioadmin123"  # os.getenv("MINIO_ROOT_PASSWORD")
# )
#
#
# base_prefix = f"{folder_prefix}/{run_id}/artifacts"
# # base_prefix = f"{folder_prefix}/mlflow/runs/{run_id}/artifacts"
# possible_paths = [
#     f"{base_prefix}/model/data/model.pth",
#     f"{base_prefix}/model/model.pth",
#     f"{base_prefix}/checkpoints/latest_checkpoint.pth"
# ]
#
# found_path = None
# found_bucket = None
#
# for path in possible_paths:
#     try:
#         print(path)
#         s3.head_object(Bucket=bucket_name, Key=path)
#         found_path = path
#         found_bucket = bucket_name
#         print(found_bucket)
#         print(found_path)
#         break
#     except ClientError as e:
#         print("error")
#         if e.response["Error"]["Code"] != "404":
#             raise  # Unexpected error
#
# # === Step 4: Handle the result ===
# if found_path:
#     print(f"Found model checkpoint at: s3://{found_path}")
#     local_file = "./downloaded_model.pth"
#     s3.download_file(found_bucket, found_path, local_file)
#
#     # Load it into PyTorch manually
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = torch.load(local_file, map_location=device)
#     model.eval()
#     print("Model loaded and ready for inference.")
# else:
#     print("Model checkpoint not found in any sub-bucket.")
#     print("Getting DenseNet Model from local")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     local_file = "./mlflowModel1.pt"
#     model = torch.load(local_file, map_location=device)
#     model.eval()


# # Set device (GPU if available, otherwise CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# model = mlflow.pytorch.load_model(model_uri)
# model.to(device)
# model.eval()

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
        transforms.Normalize([0.5] * 3, [0.5] * 3)
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
            logits = model(image)
            probabilities = torch.sigmoid(logits).squeeze(0)
            # probabilities = F.softmax(output, dim=1).squeeze()  # Shape: (num_classes,)
        prob_list = probabilities.tolist()

        # Log all confidence values
        for i, prob in enumerate(probabilities):
            confidence_histogram.observe(prob.item())
            if prob.item() > 0.5:
                class_counter.labels(class_name=classes[i]).inc()

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
