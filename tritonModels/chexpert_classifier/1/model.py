import os
import io
import base64
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import triton_python_backend_utils as pb_utils

class TritonPythonModel:

    def initialize(self, args):
        model_dir = os.path.dirname(__file__)
        model_path = os.path.join(model_dir, "chexpert_model.pth")

        instance_kind = args.get("model_instance_kind", "cpu").lower()
        if instance_kind == "gpu":
            device_id = int(args.get("model_instance_device_id", 0))
            torch.cuda.set_device(device_id)
            self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.model = torch.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        self.classes = np.array([
            "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
            "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
            "Pleural Other", "Fracture", "Support Devices"
        ])

    def preprocess(self, base64_input):
        if isinstance(base64_input, bytes):
            base64_input = base64_input.decode("utf-8")

        image_bytes = base64.b64decode(base64_input)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0)
        return img_tensor

    def execute(self, requests):
        responses = []

        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
            base64_str = in_tensor.as_numpy()[0][0]  # shape: [1, 1]
            input_tensor = self.preprocess(base64_str).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)  # shape [1, 14]
                probabilities = torch.sigmoid(output).squeeze().cpu().numpy()

            # Prepare Triton outputs
            out_labels_np = self.classes.astype(object)        # [14], dtype=object
            out_probs_np = probabilities.astype(np.float32)    # [14], dtype=float32

            out_tensor_labels = pb_utils.Tensor("CLASS_LABELS", out_labels_np)
            out_tensor_probs = pb_utils.Tensor("CONFIDENCES", out_probs_np)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_labels, out_tensor_probs]
            )
            responses.append(inference_response)

        return responses
