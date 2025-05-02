import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx


# Define dummy CheXpert model
class DummyCheXpertModel(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Instantiate and eval mode
model = DummyCheXpertModel()
model.eval()

# Save only the state_dict
model_scripted = torch.jit.script(model)  # Export to TorchScript
savePath = "models/demoModel.pt"
model_scripted.save(savePath)  # Save

print("Saved model to {}".format(savePath))

# model = torch.jit.load(MODEL_PATH)
onnx_model_path = "models/demoONNX.onnx"
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, onnx_model_path,
                  export_params=True, opset_version=20,
                  do_constant_folding=True, input_names=['input'],
                  output_names=['output'], dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

print(f"ONNX model saved to {onnx_model_path}")

onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
