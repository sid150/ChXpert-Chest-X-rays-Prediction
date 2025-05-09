import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, BackboneFinetuning
from lightning.pytorch.loggers import MLFlowLogger

import mlflow

import ray.train.torch
import ray.train.lightning
from ray.train import ScalingConfig, RunConfig, FailureConfig
from ray.train.torch import TorchTrainer
from ray.train.lightning import RayTrainReportCallback
from ray import train
csv_path = os.path.join(os.getcwd(), "filtered_chexpert_paths.csv")

class CheXpertDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_size=224):
        self.df = pd.read_csv(csv_path)
        self.image_paths = self.df["corrected_path"].values

        # Extract label columns from start to end
        start_col = "Enlarged Cardiomediastinum"
        end_col = "No Finding"
        label_columns = self.df.loc[:, start_col:end_col].columns

        # Load and convert -1 to 1
        self.labels = self.df[label_columns].astype(np.float32).values
        self.labels[self.labels == -1.0] = 1.0  # Convert -1s to 1s

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx].replace("/data/", "/")
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def get_dataloaders(csv_path, batch_size):
    dataset = CheXpertDataset(csv_path)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    return train_loader, val_loader
    
def train_func(config):
    class LightningCheXpertModel(L.LightningModule):
        def __init__(self, config, num_classes=14):
            super().__init__()
            self.save_hyperparameters()
            self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
            self.criterion = nn.BCEWithLogitsLoss()
            self.lr = config["lr"]

        @property
        def backbone(self):
            return self.model.features

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            mask = (y == 0.0) | (y == 1.0)
            loss = self.criterion(logits[mask], y[mask])
            self.log("train_loss", loss, prog_bar=True, sync_dist=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            probs = torch.sigmoid(logits)
            mask = (y == 0.0) | (y == 1.0)
            y_true = y[mask].detach().cpu().numpy()
            y_pred = (probs[mask] > 0.5).float().detach().cpu().numpy()

            loss = self.criterion(logits[mask], y[mask])
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            try:
                auroc = roc_auc_score(y_true, probs[mask].detach().cpu().numpy(), average='macro')
            except:
                auroc = 0.0

            self.log("val_loss", loss, prog_bar=True, sync_dist=True)
            self.log("val_precision", precision, prog_bar=True, sync_dist=True)
            self.log("val_recall", recall, prog_bar=True, sync_dist=True)
            self.log("val_f1", f1, prog_bar=True, sync_dist=True)
            self.log("val_accuracy", acc, prog_bar=True, sync_dist=True)
            self.log("val_auroc", auroc, prog_bar=True, sync_dist=True)
            return loss

        def configure_optimizers(self):
            return optim.Adam(self.model.classifier.parameters(), lr=self.lr)

    # === Load Data & Model ===
    train_loader, val_loader = get_dataloaders(csv_path, batch_size=config["batch_size"])
    model = LightningCheXpertModel(config)

    # === Callbacks ===
    early_stopping = EarlyStopping(monitor="val_loss", patience=config["patience"], mode="min")

    backbone_finetuning = BackboneFinetuning(
        unfreeze_backbone_at_epoch=config["initial_epochs"],
        backbone_initial_lr=config["fine_tune_lr"],
        should_align=True
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="chexpert",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    # === Trainer ===
    trainer = Trainer(
        max_epochs=config["total_epochs"],
        devices="auto",
        accelerator="auto",
        strategy=ray.train.lightning.RayDDPStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        callbacks=[early_stopping, backbone_finetuning, RayTrainReportCallback(), checkpoint_callback]
    )

    trainer = ray.train.lightning.prepare_trainer(trainer)

    # === MLflow Logging (Only on Primary Worker) ===
    if trainer.global_rank == 0:
        mlflow.set_tracking_uri("http://129.114.26.91:8000")
        mlflow.set_experiment("chexpert-classifier")
        mlflow.pytorch.autolog()
        mlflow.start_run(log_system_metrics=True)

    # === Fit + Resume from Checkpoint if Needed ===
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint.ckpt")
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # === Test on Validation Set ===
    # trainer.test(model, dataloaders=val_loader)

    # === End MLflow Run ===
    if trainer.global_rank == 0:
        mlflow.end_run()



# === Configurations ===
config = {
    "initial_epochs": 1,
    "total_epochs": 2,
    "patience": 5,
    "batch_size": 64,
    "lr": 1e-4,
    "fine_tune_lr": 1e-6,
}

run_config = RunConfig(storage_path="s3://ray", failure_config=FailureConfig(max_failures=2))
scaling_config = ScalingConfig(num_workers=2, use_gpu=True, resources_per_worker={"GPU": 1, "CPU": 30})

trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=run_config,
    train_loop_config=config
)

result = trainer.fit()
