import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, Vit_B_16_Weights
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, BackboneFinetuning
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy

import mlflow
import ray.train.torch
import ray.train.lightning
from ray.train import ScalingConfig, RunConfig, FailureConfig
from ray.train.torch import TorchTrainer
from ray.train.lightning import RayTrainReportCallback
from peft import LoraConfig, get_peft_model
from ray import train

# Path to your CSV
csv_path = os.path.join(os.getcwd(), "filtered_chexpert_paths.csv")

# === Dataset ===
class CheXpertDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_size=224):
        df = pd.read_csv(csv_path)
        self.paths = df["corrected_path"].values
        start_col = "Enlarged Cardiomediastinum"
        end_col = "No Finding"
        labels = df.loc[:, start_col:end_col].astype(np.float32).values
        labels[labels == -1.0] = 1.0
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx].replace("/data/", "/")
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

# === Dataloaders ===
def get_dataloaders(csv_path, batch_size):
    full = CheXpertDataset(csv_path)
    n = len(full)
    subset = int(0.6 * n)
    train_val = torch.utils.data.Subset(full, list(range(subset)))
    val_size = int(0.2 * subset)
    train_size = subset - val_size
    train_ds, val_ds = torch.utils.data.random_split(train_val, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    return train_loader, val_loader

# === Training function ===

def train_func(config):
    # === Model Definition ===
    class VitCheXpert(L.LightningModule):
        def __init__(self, cfg, num_classes=14):
            super().__init__()
            self.save_hyperparameters()
            # Load ViT base
            weights = Vit_B_16_Weights.IMAGENET1K_V1
            self.model = vit_b_16(weights=weights)
            in_feats = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(in_feats, num_classes)

            # Apply LoRA if requested
            if cfg.get("use_lora", False):
                lora_cfg = LoraConfig(
                    r=8,
                    lora_alpha=32,
                    target_modules=["query", "value"],
                    lora_dropout=0.05,
                    bias="none"
                )
                self.model = get_peft_model(self.model, lora_cfg)

            self.criterion = nn.BCEWithLogitsLoss()
            self.lr = cfg["lr"]

        def forward(self, x): return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            mask = (y == 0.0) | (y == 1.0)
            loss = self.criterion(logits[mask], y[mask])
            self.log("train_loss", loss, sync_dist=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            probs = torch.sigmoid(logits)
            mask = (y == 0.0) | (y == 1.0)
            y_true = y[mask].cpu().numpy()
            y_pred = (probs[mask] > 0.5).cpu().numpy()
            loss = self.criterion(logits[mask], y[mask])
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            try:
                auroc = roc_auc_score(y_true, probs[mask].cpu().numpy(), average='macro')
            except:
                auroc = 0.0
            logs = {
                'val_loss': loss,
                'val_precision': prec,
                'val_recall': rec,
                'val_f1': f1,
                'val_accuracy': acc,
                'val_auroc': auroc
            }
            self.log_dict(logs, sync_dist=True)
            return loss

        def configure_optimizers(self):
            return optim.Adam(self.model.parameters(), lr=self.lr)

    # === Prepare data ===
    train_loader, val_loader = get_dataloaders(csv_path, batch_size=config['batch_size'])
    model = VitCheXpert(config)

    # === Callbacks ===
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config.get('patience', 5), mode='min'),
        BackboneFinetuning(unfreeze_backbone_at_epoch=config.get('initial_epochs', 1),
                            backbone_initial_lr=config.get('fine_tune_lr', 1e-5), should_align=True),
        RayTrainReportCallback(),
        ModelCheckpoint(dirpath='checkpoints/', filename='vit-chexpert', monitor='val_loss', mode='min', save_top_k=1)
    ]

    # === Trainer ===
    strategy = ray.train.lightning.RayDDPStrategy()
    # For native FSDP (without Ray), uncomment:
    # strategy = FSDPStrategy(auto_wrap_policy=nn.Linear, min_num_params=1e8)

    trainer = Trainer(
        max_epochs=config['total_epochs'],
        devices=2,
        accelerator='gpu',
        precision=config.get('precision', 32),      # e.g., 16 for mixed precision
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
        strategy=strategy,
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        callbacks=callbacks
    )
    trainer = ray.train.lightning.prepare_trainer(trainer)

    # === MLflow on primary ===
    if trainer.global_rank == 0:
        mlflow.set_tracking_uri(config.get('mlflow_uri', 'http://localhost:5000'))
        mlflow.set_experiment(config.get('mlflow_experiment', 'vit-chexpert'))
        mlflow.pytorch.autolog()
        mlflow.start_run(log_system_metrics=True)

    # === Fit (with resume) ===
    ckpt = train.get_checkpoint()
    if ckpt:
        with ckpt.as_directory() as ckpt_dir:
            trainer.fit(model, train_loader, val_loader, ckpt_path=os.path.join(ckpt_dir, 'checkpoint.ckpt'))
    else:
        trainer.fit(model, train_loader, val_loader)

    if trainer.global_rank == 0:
        mlflow.end_run()

# === Configs & Launch ===
config = {
    'initial_epochs': 1,
    'total_epochs': 10,
    'patience': 3,
    'batch_size': 32,
    'lr': 1e-4,
    'fine_tune_lr': 1e-5,
    'use_lora': True,
    'precision': 16,
    'accumulate_grad_batches': 2,
    # 'fsdp': True,
    'mlflow_uri': 'http://129.114.26.91:8000',
    'mlflow_experiment': 'vit-chexpert'
}

run_config  = RunConfig(storage_path='s3://ray', failure_config=FailureConfig(max_failures=2))
scale_cfg   = ScalingConfig(num_workers=2, use_gpu=True, resources_per_worker={'GPU':1,'CPU':20})

trainer = TorchTrainer(
    train_func,
    scaling_config=scale_cfg,
    run_config=run_config,
    train_loop_config=config
)

result = trainer.fit()
