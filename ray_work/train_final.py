import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, BackboneFinetuning
import pynvml
from lightning.pytorch.callbacks import Callback
import mlflow
import ray.train.torch
import ray.train.lightning
from ray.train import ScalingConfig, RunConfig, FailureConfig
from ray.train.torch import TorchTrainer
from ray.train.lightning import RayTrainReportCallback, RayDDPStrategy
from peft import LoraConfig, get_peft_model
from transformers import ViTForImageClassification
from ray import train
import json
from torchvision import transforms
# from lightning.pytorch.profiler import SimpleProfiler  # or AdvancedProfiler

# from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
# profiler = SimpleProfiler()

# Path to your CSV
tsv_path = os.path.join(os.getcwd(), "filtered_chexpert_paths.csv")

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
    ds = CheXpertDataset(csv_path=csv_path)
    n = len(ds)
    subset = int(0.01 * n)
    sv = torch.utils.data.Subset(ds, list(range(subset)))
    val_size = int(0.2 * subset)
    train_size = subset - val_size
    train_ds, val_ds = torch.utils.data.random_split(sv, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, val_loader

# === Training function ===
def train_func(config):
    class VitCheXpert(L.LightningModule):
        def __init__(self, cfg, num_classes=14):
            super().__init__()
            self.save_hyperparameters()
            model_name = cfg.get("vit_model", "google/vit-base-patch16-224-in21k")
            self.model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_classes)

            if cfg.get("use_lora", False):
                lora_cfg = LoraConfig(
                    r=8,
                    lora_alpha=32,
                    target_modules=["query", "value", "classifier"],  # include classification head
                    lora_dropout=0.05,
                    bias="none"
                )
                self.model = get_peft_model(self.model, lora_cfg)

            self.criterion = nn.BCEWithLogitsLoss()
            self.lr = cfg["lr"]

        @property
        def backbone(self):
            # BackboneFinetuning will freeze/unfreeze this module
            return self.model.vit

        def forward(self, x):
            return self.model(pixel_values=x).logits

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
            self.log_dict({
                'val_loss': loss,
                'val_precision': prec,
                'val_recall': rec,
                'val_f1': f1,
                'val_accuracy': acc,
                'val_auroc': auroc
            }, sync_dist=True)
            return loss

        def configure_optimizers(self):
            return optim.Adam(self.model.parameters(), lr=self.lr)
            
    class MLflowGPUMetricsCallback(Callback):
        def __init__(self, gpu_indices=None, log_every_n_steps=50):
            super().__init__()
            pynvml.nvmlInit()
            # by default log all visible GPUs
            self.gpu_indices = gpu_indices or list(range(pynvml.nvmlDeviceGetCount()))
            self.log_every_n_steps = log_every_n_steps
    
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            # only log on rank 0 / driver
            if trainer.global_rank != 0:
                return
            step = trainer.global_step
            if step % self.log_every_n_steps != 0:
                return
    
            for idx in self.gpu_indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                util  = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem   = pynvml.nvmlDeviceGetMemoryInfo(handle).used // (1024**2)
                # log two separate metrics per GPU
                mlflow.log_metric(f"gpu{idx}_util_pct", util, step=step)
                mlflow.log_metric(f"gpu{idx}_mem_mb",   mem,  step=step)

                
    train_loader, val_loader = get_dataloaders(tsv_path, batch_size=config['batch_size'])
    model = VitCheXpert(config)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config.get('patience', 5), mode='min'),
        BackboneFinetuning(
            unfreeze_backbone_at_epoch=config.get('initial_epochs', 1),
            backbone_initial_lr=config.get('fine_tune_lr', 1e-5),
            should_align=True
        ),
        RayTrainReportCallback(),
        ModelCheckpoint(dirpath='checkpoints/', filename='vit-chexpert', monitor='val_loss', mode='min', save_top_k=1),
        MLflowGPUMetricsCallback(log_every_n_steps=20)
    ]

    trainer = Trainer(
        max_epochs=config['total_epochs'],
        devices=1,
        accelerator='gpu',
        precision=config.get('precision'),
        profiler="simple",
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
        strategy=RayDDPStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        callbacks=callbacks
    )
    trainer = ray.train.lightning.prepare_trainer(trainer)

    if trainer.global_rank == 0:
        mlflow.set_tracking_uri(config.get('mlflow_uri', 'http://localhost:5000'))
        mlflow.set_experiment(config.get('mlflow_experiment', 'vit-chexpert'))
        mlflow.pytorch.autolog()
        mlflow.start_run(log_system_metrics=True)
        mlflow.log_params(config)

    ckpt = train.get_checkpoint()
    if ckpt:
        with ckpt.as_directory() as ckpt_dir:
            trainer.fit(model, train_loader, val_loader, ckpt_path=os.path.join(ckpt_dir, 'checkpoint.pt'))
    else:
        trainer.fit(model, train_loader, val_loader)

    if trainer.global_rank == 0:
        prof_summary = trainer.profiler.summary()
        prof_path = "profiler_summary.json"
        with open(prof_path, "w") as f:
            json.dump(prof_summary, f, indent=2)
        mlflow.log_artifact(prof_path, artifact_path="profiler")
        mlflow.end_run()

config = {
    'initial_epochs': 0,
    'data_percent_used': 1,
    'total_epochs': 1,
    'patience': 1,
    'batch_size': 32,
    'lr': 1e-4,
    'fine_tune_lr': 1e-5,
    'use_lora': False,
    'precision': '32',
    'accumulate_grad_batches': 1,
    'vit_model': 'google/vit-large-patch16-224-in21k',
    'mlflow_uri': 'http://129.114.26.91:8000',
    'mlflow_experiment': 'vit-chexpert',
    'traing_strat': 'DDPStrategy'
}
run_config = RunConfig(
    storage_path='s3://ray',
    failure_config=FailureConfig(max_failures=1)
)
scale_cfg = ScalingConfig(num_workers=2, use_gpu=True, resources_per_worker={'GPU':1,'CPU':30})
trainer = TorchTrainer(
    train_func,
    scaling_config=scale_cfg,
    run_config=run_config,
    train_loop_config=config
)
result = trainer.fit()