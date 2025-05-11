import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
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
def get_dataloaders(csv_path, batch_size, seed=42):
    ds = CheXpertDataset(csv_path=csv_path)
    n  = len(ds)

    # 1) take 20% of the full dataset
    subset_size = int(0.05 * n)
    torch.manual_seed(seed)
    full_indices = torch.randperm(n)[:subset_size].tolist()
    subset_ds = Subset(ds, full_indices)

    # 2) split that 20% into 80/10/10
    train_len = int(0.8 * subset_size)
    val_len   = int(0.1 * subset_size)
    test_len  = subset_size - train_len - val_len

    train_ds, val_ds, test_ds = random_split(
        subset_ds,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed)
    )

    # 3) DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True
    )

    return train_loader, val_loader, test_loader

# === Training function ===
def train_func(config):
    class VitCheXpert(L.LightningModule):
        def __init__(self, cfg, num_classes=14):
            super().__init__()
            self.save_hyperparameters()
    
            # === Model & (optional) LoRA ===
            model_name = cfg.get("vit_model", "google/vit-base-patch16-224-in21k")
            self.model = ViTForImageClassification.from_pretrained(
                model_name, num_labels=num_classes
            )
    
            if cfg.get("use_lora", False):
                lora_cfg = LoraConfig(
                    r=8,
                    lora_alpha=32,
                    target_modules=["query", "value", "classifier"],
                    lora_dropout=0.05,
                    bias="none"
                )
                self.model = get_peft_model(self.model, lora_cfg)
    
            # === Loss & LR ===
            self.criterion = nn.BCEWithLogitsLoss()
            self.lr = cfg["lr"]
    
            # Buffers for test‐time aggregation
            self._test_true  = []
            self._test_probs = []
            self._test_preds = []
    
        @property
        def backbone(self):
            # for BackboneFinetuning callback
            return self.model.vit
    
        def forward(self, x):
            return self.model(pixel_values=x).logits
    
        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            mask   = (y == 0.0) | (y == 1.0)
            loss   = self.criterion(logits[mask], y[mask])
            self.log("train_loss", loss, sync_dist=True)
            return loss
    
        def validation_step(self, batch, batch_idx):
            x, y    = batch
            logits  = self(x)
            probs   = torch.sigmoid(logits)
            mask    = (y == 0.0) | (y == 1.0)
    
            # Flattened metrics
            y_true  = y[mask].cpu().numpy()
            y_pred  = (probs[mask] > 0.5).cpu().numpy()
    
            loss = self.criterion(logits[mask], y[mask])
    
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            acc = accuracy_score(y_true, y_pred)
            try:
                auroc = roc_auc_score(
                    y_true, probs[mask].cpu().numpy(), average='macro'
                )
            except ValueError:
                auroc = 0.0
    
            self.log_dict({
                'val_loss':     loss,
                'val_precision': prec,
                'val_recall':    rec,
                'val_f1':        f1,
                'val_accuracy':  acc,
                'val_auroc':     auroc
            }, sync_dist=True)
    
            return loss
    
        def test_step(self, batch, batch_idx):
            x, y    = batch
            logits  = self(x)
            probs   = torch.sigmoid(logits)
    
            # store for epoch‐end aggregation
            self._test_true.append(y.detach().cpu())
            self._test_probs.append(probs.detach().cpu())
            self._test_preds.append((probs > 0.5).detach().cpu().float())
    
            return
    
        def on_test_epoch_end(self):
            # concatenate all batches
            true  = torch.cat(self._test_true, dim=0).numpy()   # shape (N, C)
            probs = torch.cat(self._test_probs, dim=0).numpy()
            preds = torch.cat(self._test_preds, dim=0).numpy()
    
            # mask uncertain
            mask = (true == 0.0) | (true == 1.0)
            true_m   = true[mask]
            preds_m  = preds[mask]
            probs_m  = probs[mask]
    
            # 1) Label-based (micro) accuracy
            label_acc = accuracy_score(true_m, preds_m)
            self.log("test_label_accuracy", label_acc)
    
            # 2) Sample-based accuracy (all labels match per image)
            sample_acc = (preds == true).all(axis=1).mean()
            self.log("test_sample_accuracy", sample_acc)
    
            # 3) Per-class & macro metrics
            per_auc, per_prec, per_rec, per_f1 = [], [], [], []
    
            for c in range(true.shape[1]):
                idx = mask[:, c]
                if idx.sum() == 0:
                    per_auc.append(np.nan)
                    per_prec.append(np.nan)
                    per_rec.append(np.nan)
                    per_f1.append(np.nan)
                    continue
    
                y_t = true[idx, c]
                y_p = preds[idx, c]
                y_s = probs[idx, c]
    
                # AUC
                try:
                    auc = roc_auc_score(y_t, y_s)
                except ValueError:
                    auc = np.nan
                per_auc.append(auc)
    
                # P/R/F1
                p, r, f1, _ = precision_recall_fscore_support(
                    y_t, y_p, average="binary", zero_division=0
                )
                per_prec.append(p)
                per_rec.append(r)
                per_f1.append(f1)
    
                # log per-class if desired
                self.log(f"test_auc_class_{c}",  auc)
                self.log(f"test_prec_class_{c}", p)
                self.log(f"test_rec_class_{c}",  r)
                self.log(f"test_f1_class_{c}",   f1)
    
            # Macro-averages
            macro_auc  = np.nanmean(per_auc)
            macro_prec = np.nanmean(per_prec)
            macro_rec  = np.nanmean(per_rec)
            macro_f1   = np.nanmean(per_f1)
    
            self.log("test_auc_macro",      macro_auc)
            self.log("test_precision_macro", macro_prec)
            self.log("test_recall_macro",    macro_rec)
            self.log("test_f1_macro",        macro_f1)
    
            # clear buffers
            self._test_true.clear()
            self._test_probs.clear()
            self._test_preds.clear()
    
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

                
    train_loader, val_loader, test_loader = get_dataloaders(tsv_path, batch_size=config['batch_size'])
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
        
    trainer.test(model, test_loader)
    
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
    'resume_from_mlflow': False,
    'vit_model': 'google/vit-large-patch16-224-in21k',
    'mlflow_uri': 'http://129.114.26.91:8000',
    'mlflow_experiment': 'vit-chexpert',
    'traing_strat': 'DDPStrategy'
}
run_config = RunConfig(
    storage_path='s3://ray',
    failure_config=FailureConfig(max_failures=1)
)
scale_cfg = ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={'GPU':1,'CPU':30})
trainer = TorchTrainer(
    train_func,
    scaling_config=scale_cfg,
    run_config=run_config,
    train_loop_config=config
)
result = trainer.fit()