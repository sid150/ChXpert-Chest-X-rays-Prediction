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
