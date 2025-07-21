"""
This module contains the DrumTranscriptionModule, the core PyTorch Lightning
module for the drum transcription model.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from chart_hero.model_training.transformer_config import BaseConfig
from chart_hero.model_training.transformer_model import create_model


class DrumTranscriptionModule(pl.LightningModule):
    """PyTorch Lightning module for drum transcription training."""

    def __init__(self, config: BaseConfig, max_time_patches: int | None = None):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.__dict__)

        self.model = create_model(config, max_time_patches=max_time_patches)

        pos_weights = torch.ones(config.num_drum_classes) * 2.0
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        self.train_f1 = torchmetrics.F1Score(
            task="multilabel", num_labels=config.num_drum_classes
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multilabel", num_labels=config.num_drum_classes
        )
        self.test_f1 = torchmetrics.F1Score(
            task="multilabel", num_labels=config.num_drum_classes
        )
        self.train_acc = torchmetrics.Accuracy(
            task="multilabel", num_labels=config.num_drum_classes
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multilabel", num_labels=config.num_drum_classes
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multilabel", num_labels=config.num_drum_classes
        )

        self.validation_step_outputs: list[dict[str, torch.Tensor]] = []
        self.test_step_outputs: list[dict[str, torch.Tensor]] = []

    def forward(self, spectrograms):
        return self.model(spectrograms)

    def _common_step(self, batch):
        spectrograms, labels = batch
        outputs = self.model(spectrograms)
        logits = outputs["logits"]

        # Reshape labels to match logits
        num_patches = logits.shape[1]
        labels = labels[:, :num_patches, :]

        logits = logits.reshape(-1, self.config.num_drum_classes)
        labels = labels.reshape(-1, self.config.num_drum_classes)

        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits) > 0.5
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch)
        self.train_f1(preds.int(), labels.int())
        self.train_acc(preds.int(), labels.int())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch)
        self.val_f1(preds.int(), labels.int())
        self.val_acc(preds.int(), labels.int())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.validation_step_outputs.append({"preds": preds, "labels": labels})
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch)
        self.test_f1(preds.int(), labels.int())
        self.test_acc(preds.int(), labels.int())
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.test_step_outputs.append({"preds": preds, "labels": labels})
        return loss

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
            all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
            if self.trainer.logger:
                for i in range(self.config.num_drum_classes):
                    class_f1 = torchmetrics.functional.f1_score(
                        all_preds[:, i], all_labels[:, i], task="binary"
                    )
                    self.log(f"val_f1_class_{i}", class_f1)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        if self.test_step_outputs:
            all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
            all_labels = torch.cat([x["labels"] for x in self.test_step_outputs])
            if self.trainer.logger:
                for i in range(self.config.num_drum_classes):
                    class_f1 = torchmetrics.functional.f1_score(
                        all_preds[:, i], all_labels[:, i], task="binary"
                    )
                    self.log(f"test_f1_class_{i}", class_f1)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        if self.trainer.max_epochs is None:
            raise ValueError("trainer.max_epochs must be set")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]
