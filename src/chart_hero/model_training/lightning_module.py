"""
This module contains the DrumTranscriptionModule, the core PyTorch Lightning
module for the drum transcription model.
"""

from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from chart_hero.model_training.transformer_config import BaseConfig
from chart_hero.model_training.transformer_model import create_model


class DrumTranscriptionModule(pl.LightningModule):
    pos_weight: torch.Tensor
    criterion: Optional[nn.BCEWithLogitsLoss]
    """PyTorch Lightning module for drum transcription training."""

    def __init__(
        self,
        config: BaseConfig,
        max_time_patches: int | None = None,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.__dict__)

        self.model = create_model(config, max_time_patches=max_time_patches)

        if pos_weight is None and getattr(config, "class_pos_weight", None) is not None:
            pos_weight = torch.tensor(config.class_pos_weight, dtype=torch.float32)
        if pos_weight is None:
            pos_weight = torch.ones(config.num_drum_classes, dtype=torch.float32) * 2.0
        self.register_buffer("pos_weight", pos_weight)
        # Help the type checker: pos_weight is a Tensor buffer
        self.pos_weight = pos_weight
        if getattr(config, "use_focal_loss", False):
            self.criterion = None  # use custom focal in step
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

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

    def forward(self, spectrograms: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(spectrograms)

    def _common_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spectrograms, labels = batch
        outputs = self.model(spectrograms)
        logits = outputs["logits"]  # [B, T_patches, C]

        # Pool labels to patch-level to align with logits (robust to non-contiguous tensors)
        # Labels expected shape: [B, T_frames_or_patches, C]
        bsz, t_patches, num_classes = logits.shape
        labels = labels.float()
        # Optional dilation to add timing tolerance before pooling
        dilation = max(0, int(getattr(self.config, "label_dilation_frames", 0) or 0))
        if dilation > 0:
            # Max pool over time with kernel size = dilation*2+1, padding same
            k = dilation * 2 + 1
            pad = dilation
            labels = labels.permute(0, 2, 1)  # (B,C,T)
            labels = F.max_pool1d(labels, kernel_size=k, stride=1, padding=pad)
            labels = labels.permute(0, 2, 1)  # (B,T,C)
        # Use adaptive max pool along time to exactly t_patches steps
        labels = F.adaptive_max_pool1d(labels.permute(0, 2, 1), output_size=t_patches)
        labels = labels.permute(0, 2, 1)

        logits = logits.reshape(-1, self.config.num_drum_classes)
        labels = labels.reshape(-1, self.config.num_drum_classes)

        # Optional label smoothing for BCE
        eps = getattr(self.config, "label_smoothing", 0.0) or 0.0
        if eps > 0.0:
            labels = labels * (1.0 - eps) + 0.5 * eps

        if getattr(self.config, "use_focal_loss", False):
            # Focal BCE for multilabel
            p = torch.sigmoid(logits)
            alpha = getattr(self.config, "focal_alpha", 0.25)
            gamma = getattr(self.config, "focal_gamma", 2.0)
            # BCE per element
            bce = F.binary_cross_entropy(p, labels, reduction="none")
            pt = (1 - bce).clamp_min(1e-6)
            focal = (
                alpha * (1 - p) ** gamma * labels
                + (1 - alpha) * p**gamma * (1 - labels)
            ) * bce
            # Class weighting via pos_weight approximated by scaling positives
            pos_w = self.pos_weight.to(focal.device)
            focal = focal * (labels * (pos_w - 1) + 1)
            loss = focal.mean()
        else:
            assert self.criterion is not None
            loss = self.criterion(logits, labels)

        # Thresholds (global or per-class)
        raw_p = torch.sigmoid(logits)
        cls_thresh = getattr(self.config, "class_thresholds", None)
        if cls_thresh and len(cls_thresh) == self.config.num_drum_classes:
            thr = (
                torch.tensor(cls_thresh, device=raw_p.device)
                .unsqueeze(0)
                .expand_as(raw_p)
            )
        else:
            thr = torch.full_like(
                raw_p, getattr(self.config, "prediction_threshold", 0.5)
            )
        preds = raw_p > thr
        return loss, preds, labels

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, labels = self._common_step(batch)
        self.train_f1(preds.int(), labels.int())
        self.train_acc(preds.int(), labels.int())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, labels = self._common_step(batch)
        self.val_f1(preds.int(), labels.int())
        self.val_acc(preds.int(), labels.int())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.validation_step_outputs.append({"preds": preds, "labels": labels})
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, labels = self._common_step(batch)
        self.test_f1(preds.int(), labels.int())
        self.test_acc(preds.int(), labels.int())
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.test_step_outputs.append({"preds": preds, "labels": labels})
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.validation_step_outputs:
            all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
            all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
            if self.trainer.logger:
                for i in range(self.config.num_drum_classes):
                    class_f1 = torchmetrics.functional.f1_score(
                        all_preds[:, i], all_labels[:, i], task="binary"
                    )
                    self.log(f"val_f1_class_{i}", class_f1)
                # Event-level F1 (onset tolerance in patches)
                tol = max(0, int(getattr(self.config, "event_tolerance_patches", 1)))
                ev_p, ev_r, ev_f1 = self._event_level_prf(all_preds, all_labels, tol)
                self.log("val_event_precision", ev_p)
                self.log("val_event_recall", ev_r)
                self.log("val_event_f1", ev_f1, prog_bar=True)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        if self.test_step_outputs:
            all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
            all_labels = torch.cat([x["labels"] for x in self.test_step_outputs])
            if self.trainer.logger:
                for i in range(self.config.num_drum_classes):
                    class_f1 = torchmetrics.functional.f1_score(
                        all_preds[:, i], all_labels[:, i], task="binary"
                    )
                    self.log(f"test_f1_class_{i}", class_f1)
                tol = max(0, int(getattr(self.config, "event_tolerance_patches", 1)))
                ev_p, ev_r, ev_f1 = self._event_level_prf(all_preds, all_labels, tol)
                self.log("test_event_precision", ev_p)
                self.log("test_event_recall", ev_r)
                self.log("test_event_f1", ev_f1)
        self.test_step_outputs.clear()

    @staticmethod
    def _series_to_events(series: torch.Tensor) -> List[int]:
        """Convert 1D binary series to onset indices (rising edges)."""
        s = series.detach().to(torch.int8)
        # ensure 0 padded at start
        s = torch.cat([torch.tensor([0], dtype=torch.int8, device=s.device), s])
        diff = s[1:] - s[:-1]
        onsets = (diff > 0).nonzero(as_tuple=False).view(-1).tolist()
        return onsets

    def _match_events(
        self, pred: List[int], true: List[int], tol: int
    ) -> Tuple[int, int, int]:
        """Greedy match predicted to true events within +/- tol indices."""
        tp = 0
        used_true = set()
        for p in pred:
            # find closest true within tolerance
            best = None
            best_dist = tol + 1
            for idx, t in enumerate(true):
                if idx in used_true:
                    continue
                d = abs(p - t)
                if d <= tol and d < best_dist:
                    best = idx
                    best_dist = d
            if best is not None:
                tp += 1
                used_true.add(best)
        fp = len(pred) - tp
        fn = len(true) - tp
        return tp, fp, fn

    def _event_level_prf(
        self, preds: torch.Tensor, labels: torch.Tensor, tol: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute event-level precision/recall/F1 over all classes by concatenating events."""
        BxT, C = preds.shape
        # reshape to (B,T,C)
        B = -1  # unknown; treat as flat sequence
        tp_total = 0
        fp_total = 0
        fn_total = 0
        for c in range(C):
            series_pred = preds[:, c].to(torch.int8)
            series_true = labels[:, c].to(torch.int8)
            p_events = self._series_to_events(series_pred)
            t_events = self._series_to_events(series_true)
            tp, fp, fn = self._match_events(p_events, t_events, tol)
            tp_total += tp
            fp_total += fp
            fn_total += fn
        prec = torch.tensor(tp_total / max(1, tp_total + fp_total), dtype=torch.float32)
        rec = torch.tensor(tp_total / max(1, tp_total + fn_total), dtype=torch.float32)
        f1 = torch.tensor(
            0.0 if (prec + rec).item() == 0 else (2 * prec * rec / (prec + rec)),
            dtype=torch.float32,
        )
        return prec, rec, f1

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
