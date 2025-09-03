"""
This module contains the DrumTranscriptionModule, the core PyTorch Lightning
module for the drum transcription model.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

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
        self.calibrated_thresholds: Optional[list[float]] = None

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
            # reduction='none' so we can mask padded timesteps later
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight, reduction="none"
            )

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
        return cast(Dict[str, torch.Tensor], self.model(spectrograms))

    def _common_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(batch) == 3:
            spectrograms, labels, lengths = batch
        else:
            spectrograms, labels = batch  # type: ignore[misc]
            lengths = None
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

        # Build a patch-level mask from original lengths (exclude padded time)
        patch_mask: Optional[torch.Tensor] = None
        if lengths is not None:
            B = labels.shape[0]
            T_max = spectrograms.shape[-1]
            device = labels.device
            frame_mask = torch.zeros(B, T_max, dtype=torch.float32, device=device)
            for i in range(B):
                L = int(lengths[i].item())
                L = max(0, min(L, T_max))
                if L > 0:
                    frame_mask[i, :L] = 1.0
            patch_mask = F.adaptive_max_pool1d(
                frame_mask.unsqueeze(1), output_size=t_patches
            ).squeeze(1)

        # Prepare separate tensors for loss vs metrics
        # - metrics should use hard binary labels (no smoothing)
        # - loss may use smoothed labels when enabled
        labels_for_metrics = labels
        logits = logits.reshape(-1, self.config.num_drum_classes)
        labels_for_metrics = labels_for_metrics.reshape(
            -1, self.config.num_drum_classes
        )

        labels_for_loss = labels_for_metrics
        # Optional label smoothing for BCE (loss only)
        eps = getattr(self.config, "label_smoothing", 0.0) or 0.0
        if eps > 0.0:
            labels_for_loss = labels_for_loss * (1.0 - eps) + 0.5 * eps

        # Raw probabilities for thresholding/calibration
        raw_p = torch.sigmoid(logits)

        if getattr(self.config, "use_focal_loss", False):
            # Focal BCE for multilabel
            p = raw_p
            alpha = getattr(self.config, "focal_alpha", 0.25)
            gamma = getattr(self.config, "focal_gamma", 2.0)
            # BCE per element
            bce = F.binary_cross_entropy(p, labels_for_loss, reduction="none")
            focal = (
                alpha * (1 - p) ** gamma * labels_for_loss
                + (1 - alpha) * p**gamma * (1 - labels_for_loss)
            ) * bce
            # Class weighting via pos_weight approximated by scaling positives
            pos_w = self.pos_weight.to(focal.device)
            focal = focal * (labels_for_loss * (pos_w - 1) + 1)
            if patch_mask is not None:
                mask_flat = patch_mask.reshape(-1).unsqueeze(1).expand_as(focal) > 0
                loss = focal[mask_flat].mean() if mask_flat.any() else focal.mean()
            else:
                loss = focal.mean()
        else:
            assert self.criterion is not None
            per_el = self.criterion(logits, labels_for_loss)  # [N, C]
            if patch_mask is not None:
                mask_flat = patch_mask.reshape(-1).unsqueeze(1).expand_as(per_el) > 0
                loss = per_el[mask_flat].mean() if mask_flat.any() else per_el.mean()
            else:
                loss = per_el.mean()

        # Thresholds (global or per-class)
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
        # If we have a patch_mask, filter out padded rows for metrics/calibration
        if patch_mask is not None:
            valid = patch_mask.reshape(-1) > 0
            if valid.any():
                preds = preds[valid]
                labels_for_metrics = labels_for_metrics[valid]
                raw_p = raw_p[valid]
        # Return hard labels for metrics (no smoothing) and raw probabilities
        return loss, preds, labels_for_metrics, raw_p

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, preds, labels, _ = self._common_step(batch)
        self.train_f1(preds.int(), labels.int())
        self.train_acc(preds.int(), labels.int())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, preds, labels, probs = self._common_step(batch)
        self.val_f1(preds.int(), labels.int())
        self.val_acc(preds.int(), labels.int())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.validation_step_outputs.append(
            {"preds": preds, "labels": labels, "probs": probs}
        )
        return loss

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, preds, labels, _ = self._common_step(batch)
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
            all_probs = torch.cat([x["probs"] for x in self.validation_step_outputs])
            # Log base macro multilabel F1/Accuracy using current thresholds (always log to make it available
            # for callbacks like ModelCheckpoint/EarlyStopping even when no external logger is configured)
            base_f1 = torchmetrics.functional.f1_score(
                all_preds.int(),
                all_labels.int(),
                task="multilabel",
                num_labels=self.config.num_drum_classes,
            )
            self.log("val_f1", base_f1, prog_bar=True)
            base_acc = torchmetrics.functional.accuracy(
                all_preds.int(),
                all_labels.int(),
                task="multilabel",
                num_labels=self.config.num_drum_classes,
            )
            self.log("val_acc", base_acc)
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
                # Threshold calibration per class via simple grid search
                thrs = torch.linspace(0.05, 0.95, steps=19, device=all_probs.device)
                best_thrs: list[float] = []
                for i in range(self.config.num_drum_classes):
                    probs_i = all_probs[:, i]
                    labels_i = all_labels[:, i]
                    best_f1 = torch.tensor(0.0, device=probs_i.device)
                    best_thr = torch.tensor(
                        getattr(self.config, "prediction_threshold", 0.5),
                        device=probs_i.device,
                    )
                    for t in thrs:
                        preds_i = probs_i >= t
                        f1_i = torchmetrics.functional.f1_score(
                            preds_i, labels_i, task="binary"
                        )
                        if f1_i > best_f1:
                            best_f1 = f1_i
                            best_thr = t
                    self.log(f"val_best_thr_class_{i}", best_thr)
                    self.log(f"val_best_f1_class_{i}", best_f1)
                    best_thrs.append(float(best_thr.item()))

                # Store thresholds for saving/applying
                self.calibrated_thresholds = best_thrs

                # Macro multilabel F1 using calibrated per-class thresholds
                thr_vec = (
                    torch.tensor(best_thrs, device=all_probs.device)
                    .unsqueeze(0)
                    .expand_as(all_probs)
                )
                preds_cal = (all_probs >= thr_vec).int()
                cal_f1 = torchmetrics.functional.f1_score(
                    preds_cal,
                    all_labels.int(),
                    task="multilabel",
                    num_labels=self.config.num_drum_classes,
                )
                self.log("val_f1_calibrated", cal_f1, prog_bar=True)
        # Ensure the monitored scalar metrics are present at epoch end for checkpointing/early stopping
        self.validation_step_outputs.clear()
        # Apply calibrated thresholds to config for immediate downstream use
        if hasattr(self, "calibrated_thresholds") and self.calibrated_thresholds:
            self.config.class_thresholds = self.calibrated_thresholds

    def on_fit_end(self) -> None:
        # Persist calibrated thresholds to disk for reuse
        try:
            if (
                self.calibrated_thresholds
                and len(self.calibrated_thresholds) == self.config.num_drum_classes
            ):
                out_path = Path(self.config.model_dir) / "class_thresholds.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w") as f:
                    json.dump(
                        {
                            "class_thresholds": self.calibrated_thresholds,
                            "num_classes": self.config.num_drum_classes,
                        },
                        f,
                    )
                self.print(f"Saved calibrated thresholds to {out_path}")
        except Exception as e:
            self.print(f"Warning: failed to save calibrated thresholds: {e}")

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

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer],
        list[dict] | list[torch.optim.lr_scheduler._LRScheduler],
    ]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        sched_type = getattr(self.config, "lr_scheduler", "lambda_warmup_cosine")
        if sched_type == "cosine_epoch":
            if self.trainer.max_epochs is None:
                raise ValueError("trainer.max_epochs must be set")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
            return [optimizer], [scheduler]
        else:
            # Per-step linear warmup then cosine decay
            warmup_steps = int(getattr(self.config, "warmup_steps", 0) or 0)
            total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
            if total_steps is None:
                if self.trainer.max_epochs is None:
                    raise ValueError("trainer.max_epochs must be set")
                total_steps = self.trainer.max_epochs * 1000

            def lr_lambda(step: int) -> float:
                if total_steps <= 0:
                    return 1.0
                if step < warmup_steps and warmup_steps > 0:
                    return max(1e-8, float(step) / float(max(1, warmup_steps)))
                # Cosine from 1.0 to 0.0 over remaining steps
                progress = (step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                progress = min(max(progress, 0.0), 1.0)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [scheduler_conf]
