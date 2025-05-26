"""
Main training script for transformer-based drum transcription.
Supports both local (M1-Max) and cloud (Google Colab) training environments.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torchmetrics
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import argparse
import logging
import wandb
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional

from .transformer_config import get_config, auto_detect_config, validate_config
from .transformer_model import create_model
from .transformer_data import create_data_loaders
from ..utils.file_utils import get_labeled_audio_set_dir


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrumTranscriptionModule(pl.LightningModule):
    """PyTorch Lightning module for drum transcription training."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.__dict__)
        
        # Create model
        self.model = create_model(config)
        
        # Loss function with label smoothing
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.ones(config.num_drum_classes) * 2.0  # Weight positive examples more
        )
        
        # Metrics
        self.train_f1 = torchmetrics.F1Score(task='multilabel', num_labels=config.num_drum_classes)
        self.val_f1 = torchmetrics.F1Score(task='multilabel', num_labels=config.num_drum_classes)
        self.test_f1 = torchmetrics.F1Score(task='multilabel', num_labels=config.num_drum_classes)
        
        self.train_acc = torchmetrics.Accuracy(task='multilabel', num_labels=config.num_drum_classes)
        self.val_acc = torchmetrics.Accuracy(task='multilabel', num_labels=config.num_drum_classes)
        self.test_acc = torchmetrics.Accuracy(task='multilabel', num_labels=config.num_drum_classes)
        
        # Store outputs for epoch-end processing
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, spectrograms):
        return self.model(spectrograms)
    
    def training_step(self, batch, batch_idx):
        spectrograms = batch['spectrogram']
        labels = batch['labels']
        
        # Forward pass
        outputs = self.model(spectrograms)
        logits = outputs['logits']
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Apply label smoothing manually if needed
        if self.config.label_smoothing > 0:
            smooth_labels = labels * (1 - self.config.label_smoothing) + \
                           self.config.label_smoothing / self.config.num_drum_classes
            loss = self.criterion(logits, smooth_labels)
        
        # Calculate metrics
        preds = torch.sigmoid(logits) > 0.5
        self.train_f1(preds.int(), labels.int())
        self.train_acc(preds.int(), labels.int())
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        spectrograms = batch['spectrogram']
        labels = batch['labels']
        
        # Forward pass
        outputs = self.model(spectrograms)
        logits = outputs['logits']
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        preds = torch.sigmoid(logits) > 0.5
        self.val_f1(preds.int(), labels.int())
        self.val_acc(preds.int(), labels.int())
        
        # Store for epoch-end processing
        self.validation_step_outputs.append({
            'loss': loss,
            'preds': preds,
            'labels': labels,
            'logits': logits
        })
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        spectrograms = batch['spectrogram']
        labels = batch['labels']
        
        # Forward pass
        outputs = self.model(spectrograms)
        logits = outputs['logits']
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        preds = torch.sigmoid(logits) > 0.5
        self.test_f1(preds.int(), labels.int())
        self.test_acc(preds.int(), labels.int())
        
        # Store for epoch-end processing
        self.test_step_outputs.append({
            'loss': loss,
            'preds': preds,
            'labels': labels,
            'logits': logits
        })
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        # Calculate per-class metrics
        if self.validation_step_outputs:
            all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
            all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
            
            # Per-class F1 scores
            for i in range(self.config.num_drum_classes):
                class_f1 = torchmetrics.functional.f1_score(
                    all_preds[:, i], all_labels[:, i], task='binary'
                )
                self.log(f'val_f1_class_{i}', class_f1)
        
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self):
        # Calculate per-class metrics
        if self.test_step_outputs:
            all_preds = torch.cat([x['preds'] for x in self.test_step_outputs])
            all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
            
            # Per-class F1 scores
            for i in range(self.config.num_drum_classes):
                class_f1 = torchmetrics.functional.f1_score(
                    all_preds[:, i], all_labels[:, i], task='binary'
                )
                self.log(f'test_f1_class_{i}', class_f1)
        
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        # Use AdamW optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=self.config.warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs - self.config.warmup_steps,
            eta_min=self.config.learning_rate * 0.01
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_steps]
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }


def setup_callbacks(config):
    """Set up training callbacks."""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.model_dir,
        filename='drum-transformer-{epoch:02d}-{val_f1:.3f}',
        monitor=config.monitor,
        mode=config.mode,
        save_top_k=config.save_top_k,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor=config.monitor,
        mode=config.mode,
        patience=10,
        verbose=True,
        min_delta=0.001
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks


def setup_logger(config, project_name: str = "chart-hero-transformer"):
    """Set up W&B logger."""
    logger = WandbLogger(
        project=project_name,
        name=f"drum-transformer-{config.device}",
        log_model=True,
        save_dir=config.log_dir
    )
    return logger


def train_model(config, data_loaders, resume_from_checkpoint: Optional[str] = None):
    """Main training function."""
    
    # Create model
    model = DrumTranscriptionModule(config)
    
    # Set up callbacks and logger
    callbacks = setup_callbacks(config)
    wandb_logger = setup_logger(config)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        devices=1,
        accelerator='auto',
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Train model
    train_loader, val_loader, test_loader = data_loaders
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from_checkpoint
    )
    
    # Test model
    if test_loader is not None:
        trainer.test(model, dataloaders=test_loader, ckpt_path='best')
    
    return model, trainer


def main():
    parser = argparse.ArgumentParser(description='Train transformer for drum transcription')
    parser.add_argument('--config', type=str, default='auto', 
                       choices=['local', 'cloud', 'auto'],
                       help='Configuration type')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing training data')
    parser.add_argument('--audio-dir', type=str, default=None,
                       help='Directory containing audio files')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--project-name', type=str, default='chart-hero-transformer',
                       help='W&B project name')
    
    args = parser.parse_args()
    
    # Get configuration
    if args.config == 'auto':
        config = auto_detect_config()
    else:
        config = get_config(args.config)
    
    # Override data directories if provided
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.audio_dir:
        config.audio_dir = args.audio_dir
    else:
        # Default to the labeled audio set directory
        config.audio_dir = str(Path(get_labeled_audio_set_dir()).parent)
    
    # Validate configuration
    validate_config(config)
    
    logger.info(f"Using {config.__class__.__name__} configuration")
    logger.info(f"Device: {config.device}")
    logger.info(f"Precision: {config.precision}")
    logger.info(f"Batch size: {config.train_batch_size}")
    logger.info(f"Effective batch size: {config.effective_batch_size}")
    
    try:
        # Create data loaders
        logger.info("Creating data loaders...")
        data_loaders = create_data_loaders(
            config=config,
            data_dir=config.data_dir,
            audio_dir=config.audio_dir
        )
        
        # Train model
        logger.info("Starting training...")
        model, trainer = train_model(
            config=config,
            data_loaders=data_loaders,
            resume_from_checkpoint=args.resume
        )
        
        logger.info("Training completed successfully!")
        
        # Save final model
        final_model_path = Path(config.model_dir) / "final_model.ckpt"
        trainer.save_checkpoint(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        # Clean up W&B
        wandb.finish()


if __name__ == "__main__":
    main()