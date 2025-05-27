"""
Main training script for transformer-based drum transcription.
Supports both local (M1-Max) and cloud (Google Colab) training environments.
"""

import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
import gc
from typing import Dict, Any, Optional
import subprocess

from model_training.transformer_config import get_config, auto_detect_config, validate_config
from model_training.transformer_model import create_model
from model_training.transformer_data import create_data_loaders
from utils.file_utils import get_labeled_audio_set_dir


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def optimize_memory(force_gc: bool = True, aggressive: bool = False):
    """
    Optimize memory usage by clearing cache and optionally force garbage collection.
    
    Args:
        force_gc: Whether to force garbage collection
        aggressive: Whether to use more aggressive memory optimization
    """
    # First run GC to clean up any unused Python objects
    if force_gc:
        gc.collect(generation=2)  # Force full collection
    
    if torch.backends.mps.is_available():
        # Clear MPS cache
        torch.mps.empty_cache()
        
        # Collect again after cache clear
        if force_gc:
            gc.collect(generation=2)
    
    if aggressive and torch.backends.mps.is_available():
        # More aggressive memory optimization for MPS
        try:
            # Multiple rounds of creating and deleting tensors to trigger memory cleanup
            for size in [500, 1000, 1500]:
                temp = torch.ones(1, size, size, device='mps')
                del temp
                torch.mps.empty_cache()
                gc.collect(generation=2)
        except Exception as e:
            print(f"Memory optimization warning: {e}")
            pass


class DrumTranscriptionModule(pl.LightningModule):
    """PyTorch Lightning module for drum transcription training."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.__dict__)
        
        # Create model
        self.model = create_model(config)
        
        # Loss function with label smoothing
        device_type = 'mps' if torch.backends.mps.is_available() and config.device == 'mps' else 'cpu'
        pos_weights = torch.ones(config.num_drum_classes, device=device_type) * 2.0
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weights  # Weight positive examples more
        )
        
        # Metrics - ensure they're on the correct device
        device_type = 'mps' if torch.backends.mps.is_available() and config.device == 'mps' else 'cpu'
        self.train_f1 = torchmetrics.F1Score(task='multilabel', num_labels=config.num_drum_classes).to(device_type)
        self.val_f1 = torchmetrics.F1Score(task='multilabel', num_labels=config.num_drum_classes).to(device_type)
        self.test_f1 = torchmetrics.F1Score(task='multilabel', num_labels=config.num_drum_classes).to(device_type)
        
        self.train_acc = torchmetrics.Accuracy(task='multilabel', num_labels=config.num_drum_classes).to(device_type)
        self.val_acc = torchmetrics.Accuracy(task='multilabel', num_labels=config.num_drum_classes).to(device_type)
        self.test_acc = torchmetrics.Accuracy(task='multilabel', num_labels=config.num_drum_classes).to(device_type)
        
        # Store outputs for epoch-end processing
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, spectrograms):
        return self.model(spectrograms)
    
    def training_step(self, batch, batch_idx):
        spectrograms = batch['spectrogram']
        labels = batch['labels']
        
        # Ensure data is on the correct device
        device = self.device
        if spectrograms.device != device:
            spectrograms = spectrograms.to(device)
        if labels.device != device:
            labels = labels.to(device)
        
        # Periodically optimize memory every 20 batches
        if batch_idx % 20 == 0 and torch.backends.mps.is_available():
            optimize_memory(aggressive=(batch_idx % 100 == 0))
            
            # Log memory stats if using MPS
            if hasattr(torch.mps, 'current_allocated_memory'):
                self.log('gpu_mem_allocated', torch.mps.current_allocated_memory() / (1024**3), on_step=True)
        
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
        
        # Ensure data is on the correct device
        device = self.device
        if spectrograms.device != device:
            spectrograms = spectrograms.to(device)
        if labels.device != device:
            labels = labels.to(device)
        
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
        
        # Ensure data is on the correct device
        device = self.device
        if spectrograms.device != device:
            spectrograms = spectrograms.to(device)
        if labels.device != device:
            labels = labels.to(device)
        
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
        
        # Clear memory
        self.validation_step_outputs.clear()
        optimize_memory()
        
    def on_train_epoch_end(self):
        # Call parent method if it exists
        super().on_train_epoch_end() if hasattr(super(), "on_train_epoch_end") else None
        
        # Free up memory
        optimize_memory()
    
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
        
        # Clear memory
        self.test_step_outputs.clear()
        optimize_memory()
    
    def configure_optimizers(self):
        # Use AdamW optimizer with MPS-optimized parameters
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),  # Standard betas
            eps=1e-8,  # More stable epsilon value
            foreach=True  # Enable more efficient parameter updates
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


def setup_logger(config, project_name: str = "chart-hero-transformer", use_wandb: bool = True):
    """Set up W&B logger or None if disabled."""
    if not use_wandb:
        logger.info("WandB logging disabled")
        return None
    
    try:
        wandb_logger = WandbLogger(
            project=project_name,
            name=f"drum-transformer-{config.device}",
            log_model=True,
            save_dir=config.log_dir
        )
        return wandb_logger
    except Exception as e:
        logger.warning(f"Failed to initialize WandB logger: {e}")
        logger.info("Continuing without WandB logging")
        return None


def train_model(config, data_loaders, resume_from_checkpoint: Optional[str] = None, use_wandb: bool = False):
    """Main training function."""
    
    # Create model
    model = DrumTranscriptionModule(config)
    
    # Set up callbacks and logger
    callbacks = setup_callbacks(config)
    wandb_logger = setup_logger(config, use_wandb=use_wandb)
    
    # Create trainer - explicitly set accelerator for MPS
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        devices=1,
        accelerator='mps' if config.device == 'mps' and torch.backends.mps.is_available() else 'auto',
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
    parser.add_argument('--use-wandb', action='store_true',
                       help='Enable W&B logging (disabled by default)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run a quick test with minimal epochs for validation')
    parser.add_argument('--monitor-gpu', action='store_true',
                       help='Enable detailed GPU monitoring during training (logs to logs/gpu_monitoring.csv)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output for troubleshooting')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override the train batch size')
    parser.add_argument('--hidden-size', type=int, default=None,
                       help='Override the model hidden size')
    
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
    
    # Apply quick test modifications if requested
    if args.quick_test:
        logger.info("Quick test mode enabled - reducing epochs and batch size for fast validation")
        config.num_epochs = 1
        # Ensure quick test batch sizes are not smaller than MPS conservative defaults if on MPS
        if config.device == "mps":
            config.train_batch_size = min(config.train_batch_size, 4) 
            config.val_batch_size = min(config.val_batch_size, 8)
        else:
            config.train_batch_size = min(config.train_batch_size, 4)
            config.val_batch_size = min(config.val_batch_size, 8)

    # Memory safety: further reduce batch sizes if on MPS (Apple Silicon)
    # This section is now less critical if LocalConfig defaults are conservative,
    # but kept for safety if other configs are used or batch_size arg is high.
    if config.device == "mps":
        logger.info("MPS device detected - applying conservative memory settings")
        if not args.batch_size:  # Only apply if batch size wasn't manually specified
            config.train_batch_size = min(config.train_batch_size, 4)
            config.val_batch_size = min(config.val_batch_size, 8)
        config.num_workers = min(config.num_workers, 2)  # Reduce workers for MPS
    
    # Apply batch size and hidden size overrides if provided
    if args.batch_size:
        config.train_batch_size = args.batch_size
        config.val_batch_size = args.batch_size * 2  # Double for validation
        logger.info(f"Batch size overridden to {args.batch_size}")
        
    if args.hidden_size:
        config.hidden_size = args.hidden_size
        # Adjust other model parameters to maintain proportions
        config.num_heads = max(2, args.hidden_size // 64)  # 1 head per 64 hidden units
        config.intermediate_size = args.hidden_size * 4  # Standard ratio
        logger.info(f"Model size overridden to hidden_size={args.hidden_size}, heads={config.num_heads}")
    
    # Validate configuration
    validate_config(config)
    
    logger.info(f"Using {config.__class__.__name__} configuration")
    logger.info(f"Device: {config.device}")
    logger.info(f"Precision: {config.precision}")
    logger.info(f"Batch size: {config.train_batch_size}")
    logger.info(f"Effective batch size: {config.effective_batch_size}")
    
    try:
        # Add memory monitoring
        import psutil
        process = psutil.Process()
        
        # GPU monitoring setup
        gpu_monitor_process = None
        if args.monitor_gpu:
            if config.device == "mps":
                try:
                    # Using the consolidated monitor_mps.py script
                    monitor_script_path = Path(__file__).parent.parent / "monitor_mps.py"
                    log_file_path = Path(config.log_dir) / "gpu_monitoring.csv"
                    # Ensure logs directory exists
                    os.makedirs(config.log_dir, exist_ok=True)
                    
                    # Command to run the monitoring script
                    # Ensure python3 is used, adjust if your env uses 'python'
                    cmd = [
                        "python3", str(monitor_script_path),
                        "--pid", str(os.getpid()),
                        "--log-file", str(log_file_path),
                        "--interval", "5" # Log every 5 seconds, adjust as needed
                    ]
                    
                    # Start the monitoring script as a background process
                    gpu_monitor_process = subprocess.Popen(cmd)
                    logger.info(f"Started GPU monitoring. Logging to {log_file_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to start GPU monitoring: {e}")
            else:
                logger.info("GPU monitoring is currently configured for MPS devices only.")

        # Force garbage collection before starting
        optimize_memory(force_gc=True)
        initial_memory = process.memory_info().rss / (1024**3)
        logger.info(f"Initial memory usage: {initial_memory:.2f} GB")
        
        # Create data loaders
        logger.info("Creating data loaders...")
        data_loaders = create_data_loaders(
            config=config,
            data_dir=config.data_dir,
            audio_dir=config.audio_dir
        )
        
        # Check memory after data loader creation
        memory_after_data = process.memory_info().rss / (1024**3)
        logger.info(f"Memory after data loader creation: {memory_after_data:.2f} GB")
        
        # Train model
        logger.info("Starting training...")
        model, trainer = train_model(
            config=config,
            data_loaders=data_loaders,
            resume_from_checkpoint=args.resume,
            use_wandb=args.use_wandb
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
        # Clean up W&B if it was used
        if args.use_wandb:
            wandb.finish()
        
        # Terminate GPU monitoring process if it was started
        if gpu_monitor_process:
            try:
                gpu_monitor_process.terminate()
                gpu_monitor_process.wait(timeout=5) # Wait for graceful termination
                logger.info("GPU monitoring process terminated.")
            except subprocess.TimeoutExpired:
                gpu_monitor_process.kill() # Force kill if terminate times out
                logger.warning("GPU monitoring process force-killed.")
            except Exception as e:
                logger.error(f"Error terminating GPU monitoring process: {e}")

if __name__ == "__main__":
    # Add subprocess import if not already present at the top of the file
    # This is a placeholder, ensure 'import subprocess' is at the top of train_transformer.py
    import subprocess 
    main()