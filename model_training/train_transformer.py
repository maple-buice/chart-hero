"""
Main training script for transformer-based drum transcription.
Supports both local (M1-Max) and cloud (Google Colab) training environments.
"""

import os
import sys
import logging
import gc
import argparse
import torch
from typing import Optional
import wandb # Moved import to top level
from datetime import datetime # Added for logging timestamp

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
from pathlib import Path
import numpy as np
import subprocess

from model_training.transformer_config import get_config, auto_detect_config, validate_config
from model_training.transformer_model import create_model
from model_training.transformer_data import create_data_loaders
from utils.file_utils import get_labeled_audio_set_dir


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_experiment_logging(log_dir: str, experiment_tag: str):
    """Sets up file-based logging for a single experiment."""
    exp_log_file = Path(log_dir) / f"training_{experiment_tag}.log"
    exp_log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing handlers if any, to avoid duplicate logging
    current_logger = logging.getLogger() # Use a more specific variable name
    for handler in current_logger.handlers[:]:
        current_logger.removeHandler(handler)
        
    # Add new file handler for this specific experiment
    file_handler = logging.FileHandler(exp_log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Configure root logger - this will affect all loggers including Pytorch Lightning's
    # Use logging.getLogger() to get the root logger for basicConfig
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            file_handler,
            logging.StreamHandler(sys.stdout) # Also log to console
        ]
    )
    # Use the module-level logger for this specific message
    logger.info(f"Experiment-specific logging configured at: {exp_log_file}")


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

        # Check for NaN loss
        if torch.isnan(loss):
            logger.error("Loss is NaN. Stopping training.")
            logger.error(f"Spectrograms - min: {torch.min(spectrograms)}, max: {torch.max(spectrograms)}, mean: {torch.mean(spectrograms)}, has_nan: {torch.isnan(spectrograms).any()}, has_inf: {torch.isinf(spectrograms).any()}")
            logger.error(f"Labels - min: {torch.min(labels)}, max: {torch.max(labels)}, mean: {torch.mean(labels)}, has_nan: {torch.isnan(labels).any()}, has_inf: {torch.isinf(labels).any()}")
            logger.error(f"Logits - min: {torch.min(logits)}, max: {torch.max(logits)}, mean: {torch.mean(logits)}, has_nan: {torch.isnan(logits).any()}, has_inf: {torch.isinf(logits).any()}")
            logger.error(f"Loss: {loss.item()}")
            raise ValueError("Loss became NaN during training")
        
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

        # DEBUG: Check for NaN loss and log details
        if torch.isnan(loss):
            print(f"NaN loss detected in training_step! Batch Index: {batch_idx}")
            # Print hyperparams that might be relevant from self.config
            relevant_hyperparams = {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.train_batch_size,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "label_smoothing": self.config.label_smoothing,
                "device": self.config.device,
                "is_quick_test": getattr(self.config, 'is_quick_test', 'N/A')
            }
            print(f"Relevant Hyperparameters: {relevant_hyperparams}")
            print("Spectrogram stats:")
            print(f"  Shape: {spectrograms.shape}, Min: {spectrograms.min().item()}, Max: {spectrograms.max().item()}, Mean: {spectrograms.mean().item()}, Has NaN: {torch.isnan(spectrograms).any().item()}, Has Inf: {torch.isinf(spectrograms).any().item()}")
            print("Labels stats:")
            print(f"  Shape: {labels.shape}, Min: {labels.min().item()}, Max: {labels.max().item()}, Mean: {labels.mean().item()}, Has NaN: {torch.isnan(labels).any().item()}, Has Inf: {torch.isinf(labels).any().item()}")
            print("Logits stats:")
            print(f"  Shape: {logits.shape}, Min: {logits.min().item()}, Max: {logits.max().item()}, Mean: {logits.mean().item()}, Has NaN: {torch.isnan(logits).any().item()}, Has Inf: {torch.isinf(logits).any().item()}")
            
            # Optionally save tensors for offline analysis
            # from pathlib import Path
            # log_dir = Path(self.config.log_dir)
            # log_dir.mkdir(parents=True, exist_ok=True)
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # torch.save(spectrograms, log_dir / f"nan_spectrograms_batch{batch_idx}_{timestamp}.pt")
            # torch.save(labels, log_dir / f"nan_labels_batch{batch_idx}_{timestamp}.pt")
            # torch.save(logits, log_dir / f"nan_logits_batch{batch_idx}_{timestamp}.pt")
            # print(f"Saved tensors to {log_dir}")

            raise ValueError(f"NaN loss encountered during training_step at batch_idx {batch_idx}")
        
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

        # DEBUG: Check for NaN loss and log details
        if torch.isnan(loss):
            logger.error(f"NaN loss encountered during validation_step at batch_idx {batch_idx}")
            logger.error("--- Spectrogram Stats ---")
            logger.error(f"  Shape: {spectrograms.shape}")
            logger.error(f"  Min: {torch.min(spectrograms)}, Max: {torch.max(spectrograms)}, Mean: {torch.mean(spectrograms)}, Std: {torch.std(spectrograms)}")
            logger.error(f"  Has NaN: {torch.isnan(spectrograms).any()}, Has Inf: {torch.isinf(spectrograms).any()}")
            logger.error("--- Labels Stats ---")
            logger.error(f"  Shape: {labels.shape}")
            logger.error(f"  Min: {torch.min(labels)}, Max: {torch.max(labels)}, Mean: {torch.mean(labels.float())}, Std: {torch.std(labels.float())}") # Cast to float for mean/std
            logger.error(f"  Has NaN: {torch.isnan(labels).any()}, Has Inf: {torch.isinf(labels).any()}")
            logger.error("--- Logits Stats ---")
            logger.error(f"  Shape: {logits.shape}")
            logger.error(f"  Min: {torch.min(logits)}, Max: {torch.max(logits)}, Mean: {torch.mean(logits)}, Std: {torch.std(logits)}")
            logger.error(f"  Has NaN: {torch.isnan(logits).any()}, Has Inf: {torch.isinf(logits).any()}")
            logger.error(f"--- Loss ---: {loss}")
            # Optionally, save tensors for offline analysis
            # log_dir = Path(self.config.log_dir) / "nan_debug"
            # log_dir.mkdir(parents=True, exist_ok=True)
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # torch.save(spectrograms, log_dir / f"val_nan_spectrograms_batch{batch_idx}_{timestamp}.pt")
            # torch.save(labels, log_dir / f"val_nan_labels_batch{batch_idx}_{timestamp}.pt")
            # torch.save(logits, log_dir / f"val_nan_logits_batch{batch_idx}_{timestamp}.pt")
            # logger.info(f"Saved validation tensors to {log_dir}")
            # We might not want to raise an error here to allow other validation batches to run,
            # but for debugging, it can be useful.
            # raise ValueError(f"NaN loss encountered during validation_step at batch_idx {batch_idx}")
        
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
    # Ensure model_dir is an absolute path
    model_save_dir = Path(config.model_dir).resolve()
    model_save_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(model_save_dir), # Use resolved and stringified path
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
        patience=10, # Consider making this configurable
        verbose=True,
        min_delta=0.001 # Consider making this configurable
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks # Fix: return the callbacks list


def setup_logger(config, project_name: str = "chart-hero-transformer", use_wandb: bool = True, experiment_tag: Optional[str] = None):
    """Set up W&B logger or None if disabled."""
    if not use_wandb:
        logger.info("WandB logging disabled")
        return None
    
    run_name = f"drum-transformer-{config.device}"
    if experiment_tag:
        run_name += f"-{experiment_tag}"
        
    try:
        wandb_logger = WandbLogger(
            project=project_name,
            name=run_name,
            log_model=True,
            save_dir=config.log_dir
        )
        return wandb_logger
    except Exception as e:
        logger.warning(f"Failed to initialize WandB logger: {e}")
        logger.info("Continuing without WandB logging")
        return None


def train_model(config, data_loaders, resume_from_checkpoint: Optional[str] = None, use_wandb: bool = False, experiment_tag: Optional[str] = None):
    """Trains the model using PyTorch Lightning."""
    # logger = logging.getLogger(__name__) # logger is already defined at module level
    logger.info(f"Starting training with config: {config}")

    model = DrumTranscriptionModule(config)

    wandb_logger_instance = None
    if use_wandb:
        wandb_logger_instance = setup_logger(config, use_wandb=True, experiment_tag=experiment_tag) # Corrected call
    
    # Call setup_callbacks and unpack correctly
    callbacks_list = setup_callbacks(config) 

    # Unpack data loaders correctly if it's a tuple
    if isinstance(data_loaders, tuple) and len(data_loaders) == 3:
        train_loader, val_loader, test_loader = data_loaders
    elif isinstance(data_loaders, dict) and all(k in data_loaders for k in ['train', 'val', 'test']):
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        test_loader = data_loaders['test']
    else:
        raise ValueError("data_loaders must be a 3-element tuple (train, val, test) or a dict with 'train', 'val', 'test' keys.")
    
    trainer = pl.Trainer(
        logger=wandb_logger_instance,
        callbacks=callbacks_list, # Use the list of callbacks
        max_epochs=config.num_epochs,
        accelerator=config.device,
        devices=1 if config.device != 'cpu' else 'auto',
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        deterministic=config.deterministic_training,
        enable_progress_bar=True,
        enable_model_summary=True,
        limit_train_batches=5 if hasattr(config, 'is_quick_test') and config.is_quick_test else None,
        limit_val_batches=2 if hasattr(config, 'is_quick_test') and config.is_quick_test else None
    )
    
    logger.info("Starting model training...")
    try:
        if resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_from_checkpoint)
        else:
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    except Exception as e:
         logger.exception(f"Error during training: {e}")
         if use_wandb and wandb_logger_instance and wandb_logger_instance.experiment:
             wandb.finish(exit_code=1) # Ensure W&B run is marked as failed
         raise # Re-raise the exception

    if not (hasattr(config, 'is_quick_test') and config.is_quick_test):
        if test_loader:
            logger.info("Starting model testing...")
            trainer.test(model, dataloaders=test_loader)
        else:
            logger.info("No test_loader provided, skipping testing phase.")
    else:
        logger.info("Quick test mode: Skipping final testing phase.")
    
    # Finish wandb run if it was started and hasn't been finished due to an error
    if use_wandb and wandb_logger_instance and wandb_logger_instance.experiment: # Check if experiment exists
        if wandb.run: # Check if a run is active
             wandb.finish()

    # Return the trained model and trainer instance
    return model, trainer


def main():
    parser = argparse.ArgumentParser(description="Train a transformer model for drum transcription.")
    parser.add_argument("--config", type=str, default="auto", help="Configuration profile to use (e.g., local, cloud, overnight_default, or auto-detect).")
    parser.add_argument("--use-wandb", action="store_true", help="Enable WandB logging.")
    parser.add_argument("--quick-test", action="store_true", help="Run a quick test with minimal data and epochs.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (e.g., anomaly detection).")
    parser.add_argument("--experiment-tag", type=str, default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="A unique tag for this experiment run.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to processed data directory (overrides config).")
    parser.add_argument("--audio-dir", type=str, default=None, help="Path to audio directory (overrides config).")
    parser.add_argument("--monitor-gpu", action="store_true", help="Flag to indicate GPU monitoring is active (for internal script awareness).")
    # Add other CLI arguments as needed, e.g., for specific hyperparameters

    args = parser.parse_args()

    if args.use_wandb:
        # Potentially initialize wandb here or ensure it's configured if setup_logger doesn't do it fully
        logger.info("WandB logging enabled via CLI.")

    # Load base configuration
    config_profile_name = args.config
    if config_profile_name.lower() == "auto":
        logger.info("Configuration profile set to 'auto'. Auto-detecting configuration...")
        config = auto_detect_config() # Call without arguments
    else:
        logger.info(f"Loading configuration profile: {config_profile_name}")
        config = get_config(config_profile_name)
    
    # --- Apply CLI overrides to the loaded config ---
    if args.data_dir:
        config.data_dir = str(Path(args.data_dir).resolve())
        logger.info(f"Overriding data_dir with CLI argument: {config.data_dir}")
    elif not hasattr(config, 'data_dir') or not config.data_dir:
        # Fallback if not in config and not in CLI
        config.data_dir = str(Path(__file__).resolve().parent.parent / "datasets" / "processed")
        logger.warning(f"data_dir not found in config or CLI, using default: {config.data_dir}")

    if args.audio_dir:
        config.audio_dir = str(Path(args.audio_dir).resolve())
        logger.info(f"Overriding audio_dir with CLI argument: {config.audio_dir}")
    elif not hasattr(config, 'audio_dir') or not config.audio_dir:
        # Fallback if not in config and not in CLI
        config.audio_dir = str(Path(__file__).resolve().parent.parent / "datasets" / "e-gmd-v1.0.0")
        logger.warning(f"audio_dir not found in config or CLI, using default: {config.audio_dir}")
        
    # Ensure model_dir is set, default if not. This is used by ModelCheckpoint.
    if not hasattr(config, 'model_dir') or not config.model_dir:
        config.model_dir = str(Path(__file__).resolve().parent / "transformer_models" / args.experiment_tag)
        logger.warning(f"model_dir not specified in config, defaulting to: {config.model_dir}")
    else:
        # Append experiment_tag to user-defined model_dir for better organization
        config.model_dir = str(Path(config.model_dir).resolve() / args.experiment_tag)
        logger.info(f"Model checkpoints will be saved in: {config.model_dir}")
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)


    if args.monitor_gpu:
        config.monitor_gpu = True 
        logger.info("GPU monitoring flag set via CLI. train_transformer.py is aware.")
    
    if args.quick_test:
        config.is_quick_test = True
        # Apply quick test specific settings if not already in the 'quick_test' config profile
        config.num_epochs = getattr(config, 'quick_test_epochs', 1) # Example: use 1 epoch for quick test
        config.batch_size = getattr(config, 'quick_test_batch_size', min(config.batch_size if hasattr(config, 'batch_size') and config.batch_size else 4, 4))
        logger.info(f"Quick test mode enabled via CLI. Epochs: {config.num_epochs}, Batch Size: {config.batch_size}")

    if args.debug:
        config.is_debug_mode = True 
        if torch.backends.mps.is_available(): # Check if MPS is available
            # os.environ['PYTORCH_DEBUG'] = '1' # Example debug flag
            pass # Placeholder for MPS specific debug settings
        logger.info("Debug mode enabled via CLI.")
    # --- End CLI overrides ---

    # --- Setup Experiment Specific Logging ---
    if not hasattr(config, 'log_dir') or not config.log_dir:
        config.log_dir = str(Path(__file__).resolve().parent.parent / "logs")
        logger.info(f"Log directory not specified in config, defaulting to: {config.log_dir}")
    else:
        config.log_dir = str(Path(config.log_dir).resolve())
    
    setup_experiment_logging(config.log_dir, args.experiment_tag)
    # --- End Experiment Specific Logging Setup ---

    # Validate the final configuration
    if not hasattr(config, 'data_dir') or not Path(config.data_dir).exists():
        logger.error(f"Data directory does not exist: {getattr(config, 'data_dir', 'Not Set')}")
        sys.exit(1)
    # Add more validation as needed (e.g., for audio_dir)
    validate_config(config) # Assuming this function exists and is comprehensive

    logger.info(f"Final configuration for run {args.experiment_tag}: {config.__dict__ if hasattr(config, '__dict__') else config}")

    # Create data loaders
    logger.info("Creating data loaders...")
    # data_loaders = create_data_loaders(config)
    data_loaders = create_data_loaders(config, config.data_dir, config.audio_dir)


    # Setup Callbacks
    callbacks_list = setup_callbacks(config) 

    # --- Train the model ---
    logger.info("Starting model training...")
    trained_model, trainer_instance = train_model(
        config, 
        data_loaders, 
        resume_from_checkpoint=None, # No resume logic in this version
        use_wandb=args.use_wandb, 
        experiment_tag=args.experiment_tag
    )
    logger.info(f"Training finished for experiment: {args.experiment_tag}")

    # --- Optionally test the model ---
    if not (hasattr(config, 'is_quick_test') and config.is_quick_test):
        logger.info("Starting model testing...")
        trainer_instance.test(dataloaders=data_loaders['test'])
    else:
        logger.info("Quick test mode active, skipping model testing.")

if __name__ == "__main__":
    # Ensure project root is in sys.path if this script is run directly
    # This is already handled at the top of the file, but good to be mindful
    main()