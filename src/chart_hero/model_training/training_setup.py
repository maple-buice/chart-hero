"""
This module contains functions for setting up the training environment,
including argument parsing, configuration loading, and callback/logger setup.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

import wandb
from chart_hero.model_training.transformer_config import (
    auto_detect_config,
    get_config,
    validate_config,
)

logger = logging.getLogger(__name__)


def setup_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train transformer model for drum transcription."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="auto",
        choices=[
            "local",
            "local_performance",
            "local_max_performance",
            "cloud",
            "auto",
            "overnight_default",
        ],
        help="Configuration to use",
    )
    parser.add_argument("--use-wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick test with a small subset of data",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--experiment-tag", type=str, help="Tag for the experiment")
    parser.add_argument(
        "--project-name",
        type=str,
        default="chart-hero-transformer",
        help="Name of the W&B project",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable the progress bar",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on the test set",
    )
    parser.add_argument("--data-dir", type=str, help="Override data directory")
    parser.add_argument("--audio-dir", type=str, help="Override audio directory")
    parser.add_argument(
        "--monitor-gpu",
        action="store_true",
        help="Flag for GPU monitoring awareness",
    )
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--hidden-size", type=int, help="Override hidden size")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--num-workers", type=int, help="Override number of workers")
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        help="Override gradient accumulation batches",
    )
    return parser


def configure_paths(config, args):
    if args.data_dir:
        config.data_dir = str(Path(args.data_dir).resolve())
    if args.audio_dir:
        config.audio_dir = str(Path(args.audio_dir).resolve())

    if not hasattr(config, "model_dir") or not config.model_dir:
        config.model_dir = Path("models") / args.experiment_tag
    else:
        config.model_dir = Path(config.model_dir) / args.experiment_tag
    config.model_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(config, "log_dir") or not config.log_dir:
        config.log_dir = Path("logs")
    else:
        config.log_dir = Path(config.log_dir)
    config.log_dir.mkdir(parents=True, exist_ok=True)


def apply_cli_overrides(config, args):
    configure_paths(config, args)
    if args.batch_size is not None:
        config.train_batch_size = args.batch_size
        config.val_batch_size = args.batch_size
    if args.hidden_size is not None:
        config.hidden_size = args.hidden_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.accumulate_grad_batches is not None:
        config.accumulate_grad_batches = args.accumulate_grad_batches
    if args.quick_test:
        config.is_quick_test = True
        config.num_epochs = 1
    if args.debug:
        config.is_debug_mode = True


def setup_callbacks(config):
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(config.model_dir),
        filename="drum-transformer-{epoch:02d}-{val_f1:.3f}",
        monitor=config.monitor,
        mode=config.mode,
        save_top_k=config.save_top_k,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor=config.monitor, mode=config.mode, patience=10, min_delta=0.001
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    return [checkpoint_callback, early_stop_callback, lr_monitor]


def setup_logger(config, project_name, use_wandb, experiment_tag):
    if not use_wandb:
        return None
    return WandbLogger(
        project=project_name,
        name=f"drum-transformer-{config.device}-{experiment_tag}",
        log_model=True,
        save_dir=config.log_dir,
    )


def configure_run(args):
    if not args.experiment_tag and (args.resume or args.evaluate):
        logger.error("--experiment-tag is required for --resume or --evaluate.")
        sys.exit(1)
    args.experiment_tag = args.experiment_tag or datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    config = (
        auto_detect_config()
        if args.config.lower() == "auto"
        else get_config(args.config)
    )

    effective_use_wandb = not args.no_wandb and (
        args.use_wandb or getattr(config, "use_wandb", False)
    )

    if effective_use_wandb:
        wandb.init(  # type: ignore
            project=args.project_name,
            name=f"drum-transformer-{config.device}-{args.experiment_tag}",
            config=config.__dict__,
            dir=config.log_dir,
            resume="allow",
            id=args.experiment_tag,
        )

    apply_cli_overrides(config, args)
    validate_config(config)
    return config, effective_use_wandb
