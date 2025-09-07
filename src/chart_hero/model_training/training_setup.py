"""
This module contains functions for setting up the training environment,
including argument parsing, configuration loading, and callback/logger setup.
"""

import argparse
import os
import logging
import sys
from datetime import datetime
from pathlib import Path

from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

import wandb
from chart_hero.model_training.transformer_config import (
    BaseConfig,
    auto_detect_config,
    get_config,
    validate_config,
)

logger = logging.getLogger(__name__)


def _load_env_local(path: Path) -> None:
    """Load simple KEY=VALUE lines from a .env.local file into os.environ if not already set."""
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.strip().startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        # Non-fatal: env loading should not interrupt training
        pass


def setup_arg_parser() -> argparse.ArgumentParser:
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
            "local_highres",
            "local_micro",
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
    parser.add_argument(
        "--log-media",
        action="store_true",
        help="Log qualitative media (e.g., histograms/heatmaps) to W&B",
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
    parser.add_argument(
        "--dataset-fraction",
        type=float,
        help="Use a fraction of files per split (0 < f <= 1.0)",
    )
    parser.add_argument(
        "--max-files-per-split",
        type=int,
        help="Hard cap on number of files per split",
    )
    parser.add_argument(
        "--max-audio-length",
        type=float,
        help="Override max audio segment length in seconds for training windows",
    )
    return parser


def configure_paths(config: "BaseConfig", args: argparse.Namespace) -> None:
    if args.data_dir:
        config.data_dir = str(Path(args.data_dir).resolve())
    if args.audio_dir:
        config.audio_dir = str(Path(args.audio_dir).resolve())

    if not hasattr(config, "model_dir") or not config.model_dir:
        model_path = Path("models") / args.experiment_tag
        config.model_dir = str(model_path)
    else:
        model_path = Path(config.model_dir) / args.experiment_tag
        config.model_dir = str(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    if not hasattr(config, "log_dir") or not config.log_dir:
        log_path = Path("logs")
        config.log_dir = str(log_path)
    else:
        log_path = Path(config.log_dir)
        config.log_dir = str(log_path)
    log_path.mkdir(parents=True, exist_ok=True)


def apply_cli_overrides(config: "BaseConfig", args: argparse.Namespace) -> None:
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
        config.num_epochs = 1
    if args.dataset_fraction is not None:
        config.dataset_fraction = float(args.dataset_fraction)
    if args.max_files_per_split is not None:
        config.max_files_per_split = int(args.max_files_per_split)
    if getattr(args, "log_media", False):
        config.enable_media_logging = True
    if getattr(args, "max_audio_length", None) is not None:
        try:
            config.max_audio_length = float(args.max_audio_length)
        except Exception:
            pass


def setup_callbacks(config: "BaseConfig", use_logger: bool = True) -> list[Callback]:
    # Build filename pattern for best checkpoints based on the monitored metric
    filename_pattern = f"drum-transformer-{{epoch:02d}}-{{{config.monitor}:.3f}}"
    # If monitoring a validation metric (prefix 'val_'), evaluate/save after validation
    # Otherwise, evaluate/save at train epoch end (supports no-val runs)
    is_val_metric = isinstance(config.monitor, str) and config.monitor.startswith(
        "val_"
    )

    # Best-k checkpoint callback (monitored)
    best_ckpt_cb = ModelCheckpoint(
        dirpath=str(config.model_dir),
        filename=filename_pattern,
        monitor=config.monitor,
        mode=config.mode,
        verbose=True,
        save_top_k=config.save_top_k,
        save_last=False,  # handled by a dedicated 'last' callback below
        save_on_train_epoch_end=not is_val_metric,
    )

    # Always-save last checkpoint every epoch regardless of metric value/improvement
    # This guarantees a recoverable state even if monitored metric is NaN/missing.
    last_ckpt_cb = ModelCheckpoint(
        dirpath=str(config.model_dir),
        filename="last",
        monitor=None,
        verbose=True,
        save_top_k=0,
        save_last=True,
        save_on_train_epoch_end=True,
        every_n_epochs=1,
    )

    logger.info(
        "Checkpointing configured: best(dir=%s, filepat=%s, monitor=%s, mode=%s, top_k=%s); last(every epoch)=True",
        best_ckpt_cb.dirpath,
        best_ckpt_cb.filename,
        best_ckpt_cb.monitor,
        best_ckpt_cb.mode,
        best_ckpt_cb.save_top_k,
    )

    # Early stopping aligned with where the monitored metric is produced
    early_stop_callback = EarlyStopping(
        monitor=config.monitor,
        mode=config.mode,
        patience=10,
        min_delta=0.001,
        check_on_train_epoch_end=not is_val_metric,
    )

    callbacks = [best_ckpt_cb, last_ckpt_cb, early_stop_callback]
    if use_logger:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    return callbacks


def setup_logger(
    config: "BaseConfig", project_name: str, use_wandb: bool, experiment_tag: str
) -> WandbLogger | None:
    if not use_wandb:
        return None
    return WandbLogger(
        project=project_name,
        name=f"drum-transformer-{config.device}-{experiment_tag}",
        save_dir=config.log_dir,
        log_model="all",
        group=experiment_tag,
        tags=[
            str(getattr(config, "device", "cpu")),
            str(getattr(config, "model_name", "transformer")),
            str(getattr(config, "monitor", "val_f1")),
        ],
        job_type="train",
    )


def configure_run(args: argparse.Namespace) -> tuple[BaseConfig, bool]:
    # Load environment variables from .env.local if present (e.g., WANDB_API_KEY)
    _load_env_local(Path(".env.local"))
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

    # Let PyTorch Lightning's WandbLogger manage wandb runs to avoid duplicate/nested runs.

    apply_cli_overrides(config, args)
    validate_config(config)
    return config, effective_use_wandb
