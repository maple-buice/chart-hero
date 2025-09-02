"""
Main training script for transformer-based drum transcription.
This script orchestrates the training process by leveraging helper modules for
setup, data loading, and the core Lightning module.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch

import wandb
from chart_hero.model_training.lightning_module import DrumTranscriptionModule
from chart_hero.model_training.training_setup import (
    configure_run,
    setup_arg_parser,
    setup_callbacks,
    setup_logger,
)
from chart_hero.model_training.transformer_data import (
    compute_class_pos_weights,
    create_data_loaders,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting training script.")
    parser = setup_arg_parser()
    args = parser.parse_args()
    logger.info(f"Arguments parsed: {args}")

    config, use_wandb = configure_run(args)
    logger.info(
        f"Configuration loaded for '{args.config}'. Using device: {config.device}"
    )
    logger.info(f"WandB logging is {'enabled' if use_wandb else 'disabled'}.")

    try:
        # Compute/load class pos_weight if requested and not a quick test
        pos_weight = None
        if (
            not args.quick_test
            and getattr(config, "pos_weight_strategy", "auto") == "auto"
        ):
            train_split = Path(config.data_dir) / "train"
            cache_path = Path(config.model_dir) / "pos_weight.pt"

            def _load_cache() -> torch.Tensor | None:
                if not cache_path.exists():
                    return None
                try:
                    payload = torch.load(cache_path, map_location="cpu")
                    if not isinstance(payload, dict):
                        return None
                    if payload.get("data_dir") != str(Path(config.data_dir).resolve()):
                        return None
                    if not train_split.exists():
                        return None
                    # Validate basic consistency with current dataset
                    current_files = list(train_split.glob("*_label.npy"))
                    if len(current_files) != payload.get("num_files"):
                        return None
                    pw = payload.get("pos_weight")
                    if (
                        isinstance(pw, torch.Tensor)
                        and pw.numel() == config.num_drum_classes
                    ):
                        logger.info("Loaded class pos_weight from cache")
                        return pw.float()
                except Exception as e:
                    logger.warning(f"Failed to load pos_weight cache: {e}")
                return None

            def _save_cache(pw: torch.Tensor) -> None:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    num_files = 0
                    if train_split.exists():
                        num_files = len(list(train_split.glob("*_label.npy")))
                    torch.save(
                        {
                            "data_dir": str(Path(config.data_dir).resolve()),
                            "num_files": num_files,
                            "pos_weight": pw.detach().cpu(),
                        },
                        cache_path,
                    )
                    logger.info(f"Saved class pos_weight cache to {cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to save pos_weight cache: {e}")

            pos_weight = _load_cache()
            if pos_weight is None:
                if train_split.exists():
                    pos_weight = compute_class_pos_weights(
                        data_dir=config.data_dir,
                        num_classes=config.num_drum_classes,
                        split="train",
                    )
                    # Optional sampling cap (estimate on subset for huge datasets)
                    # If needed, implement sampling inside compute_class_pos_weights.
                    logger.info(f"Computed class pos_weight: {pos_weight.tolist()}")
                    _save_cache(pos_weight)
                else:
                    logger.info(
                        f"Skipping class pos_weight computation (missing split: {train_split})"
                    )

        model = DrumTranscriptionModule(config, pos_weight=pos_weight)
        # Load calibrated per-class thresholds if available (for resume/eval and immediate use)
        try:
            thr_path = Path(config.model_dir) / "class_thresholds.json"
            if thr_path.exists():
                payload = (
                    torch.load(str(thr_path)) if thr_path.suffix == ".pt" else None
                )
                if thr_path.suffix == ".json":
                    import json as _json

                    with thr_path.open("r") as f:
                        payload = _json.load(f)
                if isinstance(payload, dict) and "class_thresholds" in payload:
                    thrs = payload["class_thresholds"]
                    if (
                        isinstance(thrs, (list, tuple))
                        and len(thrs) == config.num_drum_classes
                    ):
                        config.class_thresholds = list(map(float, thrs))
                        logger.info(
                            f"Loaded calibrated thresholds from {thr_path}: {config.class_thresholds}"
                        )
        except Exception as e:
            logger.warning(f"Failed to load calibrated thresholds: {e}")
        wandb_logger = setup_logger(
            config, args.project_name, use_wandb, args.experiment_tag
        )
        callbacks = setup_callbacks(config, use_logger=use_wandb)

        trainer_kwargs: dict[str, Any] = {
            "logger": wandb_logger if use_wandb else False,
            "callbacks": callbacks,
            "max_epochs": config.num_epochs,
            "accelerator": config.device,
            "devices": 1,
            "precision": config.precision,
            "gradient_clip_val": config.gradient_clip_val,
            "accumulate_grad_batches": config.accumulate_grad_batches,
            "enable_progress_bar": not args.no_progress_bar,
            "limit_train_batches": 0.1 if args.quick_test else 1.0,
            "limit_val_batches": 0.1 if args.quick_test else 1.0,
        }

        try:
            trainer = pl.Trainer(**trainer_kwargs)
        except Exception as e:
            logger.warning(f"MPS accelerator failed to initialize: {e}")
            logger.warning("Falling back to CPU.")
            config.device = "cpu"
            trainer_kwargs["accelerator"] = "cpu"
            trainer = pl.Trainer(**trainer_kwargs)

        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            config=config, data_dir=config.data_dir, with_lengths=True
        )
        logger.info("Data loaders created successfully.")

        checkpoint_path = None
        if args.resume or args.evaluate:
            last_ckpt = Path(config.model_dir) / "last.ckpt"
            if not last_ckpt.exists():
                logger.error(f"Checkpoint not found in {config.model_dir}")
                sys.exit(1)
            checkpoint_path = str(last_ckpt)

        if args.evaluate:
            trainer.test(model, dataloaders=test_loader, ckpt_path=checkpoint_path)
        else:
            trainer.fit(
                model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=checkpoint_path,
            )
            if test_loader:
                trainer.test(model, dataloaders=test_loader, ckpt_path="best")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        if use_wandb:
            wandb.finish(exit_code=1)
        sys.exit(1)
    finally:
        if use_wandb and wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()
