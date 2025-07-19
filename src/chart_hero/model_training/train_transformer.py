"""
Main training script for transformer-based drum transcription.
This script orchestrates the training process by leveraging helper modules for
setup, data loading, and the core Lightning module.
"""

import logging
import sys
from pathlib import Path

import pytorch_lightning as pl

import wandb
from chart_hero.model_training.lightning_module import DrumTranscriptionModule
from chart_hero.model_training.training_setup import (
    configure_run,
    setup_arg_parser,
    setup_callbacks,
    setup_logger,
)
from chart_hero.model_training.transformer_data import create_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    config, use_wandb = configure_run(args)

    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            config=config, data_dir=config.data_dir
        )

        checkpoint_path = None
        if args.resume or args.evaluate:
            last_ckpt = Path(config.model_dir) / "last.ckpt"
            if not last_ckpt.exists():
                logger.error(f"Checkpoint not found in {config.model_dir}")
                sys.exit(1)
            checkpoint_path = str(last_ckpt)

        model = DrumTranscriptionModule(config)
        wandb_logger = setup_logger(
            config, args.project_name, use_wandb, args.experiment_tag
        )
        callbacks = setup_callbacks(config)

        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=callbacks,
            max_epochs=config.num_epochs,
            accelerator=config.device,
            devices=1,
            precision=config.precision,
            gradient_clip_val=config.gradient_clip_val,
            accumulate_grad_batches=config.accumulate_grad_batches,
            enable_progress_bar=not args.no_progress_bar,
            limit_train_batches=0.1 if args.quick_test else 1.0,
            limit_val_batches=0.1 if args.quick_test else 1.0,
        )

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
