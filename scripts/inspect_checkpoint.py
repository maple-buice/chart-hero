#!/usr/bin/env python3
"""
Inspect a model checkpoint to see what hyperparameters are stored.

Usage:
    python scripts/inspect_checkpoint.py models/local_transformer_models/20250908_203931/last.ckpt
"""

import argparse
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Inspect model checkpoint hyperparameters"
    )
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--key", help="Show specific hyperparameter key")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # Get hyperparameters
    hparams = ckpt.get("hyper_parameters") or ckpt.get("hparams")
    if not hparams:
        print("No hyperparameters found in checkpoint")
        return

    if args.key:
        # Show specific key
        if args.key in hparams:
            print(f"{args.key}: {hparams[args.key]}")
        else:
            print(f"Key '{args.key}' not found")
            print(f"Available keys: {list(hparams.keys())}")
    else:
        # Show all critical hyperparameters
        critical_params = [
            "sample_rate",
            "n_mels",
            "n_fft",
            "hop_length",
            "max_audio_length",
            "patch_size",
            "patch_stride",
            "prediction_threshold",
            "num_drum_classes",
            "learning_rate",
            "use_focal_loss",
            "pos_weight_cap",
        ]

        print("Critical hyperparameters:")
        for param in critical_params:
            if param in hparams:
                print(f"  {param}: {hparams[param]}")
            else:
                print(f"  {param}: [NOT FOUND]")

        print(f"\nTotal hyperparameters: {len(hparams)}")
        if len(hparams) < 20:
            print("\nAll hyperparameters:")
            for k, v in sorted(hparams.items()):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
