#!/usr/bin/env python3
"""
Convenience launcher for high-resolution training.

Usage examples:
  python scripts/train_highres.py --data-dir datasets/processed_highres \
    --model-dir models/highres --log-dir logs/highres --num-epochs 100

This forwards all arguments to the standard trainer with --config local_highres.
"""

from __future__ import annotations

import os
import subprocess
import sys
from shutil import which


def main() -> None:
    py = sys.executable or which("python") or "python3"
    # If caller doesn't specify --data-dir, default to the high-res dataset path
    passed = " ".join(sys.argv[1:])
    add_data = "--data-dir" not in passed
    args = (
        [
            py,
            "-m",
            "chart_hero.model_training.train_transformer",
            "--config",
            "local_highres",
        ]
        + (["--data-dir", "datasets/processed_highres"] if add_data else [])
        + sys.argv[1:]
    )
    # Run and forward exit code
    res = subprocess.run(args)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
