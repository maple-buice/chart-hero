#!/usr/bin/env python3
"""
Convenience launcher for high-resolution training.

Usage examples:
  python scripts/train_highres.py --data-dir datasets/processed_highres \
    --model-dir models/highres --log-dir logs/highres --num-epochs 100

This forwards all arguments to the standard trainer with --config local_highres.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from shutil import which


def main() -> None:
    py = sys.executable or which("python") or "python3"

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir")
    parsed, remaining = parser.parse_known_args()

    args = [
        py,
        "-m",
        "chart_hero.model_training.train_transformer",
        "--config",
        "local_highres",
    ]
    if parsed.data_dir is None:
        args += ["--data-dir", "datasets/processed_highres"]
    args += remaining
    # Run and forward exit code
    res = subprocess.run(args)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
