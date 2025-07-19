import argparse
import itertools
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Setup logging for the experimentation script
log_file_path = (
    Path(__file__).parent.parent
    / "logs"
    / f"experiment_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
log_file_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Define hyperparameter ranges
PARAM_RANGES = {
    "batch_size": [2, 4, 8],
    "hidden_size": [256, 384, 512],
    "learning_rate": [1e-5, 5e-5, 1e-4],
    "num_workers": [0, 1, 2],
    "accumulate_grad_batches": [1, 2, 4],
}

# Path to the training script
TRAIN_SCRIPT_PATH = Path(__file__).parent / "train_transformer.py"
# Base directory for chart-hero (assuming run_experiments.py is in model_training)
CHART_HERO_BASE_DIR = Path(__file__).parent.parent


def run_experiment(params: dict[str, Any], use_wandb: bool, quick_test: bool, monitor_gpu: bool):
    """
    Runs a single training experiment with the given parameters.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_tag_parts = [
        f"{key.replace('_', '')[:2]}{value}" for key, value in params.items()
    ]
    experiment_tag = "_".join(experiment_tag_parts) + f"_{timestamp}"

    cmd = [
        "python3",
        str(TRAIN_SCRIPT_PATH),
        "--config",
        "local",
        "--experiment-tag",
        experiment_tag,
    ]

    # Pass --quick-test to train_transformer.py only if the quick_test flag for this function is True.
    if quick_test:
        cmd.append("--quick-test")

    if use_wandb:
        cmd.append("--use-wandb")
    if monitor_gpu:
        cmd.append("--monitor-gpu")

    for key, value in params.items():
        # Convert snake_case keys to kebab-case for CLI arguments
        cli_key = key.replace("_", "-")
        cmd.extend([f"--{cli_key}", str(value)])

    logger.info(f"Running experiment: {experiment_tag}")
    logger.info(f"Command: {' '.join(cmd)}")

    # GPU log path will be based on experiment_tag in train_transformer.py
    # Example: logs/gpu_monitoring_bs4_hs512_lr0.0001_nw2_agb1_20230101_120000.csv
    expected_gpu_log_name = f"gpu_monitoring_{experiment_tag}.csv"
    expected_gpu_log_path = CHART_HERO_BASE_DIR / "logs" / expected_gpu_log_name

    # Training log path will also be based on experiment_tag in train_transformer.py
    expected_train_log_name = f"training_{experiment_tag}.log"
    expected_train_log_path = CHART_HERO_BASE_DIR / "logs" / expected_train_log_name

    start_time = time.time()
    try:
        process = subprocess.run(
            cmd, capture_output=True, text=True, check=False, cwd=CHART_HERO_BASE_DIR
        )
        end_time = time.time()
        duration = end_time - start_time

        if process.returncode == 0:
            logger.info(
                f"Experiment {experiment_tag} completed successfully in {duration:.2f} seconds."
            )
            logger.info(
                f"  STDOUT:\n{process.stdout[-500:]}"
            )  # Log last 500 chars of stdout
            if process.stderr:
                logger.info(f"  STDERR:\n{process.stderr[-500:]}")
            status = "success"
        else:
            logger.error(
                f"Experiment {experiment_tag} failed with return code {process.returncode} after {duration:.2f} seconds."
            )
            logger.error(f"  STDOUT:\n{process.stdout}")
            logger.error(f"  STDERR:\n{process.stderr}")
            status = "failed"

        return {
            "params": params,
            "tag": experiment_tag,
            "status": status,
            "duration_seconds": duration,
            "return_code": process.returncode,
            "gpu_log": str(expected_gpu_log_path) if monitor_gpu else None,
            "train_log": str(
                expected_train_log_path
            ),  # train_transformer.py should create this
        }

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logger.exception(
            f"Exception during experiment {experiment_tag} after {duration:.2f} seconds: {e}"
        )
        return {
            "params": params,
            "tag": experiment_tag,
            "status": "exception",
            "duration_seconds": duration,
            "error_message": str(e),
            "gpu_log": str(expected_gpu_log_path) if monitor_gpu else None,
            "train_log": str(expected_train_log_path),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning experiments for drum transcription."
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Enable WandB logging."
    )
    # Ensure the quick_test argument for run_experiments.py itself is correctly defined.
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run only the first experiment combination for a quick test of the runner.",
    )
    parser.add_argument(
        "--monitor-gpu",
        action="store_true",
        help="Enable GPU monitoring (if applicable).",
    )

    args = parser.parse_args()

    logger.info("Starting hyperparameter tuning session.")
    logger.info(f"Quick test mode: {'Enabled' if args.quick_test else 'Disabled'}")
    logger.info(f"W&B logging: {'Enabled' if args.use_wandb else 'Disabled'}")
    logger.info(f"GPU monitoring: {'Enabled' if args.monitor_gpu else 'Disabled'}")

    param_names = list(PARAM_RANGES.keys())
    param_values = [PARAM_RANGES[name] for name in param_names]

    all_combinations = list(itertools.product(*param_values))  # type: ignore
    # Limit to 1 experiment if quick_test is enabled, for faster iteration
    if args.quick_test:
        logger.info(
            "Quick test mode is enabled for run_experiments.py, running only the first experiment combination."
        )
        all_combinations = all_combinations[:1]

    total_experiments = len(all_combinations)
    logger.info(f"Total experiments to run: {total_experiments}")

    results_summary = []

    # Determine if running a quick test for the entire suite
    # This is controlled by the --quick-test flag for run_experiments.py
    run_only_first_combination = args.quick_test

    # The actual hyperparameter combinations to iterate through
    # If run_only_first_combination is True, these loops will effectively run once.

    experiment_count = 0
    for combination_idx, combo_values in enumerate(all_combinations):
        params = dict(zip(param_names, combo_values))

        experiment_count += 1
        logger.info(
            f"--- Starting Experiment {experiment_count}/{total_experiments} ---"
        )

        # The quick_test flag for run_experiment (and thus train_transformer.py) is now always True
        # to ensure individual experiments are short.
        # args.quick_test (now run_only_first_combination) controls if we run 1 or all combinations.
        # Pass args.quick_test (from run_experiments.py CLI) to the quick_test parameter of run_experiment.
        # This determines if train_transformer.py itself runs in quick_test mode.
        result = run_experiment(
            params, args.use_wandb, args.quick_test, args.monitor_gpu
        )
        results_summary.append(result)
        logger.info(
            f"--- Finished Experiment {experiment_count}/{total_experiments} ---"
        )

        if run_only_first_combination:
            logger.info(
                "Exiting after the first experiment due to --quick-test flag for run_experiments.py."
            )
            break

    logger.info("All experiments completed.")
    logger.info("--- Experiment Summary ---")
    for res in results_summary:
        logger.info(
            f"Tag: {res['tag']}, Status: {res['status']}, Duration: {res['duration_seconds']:.2f}s, "
            f"Params: {res['params']}"
        )
        if res["status"] != "success":
            logger.info(
                f"  Error (if any): {res.get('error_message', 'See logs for details')}"
            )
        if res.get("gpu_log"):
            logger.info(f"  GPU Log: {res['gpu_log']}")
        if res.get("train_log"):
            logger.info(f"  Train Log: {res['train_log']}")

    # You can save results_summary to a CSV or JSON file here for further analysis
    summary_file_path = (
        Path(__file__).parent.parent
        / "logs"
        / f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    import json

    with open(summary_file_path, "w") as f:
        json.dump(results_summary, f, indent=4)
    logger.info(f"Experiment summary saved to: {summary_file_path}")


if __name__ == "__main__":
    main()
