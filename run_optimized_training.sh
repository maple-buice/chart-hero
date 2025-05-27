#!/bin/zsh
# run_optimized_training.sh
# This script runs the model training with optimized settings for MPS (Apple Silicon)

# Stop on errors
set -e

# Clear any existing cached models from PyTorch
echo "Clearing PyTorch cache..."
python -c "import torch; torch.mps.empty_cache() if torch.backends.mps.is_available() else print('MPS not available')"

# Kill any stray Python processes that might be using GPU memory
echo "Killing any stray Python processes..."
pkill -f python3 || true

# Alternative to sudo purge - use memory trimming without sudo
echo "Freeing unused memory..."
python -c "import gc; gc.collect()"
sleep 1

# Apply optimized MPS environment variables
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPS_USE_DETERMINISTIC_ALGORITHM=0  # Disable for better performance
export MPS_CACHED_ALLOCATOR_RELEASE_THREADED=1  # Better memory management

# Apple Silicon specific optimizations
export MPS_CACHEABLE_BUFFERS=1  # Improves GPU memory management
export MPS_PRIMARY_FWD_BUFFER_ALLOCATION=1  # Improves memory allocation
export MPS_DISPATCH_THREADS=8  # Controls number of threads for dispatch
export MPS_MAX_BUFFER_SIZE=536870912  # 512MB max buffer to avoid memory spikes

# Display environment info
echo "======= Environment Information ======="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"

# Create a timestamp for the log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"
MONITOR_LOG="logs/monitor_${TIMESTAMP}.csv"

# Make sure logs directory exists
mkdir -p logs

echo "\n======= Running Training with GPU Monitoring ======="
# Launch training process and immediately get its PID
python model_training/train_transformer.py --config local --data-dir datasets/processed --audio-dir datasets/e-gmd-v1.0.0 --monitor-gpu --debug > >(tee $LOG_FILE) &
TRAINING_PID=$!

# Give the training process a moment to initialize
sleep 2

# Start monitoring with explicit PID targeting
echo "Starting resource monitoring for training process PID: $TRAINING_PID..."
python monitor_training.py --watch-pid $TRAINING_PID --interval 1 --compact --log $MONITOR_LOG &
MONITOR_PID=$!

# Trap to ensure monitoring process is killed when script exits
trap "kill $MONITOR_PID 2>/dev/null || true" EXIT

# Wait for the training process to complete
echo "Monitoring active - waiting for training to complete (PID: $TRAINING_PID)..."
wait $TRAINING_PID

echo "\nTraining complete. Logs saved to $LOG_FILE and $MONITOR_LOG"
