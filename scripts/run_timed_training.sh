#!/bin/zsh
# run_timed_training.sh
# This script runs the model training with a time limit for monitoring resource usage

# Stop on errors
set -e

# Get parameters from arguments or use defaults
TIME_LIMIT_MINS=${1:-5}
BATCH_SIZE=${2:-8}
HIDDEN_SIZE=${3:-384}
OUTPUT_DIR=${4:-"logs"}
TIME_LIMIT_SECS=$((TIME_LIMIT_MINS * 60))

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

# Set up output directory - either use provided or create in logs
if [[ "$OUTPUT_DIR" == "logs" ]]; then
  mkdir -p logs
  LOG_FILE="logs/training_${TIMESTAMP}.log"
  MONITOR_LOG="logs/monitor_${TIMESTAMP}.csv"
else
  mkdir -p "$OUTPUT_DIR"
  LOG_FILE="${OUTPUT_DIR}/training_${TIMESTAMP}.log"
  MONITOR_LOG="${OUTPUT_DIR}/monitor_${TIMESTAMP}.csv"
fi

echo "\n======= Running Training with GPU Monitoring ======="
echo "Duration: ${TIME_LIMIT_MINS} minutes"
echo "Batch Size: ${BATCH_SIZE}"
echo "Hidden Size: ${HIDDEN_SIZE}"
echo "Output Directory: ${OUTPUT_DIR}"

# Launch training process with custom parameters and immediately get its PID
python model_training/train_transformer.py \
  --config local \
  --data-dir datasets/processed \
  --audio-dir datasets/e-gmd-v1.0.0 \
  --monitor-gpu \
  --batch-size ${BATCH_SIZE} \
  --hidden-size ${HIDDEN_SIZE} \
  --debug > >(tee $LOG_FILE) &
TRAINING_PID=$!

# Give the training process a moment to initialize
sleep 2

# Start monitoring with explicit PID targeting
echo "Starting resource monitoring for training process PID: $TRAINING_PID..."
python monitor_training.py --watch-pid $TRAINING_PID --interval 1 --compact --log $MONITOR_LOG &
MONITOR_PID=$!

# Trap to ensure monitoring process is killed when script exits
trap "kill $TRAINING_PID $MONITOR_PID 2>/dev/null || true" EXIT

echo "Training and monitoring will run for ${TIME_LIMIT_MINS} minutes..."

# Wait for the specified time limit
sleep $TIME_LIMIT_SECS

echo "\nTime limit reached (${TIME_LIMIT_MINS} minutes). Stopping training and monitoring processes..."

# Kill the training and monitoring processes
kill $TRAINING_PID $MONITOR_PID 2>/dev/null || true

echo "\nTraining stopped. Logs saved to $LOG_FILE and $MONITOR_LOG"
echo "Running analysis on the collected monitoring data..."

# Run analysis on the collected data
python analyze_monitoring_logs.py "$MONITOR_LOG" --output "${OUTPUT_DIR}/analysis_${TIMESTAMP}"

# Save configuration parameters for reference
CONFIG_JSON="${OUTPUT_DIR}/config_${TIMESTAMP}.json"
echo "{" > $CONFIG_JSON
echo "  \"duration_mins\": $TIME_LIMIT_MINS," >> $CONFIG_JSON
echo "  \"batch_size\": $BATCH_SIZE," >> $CONFIG_JSON
echo "  \"hidden_size\": $HIDDEN_SIZE," >> $CONFIG_JSON
echo "  \"timestamp\": \"$TIMESTAMP\"," >> $CONFIG_JSON
echo "  \"training_log\": \"$LOG_FILE\"," >> $CONFIG_JSON
echo "  \"monitor_log\": \"$MONITOR_LOG\"" >> $CONFIG_JSON
echo "}" >> $CONFIG_JSON

echo "\nAnalysis complete. You can find the results in the ${OUTPUT_DIR}/analysis_${TIMESTAMP} directory."
echo "Configuration saved to $CONFIG_JSON"
