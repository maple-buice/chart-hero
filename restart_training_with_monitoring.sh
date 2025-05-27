#!/bin/zsh
# restart_training_with_monitoring.sh
# This script kills existing processes, restarts training with new optimizations,
# and launches monitoring tools
# Usage: ./restart_training_with_monitoring.sh [optional: specific_training_script]

# Stop on errors
set -e

echo "==== Stopping existing processes ===="
# Kill any existing training or monitoring processes
pkill -f "train.*\.py" || true
pkill -f monitor_training.py || true
pkill -f monitor_mps.py || true
pkill -f auto_monitor.sh || true

# Give processes time to shut down
sleep 2

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MONITORING_LOG="logs/monitoring_${TIMESTAMP}.csv"

echo "==== Starting enhanced resource monitor ===="
# Check if monitor_training.py exists
if [ ! -f "monitor_training.py" ]; then
    echo "Error: monitor_training.py not found in the current directory"
    exit 1
fi

# Start the resource monitor with auto-detection in a new terminal
osascript -e 'tell application "Terminal" to do script "cd '$PWD' && python3 monitor_training.py --auto-detect --interval 1 --log '$MONITORING_LOG'"'

# Determine which training script to run
TRAINING_SCRIPT="${1:-./run_optimized_training.sh}"

echo "==== Starting training: $TRAINING_SCRIPT ===="
# Run the specified training script or the default optimized training script
if [[ "$TRAINING_SCRIPT" == ./run_optimized_training.sh ]]; then
    # Run the optimized training script
    ./run_optimized_training.sh
else
    # Run the specified script
    python3 "$TRAINING_SCRIPT"
fi

echo "==== Training complete ===="
echo "Monitoring log saved to: $MONITORING_LOG"

# Generate performance analysis if the script exists
if [ -f "analyze_monitoring_logs.py" ]; then
    echo "==== Generating performance analysis ===="
    python3 analyze_monitoring_logs.py "$MONITORING_LOG"
fi
