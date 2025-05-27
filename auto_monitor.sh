#!/bin/bash
# auto_monitor.sh
# Script to automatically detect and monitor Python training processes
# Usage: ./auto_monitor.sh [OPTIONS]
#
# Options:
#   --watch-pattern PATTERN  Only monitor processes matching this pattern
#   --interval SECONDS       Set monitoring interval (default: 2 seconds)
#   --no-color               Disable colored output
#   --compact                Use compact display mode
#   --log FILE               Log monitoring data to file
#   --monitor-script PATH    Path to monitor_training.py (default: same directory)

# Default values
WATCH_PATTERN=""
INTERVAL=2
NO_COLOR=""
COMPACT=""
LOG_FILE=""
MONITOR_SCRIPT="./monitor_training.py"
AUTO_DETECT="--auto-detect"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --watch-pattern)
            WATCH_PATTERN="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --no-color)
            NO_COLOR="--no-color"
            shift
            ;;
        --compact)
            COMPACT="--compact"
            shift
            ;;
        --log)
            LOG_FILE="--log $2"
            shift 2
            ;;
        --monitor-script)
            MONITOR_SCRIPT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if monitor script exists
if [[ ! -f "$MONITOR_SCRIPT" ]]; then
    echo "Error: Monitoring script not found at $MONITOR_SCRIPT"
    echo "Please specify the correct path with --monitor-script"
    exit 1
fi

# Function to check if the monitor is already running
is_monitor_running() {
    pgrep -f "python.*monitor_training.py" > /dev/null
    return $?
}

# Function to detect training processes
detect_training_process() {
    local pattern=$1
    local training_pids=()
    
    # Find Python processes that look like they're training models
    if [[ -n "$pattern" ]]; then
        # If pattern provided, use it
        training_pids=($(pgrep -f "python.*$pattern"))
    else
        # Otherwise look for common ML keywords
        for keyword in "train" "learning" "tensorflow" "pytorch" "torch" "model" "deep" "neural" "cuda" "transformer" "keras" "lightning" "ai" "ml" "gpu" "mps" "huggingface" "sklearn"; do
            pids=($(pgrep -f "python.*$keyword"))
            if [[ ${#pids[@]} -gt 0 ]]; then
                training_pids+=("${pids[@]}")
            fi
        done
    fi
    
    # Remove duplicates
    training_pids=($(echo "${training_pids[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))
    
    # Filter out this script and the monitor itself
    local filtered_pids=()
    for pid in "${training_pids[@]}"; do
        if ! ps -p "$pid" -o command= | grep -q "monitor_training\|auto_monitor"; then
            filtered_pids+=("$pid")
        fi
    done
    
    echo "${filtered_pids[@]}"
}

# Main loop to detect and monitor training processes
echo "Starting auto-monitoring for training processes..."
echo "Press Ctrl+C to exit"

# Handle cleanup on exit
cleanup() {
    echo "Cleaning up and exiting..."
    # Kill any monitor processes we started
    if [[ -n "$MONITOR_PID" ]]; then
        kill $MONITOR_PID 2>/dev/null || true
    fi
    exit 0
}

# Set up trap to catch signals
trap cleanup SIGINT SIGTERM EXIT

# Create logs directory if it doesn't exist
mkdir -p logs

while true; do
    # Check if monitoring is already running
    if is_monitor_running; then
        echo "Monitoring already running, waiting..."
        sleep 5
        continue
    fi
    
    # Get training PIDs
    training_pids=($(detect_training_process "$WATCH_PATTERN"))
    
    if [[ ${#training_pids[@]} -gt 0 ]]; then
        echo "Detected training process with PID: ${training_pids[0]}"
        
        # Generate timestamp for log file if logging is enabled but no file specified
        if [[ -z "$LOG_FILE" ]]; then
            TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
            LOG_FILE="--log logs/auto_monitoring_${TIMESTAMP}.csv"
        fi
        
        # Build the monitoring command
        monitor_cmd="python3 $MONITOR_SCRIPT --interval $INTERVAL $NO_COLOR $COMPACT $LOG_FILE $AUTO_DETECT --watch-pid ${training_pids[0]}"
        
        echo "Starting monitoring with command: $monitor_cmd"
        
        # On macOS, we can open a new terminal window
        if [[ "$(uname)" == "Darwin" ]]; then
            osascript -e "tell application \"Terminal\" to do script \"cd '$PWD' && $monitor_cmd\""
            # Wait a bit to make sure the new terminal window opens
            sleep 2
        else
            # On other systems, run in the background
            eval "$monitor_cmd" &
            MONITOR_PID=$!
        fi
        
        echo "Monitoring started. Waiting for process to complete..."
        
        # Wait until the process no longer exists
        while ps -p ${training_pids[0]} >/dev/null 2>&1; do
            sleep 2
        done
        
        echo "Training process ${training_pids[0]} has ended."
        sleep 2
    fi
    
    sleep 1
done
