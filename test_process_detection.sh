#!/bin/zsh
# test_process_detection.sh
# Script to test if the optimized training script correctly monitors processes

# Stop on errors
set -e

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_LOG="logs/test_monitoring_${TIMESTAMP}.log"
MONITOR_LOG="logs/test_monitoring_${TIMESTAMP}.csv"

echo "=== Testing Process Detection and Monitoring ==="
echo "1. Running a test training process"
echo "2. Monitoring it with direct PID targeting"

# First run the test_monitoring.py script in the background
echo "Starting test training process..."
python test_monitoring.py > $TEST_LOG &
TEST_PID=$!

# Give it a moment to initialize
sleep 2

echo "Test process started with PID: $TEST_PID"

# Show the process in our diagnostic tool
echo "Running process detection diagnostic..."
python list_python_processes.py

# Start monitoring with explicit PID targeting
echo "Starting monitoring with direct PID targeting..."
python monitor_training.py --watch-pid $TEST_PID --interval 1 --compact --log $MONITOR_LOG &
MONITOR_PID=$!

# Display PIDs for verification
echo "\nProcesses:"
echo "- Test process: $TEST_PID"
echo "- Monitor process: $MONITOR_PID"

# Wait for a short time to collect data
echo "\nRunning monitoring for 10 seconds..."
sleep 10

# Kill the processes
echo "Test complete. Stopping processes..."
kill $MONITOR_PID 2>/dev/null || true
kill $TEST_PID 2>/dev/null || true

echo "\nTest completed. Log files:"
echo "- Test log: $TEST_LOG"
echo "- Monitor log: $MONITOR_LOG"
echo "\nVerify that the monitor successfully tracked the test process."
