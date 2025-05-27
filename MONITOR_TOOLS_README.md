# Resource Monitoring Tools for ML Training

This set of tools helps you monitor system resources during machine learning training, with special features for macOS and PyTorch MPS devices.

## Main Features

- System resource monitoring (CPU, memory, disk I/O, network)
- Enhanced PyTorch MPS GPU memory monitoring on Apple Silicon Macs
- Automatic detection of ML training processes
- Visual representation with ASCII bar charts and color-coded metrics
- Trend indicators to see resource usage patterns (↑, ↓, →)
- Customizable visualization options
- CSV logging for later analysis
- Helper scripts for automatic monitoring and training restarts

## Tools Overview

1. **`monitor_training.py`** - Main monitoring script
2. **`auto_monitor.sh`** - Automatically detects and monitors ML training processes
3. **`restart_training_with_monitoring.sh`** - Restarts training with monitoring
4. **`test_monitoring.py`** - Test script to verify monitoring functionality

## Basic Usage

### Direct Monitoring

```bash
python3 monitor_training.py --auto-detect
```

This will automatically detect and monitor any Python processes running ML frameworks or training-related code.

### Using Auto-Monitor Script

```bash
./auto_monitor.sh
```

This script will launch a terminal window running the monitoring tool with process auto-detection.

### Restarting Training with Monitoring

```bash
./restart_training_with_monitoring.sh [your_training_script.py]
```

This kills any existing training processes, starts monitoring in a new terminal, and launches your training script.

## Configuration Options

### monitor_training.py

| Option | Description |
|--------|-------------|
| `--interval` | Update interval in seconds (default: 2) |
| `--log` | Log data to a specified CSV file |
| `--processes` | Maximum number of processes to show |
| `--save-config` | Save current settings as default configuration |
| `--auto-detect` | Auto-detect training processes |
| `--enable-debug` | Enable debug output |
| `--watch-pid` | Watch a specific process ID |
| `--no-color` | Disable colored output |
| `--compact` | Use compact display format |
| `--viz-width` | Width of visualization bars |
| `--no-process-scan` | Disable Python process scanning |

### auto_monitor.sh

| Option | Description |
|--------|-------------|
| `--watch-pattern` | Only monitor processes matching this pattern |
| `--interval` | Set monitoring interval |
| `--no-color` | Disable colored output |
| `--compact` | Use compact display mode |
| `--log` | Log monitoring data to file |
| `--monitor-script` | Path to monitor_training.py |

## PyTorch MPS GPU Monitoring

The tool provides enhanced monitoring for PyTorch MPS devices on Apple Silicon Macs:

- Automatic detection of MPS device
- Memory allocation tracking
- Tensor counting and memory usage
- Device-specific memory estimates based on M1/M2 models
- Error handling for MPS initialization

## Log File Analysis

Logs are saved in CSV format for easy analysis. The logs include:

- Timestamp for each measurement
- System-wide metrics (CPU, memory, GPU, I/O)
- Per-process metrics (CPU, memory usage)
- Command line information

Example of analyzing logs:

```bash
# Using pandas in Python
import pandas as pd
df = pd.read_csv('your_log_file.csv')
system_metrics = df[df['type'] == 'system']
process_metrics = df[df['type'] == 'process']

# Plot CPU usage over time
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(system_metrics['timestamp'], system_metrics['cpu_percent'])
plt.title('CPU Usage Over Time')
plt.xlabel('Time')
plt.ylabel('CPU %')
plt.grid()
plt.savefig('cpu_usage.png')
```

## Tips for Effective Monitoring

1. **Use `--auto-detect` for automatic process detection**
   The tool will identify Python processes running common ML frameworks.

2. **Enable logging for post-training analysis**
   Use the `--log` option to save data for later analysis.

3. **Use compact mode on smaller screens**
   The `--compact` option provides a condensed view.

4. **Adjust visualization width**
   Use `--viz-width` to change the size of bar charts.

5. **Save your preferred settings**
   Use `--save-config` to store your preferred configuration.

## Troubleshooting

### Process Not Detected

If your training process is not automatically detected:
- Use `--enable-debug` to see process scanning details
- Specify the PID directly with `--watch-pid`

### MPS Memory Issues

If you encounter MPS memory reporting issues:
- Ensure PyTorch is properly installed with MPS support
- Try running with `--enable-debug` for more details
- Check if your version of PyTorch supports the MPS backend

### Script Crashes

If the monitoring script crashes:
- Update to the latest version of psutil (`pip install --upgrade psutil`)
- Check if you have appropriate permissions
- Try running with fewer features enabled
