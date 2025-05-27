# GPU Utilization Analysis System

This suite of tools allows for structured analysis of GPU utilization during model training with different configurations. It helps identify optimal batch sizes and model architectures for maximizing GPU performance.

## Components

1. **gpu_utilization_analysis.py**: Main script for running multiple test scenarios with different configurations
2. **run_timed_training.sh**: Enhanced to accept batch size and model parameters
3. **analyze_monitoring_logs.py**: Updated with GPU efficiency metrics

## Usage

### Running a Full Analysis

To run a complete analysis across multiple configurations:

```bash
python gpu_utilization_analysis.py --output-dir gpu_analysis_results
```

This will:
1. Run multiple training scenarios with different batch sizes and model sizes
2. Collect GPU utilization data for each scenario
3. Generate comparative visualizations and recommendations

### Running a Single Scenario

To test a specific configuration:

```bash
python gpu_utilization_analysis.py --run-single small_batch --output-dir single_test
```

### Custom Test Scenarios

You can define custom test scenarios in a JSON file:

```bash
python gpu_utilization_analysis.py --scenarios my_scenarios.json --output-dir custom_tests
```

Example scenarios JSON format:
```json
[
  {"name": "custom_batch_4", "batch_size": 4, "hidden_size": 256, "time_mins": 5},
  {"name": "custom_batch_16", "batch_size": 16, "hidden_size": 384, "time_mins": 5}
]
```

## Analysis Outputs

The analysis generates several outputs:

1. **GPU Utilization Charts**: Shows GPU usage over time for each scenario
2. **Comparative Visualizations**: Compares GPU utilization across different batch sizes and model sizes
3. **Efficiency Metrics**: 
   - GPU Utilization Percentage
   - GPU/CPU Utilization Ratio
   - GPU Stability Score
4. **Recommendations**: Suggests optimal configurations based on the analysis

## Understanding the Results

- **Higher average GPU utilization** (>70%) indicates efficient use of GPU resources
- **GPU/CPU ratio > 1.0** indicates the workload is effectively utilizing the GPU relative to CPU
- **GPU stability score** measures how consistent the GPU utilization is (higher is better)

## Troubleshooting

If you encounter issues:

1. **Memory errors**: Try reducing batch size or model size
2. **Low GPU utilization**: Increase batch size or model complexity
3. **Process detection issues**: Use `--watch-pid` explicitly in monitor_training.py

## Example Analysis Workflow

1. Clear system resources: `python -c "import torch; torch.mps.empty_cache()"`
2. Run a full analysis: `python gpu_utilization_analysis.py`
3. Review results in the output directory
4. Apply the recommended configuration to your training script
