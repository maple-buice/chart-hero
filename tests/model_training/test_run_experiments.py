from unittest.mock import MagicMock, patch

from chart_hero.model_training.run_experiments import run_experiment


@patch("subprocess.run")
def test_run_experiment(mock_subprocess_run):
    """Test the run_experiment function."""
    # Mock the subprocess.run call to return a successful result
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "Success"
    mock_process.stderr = ""
    mock_subprocess_run.return_value = mock_process

    # Define some dummy parameters
    params = {
        "batch_size": 8,
        "hidden_size": 256,
        "learning_rate": 1e-4,
    }

    # Run the experiment
    result = run_experiment(params, use_wandb=False, quick_test=True, monitor_gpu=False)

    # Check that subprocess.run was called with the correct command
    mock_subprocess_run.assert_called_once()
    cmd = mock_subprocess_run.call_args[0][0]
    assert "--batch-size" in cmd
    assert "8" in cmd
    assert "--hidden-size" in cmd
    assert "256" in cmd
    assert "--learning-rate" in cmd
    assert "0.0001" in cmd
    assert "--quick-test" in cmd

    # Check that the result is correct
    assert result["status"] == "success"
    assert result["params"] == params


@patch("chart_hero.model_training.run_experiments.run_experiment")
@patch("sys.argv", ["run_experiments.py", "--quick-test"])
def test_main(mock_run_experiment):
    """Test the main function of the experiment runner."""
    from chart_hero.model_training.run_experiments import main

    # Configure the mock to return a dictionary
    mock_run_experiment.return_value = {
        "tag": "test_tag",
        "status": "success",
        "duration_seconds": 1.23,
        "params": {},
    }

    main()
    mock_run_experiment.assert_called_once()
