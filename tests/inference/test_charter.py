import numpy as np
import pandas as pd
import pytest
import torch
from chart_hero.inference.charter import Charter, ChartGenerator
from chart_hero.model_training.train_transformer import DrumTranscriptionModule
from chart_hero.model_training.transformer_config import get_config, get_drum_hits
from music21 import stream


@pytest.fixture
def charter_fixture(tmp_path):
    """Provides a Charter instance for testing."""
    config = get_config("local")

    # Create a dummy model and save a checkpoint
    model = DrumTranscriptionModule(config)
    checkpoint_path = tmp_path / "dummy_model.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": {"config": config},
            "pytorch-lightning_version": "2.0.0",
        },
        checkpoint_path,
    )

    charter = Charter(config, checkpoint_path)
    return charter


def test_charter_initialization(charter_fixture):
    """Test that the Charter class can be initialized."""
    assert charter_fixture is not None
    assert charter_fixture.config is not None
    assert charter_fixture.model is not None


def test_predict(charter_fixture):
    """Test the predict method."""
    # Create a list of dummy spectrogram tensors
    dummy_tensors = [torch.randn(1, 1, 430, 128) for _ in range(5)]

    chart = charter_fixture.predict(dummy_tensors)

    assert isinstance(chart, pd.DataFrame)
    assert "peak_sample" in chart.columns
    assert len(chart) > 0


def test_chart_generator():
    """Test the ChartGenerator class."""
    # Create a dummy prediction dataframe
    drum_hits = get_drum_hits()
    data = {
        "peak_sample": np.arange(1, 11) * 22050,
        **{hit: np.random.randint(0, 2, 10) for hit in drum_hits},
    }
    prediction_df = pd.DataFrame(data)

    chart_generator = ChartGenerator(
        prediction_df,
        song_duration=11,
        bpm=120,
        sample_rate=22050,
        song_title="Test Song",
    )

    assert isinstance(chart_generator.sheet, stream.Score)
    assert chart_generator.sheet.metadata.title == "Test Song"
