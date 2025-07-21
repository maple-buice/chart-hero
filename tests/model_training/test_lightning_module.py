import torch
from chart_hero.model_training.lightning_module import DrumTranscriptionModule
from chart_hero.model_training.transformer_config import get_config


def test_common_step_with_non_contiguous_tensor():
    """
    Test that the _common_step method can handle non-contiguous labels.
    This reproduces the RuntimeError when using .view() on a non-contiguous tensor.
    """
    config = get_config("local")
    time_dimension = 224
    max_time_patches = time_dimension // config.patch_size[0]
    model = DrumTranscriptionModule(config)

    # Create a non-contiguous tensor for the labels by taking a slice
    base_tensor = torch.randint(
        0, 2, (2, max_time_patches * 2, config.num_drum_classes)
    ).float()
    non_contiguous_labels = base_tensor[:, ::2, :]
    assert not non_contiguous_labels.is_contiguous()

    # Create a dummy batch with the non-contiguous labels
    batch = (
        torch.randn(2, 1, config.n_mels, time_dimension),
        non_contiguous_labels,
    )

    # This should raise a RuntimeError with the original code
    # and pass with the fix (.reshape() instead of .view()).
    model._common_step(batch)
