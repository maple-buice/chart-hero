from chart_hero.inference.inference_utils import map_patch_to_sample


def test_map_patch_to_sample():
    sample = map_patch_to_sample(
        start_frame=100,
        patch_idx=3,
        stride_frames=2,
        patch_size=4,
        hop_length=10,
        offset_samples=5,  # Not used anymore, but kept for API compatibility
        add_ms=0.0,
        sample_rate=1000,
    )
    # offset_samples is no longer added to prevent double offset application
    expected = (100 + 3 * 2 + 2) * 10  # frame * hop_length
    assert sample == expected
