from chart_hero.inference.inference_utils import map_patch_to_sample


def test_map_patch_to_sample():
    sample = map_patch_to_sample(
        start_frame=100,
        patch_idx=3,
        stride_frames=2,
        patch_size=4,
        hop_length=10,
        offset_samples=5,
        add_ms=0.0,
        sample_rate=1000,
    )
    expected = 5 + (100 + 3 * 2 + 2) * 10
    assert sample == expected
