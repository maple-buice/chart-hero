import filecmp
import os
import tempfile

from chart_hero.prepare_egmd_data import main as prepare_egmd_data


def test_data_pipeline_integrity():
    """
    Test that the data preparation pipeline produces a consistent output.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run the data preparation script
        prepare_egmd_data(
            args=[
                "--input-dir",
                "tests/assets/dummy_data",
                "--output-dir",
                tmpdir,
            ]
        )

        # Compare the output with the golden dataset
        golden_data_dir = "tests/assets/golden_data"
        match, mismatch, errors = filecmp.cmpfiles(
            golden_data_dir, tmpdir, os.listdir(golden_data_dir), shallow=False
        )

        assert len(mismatch) == 0
        assert len(errors) == 0
