from pathlib import Path
from unittest.mock import patch

from chart_hero.utils.file_utils import (
    find_chart_hero_project_root,
    get_audio_set_dir,
    get_e_gmd_v1_0_0_dir,
    get_labeled_audio_set_dir,
    get_model_backup_dir,
    get_model_path,
    get_model_training_dir,
    get_process_later_dir,
    get_training_data_dir,
    parallel_rmtree,
)


def test_find_chart_hero_project_root() -> None:
    """Test that the project root can be found."""
    root = find_chart_hero_project_root()
    assert root is not None
    assert (root / ".git").is_dir()


def test_get_training_data_dir(tmp_path: Path) -> None:
    """Test that the training_data directory can be found."""
    with patch(
        "chart_hero.utils.file_utils.find_chart_hero_project_root"
    ) as mock_find_root:
        mock_find_root.return_value = tmp_path
        (tmp_path / "training_data").mkdir()
        training_data_dir = get_training_data_dir()
        assert training_data_dir.is_dir()
        assert training_data_dir.name == "training_data"


def test_get_model_training_dir(tmp_path: Path) -> None:
    """Test that the model_training directory can be found."""
    with patch(
        "chart_hero.utils.file_utils.find_chart_hero_project_root"
    ) as mock_find_root:
        mock_find_root.return_value = tmp_path
        (tmp_path / "model_training").mkdir()
        model_training_dir = get_model_training_dir()
        assert model_training_dir.is_dir()
        assert model_training_dir.name == "model_training"


def test_get_e_gmd_v1_0_0_dir(tmp_path: Path) -> None:
    """Test that the e-gmd-v1.0.0 directory can be found."""
    with patch(
        "chart_hero.utils.file_utils.get_training_data_dir"
    ) as mock_get_training_data_dir:
        mock_get_training_data_dir.return_value = tmp_path
        (tmp_path / "e-gmd-v1.0.0").mkdir()
        e_gmd_dir = get_e_gmd_v1_0_0_dir()
        assert e_gmd_dir.is_dir()
        assert e_gmd_dir.name == "e-gmd-v1.0.0"


def test_get_audio_set_dir(tmp_path: Path) -> None:
    """Test that the audio_set directory can be found."""
    with patch(
        "chart_hero.utils.file_utils.get_e_gmd_v1_0_0_dir"
    ) as mock_get_e_gmd_dir:
        mock_get_e_gmd_dir.return_value = tmp_path
        (tmp_path / "drummer1" / "1" / "audio_set").mkdir(parents=True)
        audio_set_dir = get_audio_set_dir("drummer1", 1)
        assert audio_set_dir.is_dir()
        assert audio_set_dir.name == "audio_set"


def test_get_labeled_audio_set_dir() -> None:
    """Test that the labeled_audio_set directory can be created."""
    labeled_audio_set_dir = get_labeled_audio_set_dir("drummer1", 1)
    assert labeled_audio_set_dir.is_dir()
    assert labeled_audio_set_dir.name == "labeled_audio_set"


def test_get_process_later_dir() -> None:
    """Test that the process_later directory can be created."""
    process_later_dir = get_process_later_dir("drummer1", 1)
    assert process_later_dir.is_dir()
    assert process_later_dir.name == "process_later"


def test_get_model_backup_dir() -> None:
    """Test that the model_backup directory can be created."""
    model_backup_dir = get_model_backup_dir()
    assert model_backup_dir.is_dir()
    assert model_backup_dir.name == "model_backup"


def test_get_model_path() -> None:
    """Test that the model.keras path can be found."""
    model_path = get_model_path()
    assert model_path.name == "model.keras"


def test_parallel_rmtree(tmp_path: Path) -> None:
    """
    Test that parallel_rmtree correctly deletes a directory structure.
    """
    # Create a dummy directory structure for testing.
    dummy_root = tmp_path / "temp_test_dir"
    dummy_root.mkdir(exist_ok=True)
    num_dirs = 5
    files_per_dir = 20

    for i in range(num_dirs):
        sub_dir = dummy_root / f"subdir_{i}"
        sub_dir.mkdir(exist_ok=True)
        for j in range(files_per_dir):
            (sub_dir / f"file_{j}.txt").touch()

    # Run the parallel deletion
    parallel_rmtree(dummy_root)

    # Assert that the directory no longer exists
    assert not dummy_root.exists()


def test_parallel_rmtree_empty_dir(tmp_path: Path) -> None:
    """
    Test that parallel_rmtree correctly deletes an empty directory.
    """
    # Create an empty directory
    dummy_root = tmp_path / "empty_dir"
    dummy_root.mkdir(exist_ok=True)

    # Run the parallel deletion
    parallel_rmtree(dummy_root)

    # Assert that the directory no longer exists
    assert not dummy_root.exists()


def test_parallel_rmtree_non_existent_dir() -> None:
    """
    Test that parallel_rmtree handles a non-existent directory gracefully.
    """
    # Run the parallel deletion on a non-existent directory
    parallel_rmtree("non_existent_dir")
