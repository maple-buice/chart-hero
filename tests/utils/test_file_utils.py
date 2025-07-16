import os

import pytest

from chart_hero.utils.file_utils import (
    get_dataset_dir,
    get_dir,
    get_first_match,
    get_model_training_dir,
    get_training_data_dir,
)


def test_get_dir(fs):
    """Test the get_dir function."""
    fs.create_dir("/test")
    # Test creating a directory
    new_dir = get_dir("/test", "new_dir", True)
    assert os.path.exists(new_dir)
    assert os.path.isdir(new_dir)

    # Test getting an existing directory
    existing_dir = get_dir("/test", "new_dir", False)
    assert existing_dir == new_dir

    # Test that it raises an error for a non-existent directory
    with pytest.raises(FileNotFoundError):
        get_dir("/test", "non_existent_dir", False)

    # Test that it raises an error for a file
    fs.create_file("/test/a_file")
    with pytest.raises(FileNotFoundError):
        get_dir("/test", "a_file", False)


def test_get_first_match(fs):
    """Test the get_first_match function."""
    fs.create_dir("/test")
    fs.create_file("/test/foo_bar.txt")
    fs.create_file("/test/foo_baz.txt")

    # Test finding a file
    match = get_first_match("/test", ["foo", "bar"])
    assert match == "/test/foo_bar.txt"

    # Test that it raises an error when no match is found
    with pytest.raises(Exception):
        get_first_match("/test", ["foo", "qux"])


def test_get_training_data_dir(fs):
    """Test the get_training_data_dir function."""
    # Test finding the directory in the current directory
    fs.create_dir("/test/training_data")
    os.chdir("/test")
    assert get_training_data_dir() == "/test/training_data"

    # Test finding the directory in the parent directory
    fs.create_dir("/parent/training_data")
    fs.create_dir("/parent/child")
    os.chdir("/parent/child")
    assert get_training_data_dir() == "/parent/training_data"

    # Test that it raises an error when the directory is not found
    os.chdir("/")
    with pytest.raises(FileNotFoundError):
        get_training_data_dir()


def test_get_model_training_dir(fs):
    """Test the get_model_training_dir function."""
    # Test finding the directory in the current directory
    fs.create_dir("/test/model_training")
    os.chdir("/test")
    assert get_model_training_dir() == "/test/model_training"

    # Test finding the directory in the parent directory
    fs.create_dir("/parent/model_training")
    fs.create_dir("/parent/child")
    os.chdir("/parent/child")
    assert get_model_training_dir() == "/parent/model_training"

    # Test that it raises an error when the directory is not found
    os.chdir("/")
    with pytest.raises(FileNotFoundError):
        get_model_training_dir()


def test_get_dataset_dir(fs):
    """Test the get_dataset_dir function."""
    fs.create_dir("/test/training_data/e-gmd-v1.0.0")
    os.chdir("/test")
    # This function doesn't return anything, it just raises an error if the dir is not found
    get_dataset_dir()

    # Test that it raises an error when the directory is not found
    fs.create_dir("/no_training_data")
    os.chdir("/no_training_data")
    with pytest.raises(FileNotFoundError):
        get_dataset_dir()
