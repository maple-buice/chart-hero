import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from tqdm import tqdm


def find_chart_hero_project_root() -> Optional[Path]:
    """Find the project root by looking for a .git directory."""
    current_path = Path.cwd()
    while current_path != current_path.parent:
        if (current_path / ".git").is_dir():
            return current_path
        current_path = current_path.parent
    return None


def get_training_data_dir() -> Path:
    """Get the path to the training_data directory."""
    root = find_chart_hero_project_root()
    if root is None:
        raise FileNotFoundError("Could not find project root.")
    return root / "training_data"


def get_model_training_dir() -> Path:
    """Get the path to the model_training directory."""
    root = find_chart_hero_project_root()
    if root is None:
        raise FileNotFoundError("Could not find project root.")
    return root / "model_training"


def get_e_gmd_v1_0_0_dir() -> Path:
    """Get the path to the e-gmd-v1.0.0 directory."""
    training_data_dir = get_training_data_dir()
    return training_data_dir / "e-gmd-v1.0.0"


def get_audio_set_dir(drummer: str, session: int) -> Path:
    """Get the path to the audio_set directory for a given drummer and session."""
    e_gmd_dir = get_e_gmd_v1_0_0_dir()
    return e_gmd_dir / drummer / str(session) / "audio_set"


def get_labeled_audio_set_dir(drummer: str, session: int) -> Path:
    """Get the path to the labeled_audio_set directory for a given drummer and session."""
    e_gmd_dir = get_e_gmd_v1_0_0_dir()
    labeled_audio_dir = e_gmd_dir / drummer / str(session) / "labeled_audio_set"
    labeled_audio_dir.mkdir(parents=True, exist_ok=True)
    return labeled_audio_dir


def get_process_later_dir(drummer: str, session: int) -> Path:
    """Get the path to the process_later directory for a given drummer and session."""
    e_gmd_dir = get_e_gmd_v1_0_0_dir()
    process_later_dir = e_gmd_dir / drummer / str(session) / "process_later"
    process_later_dir.mkdir(parents=True, exist_ok=True)
    return process_later_dir


def get_model_backup_dir() -> Path:
    """Get the path to the model_backup directory."""
    model_training_dir = get_model_training_dir()
    model_backup_dir = model_training_dir / "model_backup"
    model_backup_dir.mkdir(parents=True, exist_ok=True)
    return model_backup_dir


def get_model_path() -> Path:
    """Get the path to the model.keras file."""
    model_training_dir = get_model_training_dir()
    return model_training_dir / "model.keras"


def find_wav_files(directory: Path) -> list[Path]:
    """Find all .wav files in a directory."""
    return list(directory.glob("*.wav"))


def find_files_by_name_parts(directory: Path, name_parts: list[str]) -> list[Path]:
    """Find files in a directory that contain all of the given name parts."""
    files = []
    for file in directory.iterdir():
        if file.is_file() and all(part in file.name for part in name_parts):
            files.append(file)
    return files


def parallel_rmtree(path: str | Path, num_workers: int | None = None) -> None:
    """
    Deletes a directory tree in parallel with a progress bar.

    This function is significantly faster than shutil.rmtree() for directories
    containing a very large number of files.

    Args:
        path (str | Path): The path to the directory to delete.
        num_workers (int, optional): The number of worker threads to use.
                                     Defaults to the number of CPU cores.
    """
    path = Path(path)
    if not path.exists():
        print(f"Directory not found: {path}")
        return

    if num_workers is None:
        # os.cpu_count() is a good default for I/O-bound tasks
        num_workers = os.cpu_count() or 4

    # --- Phase 1: Scan for all files ---
    print("Scanning for files to delete...")
    all_files = [p for p in path.rglob("*") if p.is_file()]

    if not all_files:
        print("No files found to delete. Cleaning up empty directories.")
        shutil.rmtree(path)
        return

    print(f"Found {len(all_files)} files. Starting parallel deletion...")

    # --- Phase 2: Delete all files in parallel with a progress bar ---
    with tqdm(total=len(all_files), desc="Deleting files", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # The map function will apply os.remove to each file path.
            # We iterate through the results to drive the process and update the progress bar.
            for _ in executor.map(os.remove, all_files):
                pbar.update(1)

    # --- Phase 3: Delete the now-empty directory tree ---
    print("All files deleted. Removing empty directory tree.")
    shutil.rmtree(path)

    print(f"Successfully deleted {path}")
