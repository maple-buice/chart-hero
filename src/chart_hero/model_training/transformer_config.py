import os
from dataclasses import dataclass
from typing import Tuple

import torch

# region Drum Hits
# Final map based on README.md
DRUM_HIT_MAP = {
    22: "67",  # Hi-hat Closed (Edge) -> HiHatCymbal
    26: "67",  # Hi-hat Open (Edge) -> HiHatCymbal
    35: "0",  # Acoustic Bass Drum -> Kick
    36: "0",  # Kick / Bass Drum 1 -> Kick
    37: "1",  # Snare X-Stick / Side Stick -> Snare
    38: "1",  # Snare (Head) / Acoustic Snare -> Snare
    39: "67",  # Hand Clap / Cowbell -> HiHatCymbal (Treating as percussion)
    40: "1",  # Snare (Rim) / Electric Snare -> Snare
    41: "4",  # Low Floor Tom	-> LowTom
    42: "67",  # Hi-hat Closed (Bow) / Closed Hi-Hat -> HiHatCymbal
    43: "4",  # Tom 3 (Head) / High Floor Tom -> LowTom
    44: "67",  # Hi-hat Pedal / Pedal Hi-Hat -> HiHatCymbal
    45: "3",  # Tom 2 / Low Tom -> MiddleTom
    46: "67",  # Hi-hat Open (Bow) / Open Hi-Hat -> HiHatCymbal
    47: "3",  # Tom 2 (Rim) / Low-Mid Tom -> MiddleTom
    48: "2",  # Tom 1 / Hi-Mid Tom -> HighTom
    49: "66",  # Crash 1 (Bow) / Crash Cymbal 1 -> CrashCymbal
    50: "2",  # Tom 1 (Rim) / High Tom -> HighTom
    51: "68",  # Ride (Bow) / Ride Cymbal 1 -> RideCymbal
    52: "66",  # Crash 2 (Edge) / Chinese Cymbal -> CrashCymbal
    53: "68",  # Ride (Bell) / Ride Bell -> RideCymbal
    54: "67",  # Tambourine / Cowbell -> HiHatCymbal (Treating as percussion)
    55: "66",  # Crash 1 (Edge) / Splash Cymbal -> CrashCymbal
    56: "67",  # Cowbell -> HiHatCymbal (Treating as percussion)
    57: "66",  # Crash 2 (Bow) / Crash Cymbal 2 -> CrashCymbal
    58: "4",  # Tom 3 (Rim) / Vibraslap -> LowTom
    59: "68",  # Ride (Edge) / Ride Cymbal 2 -> RideCymbal
    # --- Adding potentially missing mappings based on common GM ---
    # Toms
    60: "2",  # Hi Bongo -> HighTom
    61: "3",  # Low Bongo -> MiddleTom
    62: "2",  # Mute Hi Conga -> HighTom
    63: "3",  # Open Hi Conga -> MiddleTom
    64: "4",  # Low Conga -> LowTom
    65: "2",  # High Timbale -> HighTom
    66: "3",  # Low Timbale -> MiddleTom
    # Percussion -> Map to HiHat for simplicity or create separate classes later
    67: "2",  # High Agogo -> HighTom (Could be percussion)
    68: "3",  # Low Agogo -> MiddleTom (Could be percussion)
    69: "67",  # Cabasa -> HiHatCymbal
    70: "67",  # Maracas -> HiHatCymbal
    # Cymbals/Effects -> Map reasonably
    71: "68",  # Short Whistle -> RideCymbal (Treat as effect/cymbal)
    72: "66",  # Long Whistle -> CrashCymbal (Treat as effect/cymbal)
    73: "68",  # Short Guiro -> RideCymbal (Treat as effect/cymbal)
    74: "66",  # Long Guiro -> CrashCymbal (Treat as effect/cymbal)
    75: "67",  # Claves -> HiHatCymbal
    # Wood Blocks -> Map to Toms
    76: "2",  # Hi Wood Block -> HighTom
    77: "3",  # Low Wood Block -> MiddleTom
    # Cuica -> Map to Toms
    78: "2",  # Mute Cuica -> HighTom
    79: "3",  # Open Cuica -> MiddleTom
    # Triangle -> Map to Cymbals
    80: "68",  # Mute Triangle -> RideCymbal
    81: "66",  # Open Triangle -> CrashCymbal
}

# Define the target classes based on the Clone Hero mapping values
TARGET_CLASSES = sorted(
    list(set(DRUM_HIT_MAP.values()))
)  # ['0', '1', '2', '3', '4', '66', '67', '68']
NUM_DRUM_HITS = len(TARGET_CLASSES)
DRUM_HIT_TO_INDEX = {hit: idx for idx, hit in enumerate(TARGET_CLASSES)}
INDEX_TO_DRUM_HIT = {idx: hit for idx, hit in enumerate(TARGET_CLASSES)}


def get_drum_hits() -> list[str]:
    """Returns the sorted list of target drum hit classes."""
    return TARGET_CLASSES


def get_drum_hits_as_strings() -> list[str]:
    """Returns the sorted list of target drum hit classes as strings (same as get_drum_hits)."""
    # Simple mapping for clarity in reports
    name_map = {
        "0": "Kick",
        "1": "Snare",
        "2": "HiTom",
        "3": "MidTom",
        "4": "LowTom",
        "66": "Crash",
        "67": "HiHat",
        "68": "Ride",
    }
    return [name_map.get(hit, hit) for hit in TARGET_CLASSES]


# endregion


@dataclass
class BaseConfig:
    """Base configuration for transformer training."""

    # Model architecture - smaller defaults for memory efficiency
    model_name: str = "audio_spectrogram_transformer"
    hidden_size: int = 384  # Reduced from 768
    num_layers: int = 6  # Reduced from 12
    num_heads: int = 6  # Reduced from 12
    intermediate_size: int = 1536  # Reduced from 3072
    dropout: float = 0.1

    # Audio processing
    sample_rate: int = 22050
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    max_audio_length: float = 10.0  # seconds

    # Transformer input
    patch_size: Tuple[int, int] = (16, 16)  # (time, frequency)
    max_seq_len: int = 1024

    # Training
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    gradient_checkpointing: bool = False
    deterministic_training: bool = True

    # Data
    train_batch_size: int = 32
    val_batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    data_dir: str = "../datasets/processed/"
    audio_dir: str = "../datasets/processed/"
    log_dir: str = "logs/"
    model_dir: str = "models/local_transformer_models/"

    # Device
    device: str = "cpu"
    mixed_precision: bool = False
    precision: str = "32"

    # Drum classification
    num_drum_classes: int = len(TARGET_CLASSES)

    # Regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2

    # SpecAugment
    enable_spec_augmentation: bool = True
    spec_aug_num_time_masks: int = 2
    spec_aug_max_time_mask_percentage: float = 0.05
    spec_aug_num_freq_masks: int = 2
    spec_aug_max_freq_mask_percentage: float = 0.15

    # Timbre Augmentation
    enable_timbre_augmentation: bool = True
    pitch_jitter_prob: float = 0.5
    time_stretch_prob: float = 0.5
    dynamic_eq_prob: float = 0.5

    # Checkpointing
    save_top_k: int = 3
    monitor: str = "val_f1"
    mode: str = "max"

    # Logging
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0  # How often to run validation (1.0 = every epoch)
    seed: int = 42


@dataclass
class LocalConfig(BaseConfig):
    """Configuration optimized for M1-Max MacBook Pro (64GB RAM, 1TB storage)."""

    # Device settings
    device: str = "cpu"
    mixed_precision: bool = False  # MPS has limited mixed precision support
    precision: str = (
        "32"  # MPS doesn't fully support mixed precision yet in PyTorch Lightning
    )

    # Smaller model for memory efficiency
    num_heads: int = 4  # Changed from 6 to 4 for compatibility with hidden_size 256

    # Memory optimization for 64GB RAM with MPS GPU
    train_batch_size: int = 4  # Conservative default for MPS
    val_batch_size: int = 8  # Conservative default for MPS
    num_workers: int = 2  # Reduced for MPS stability
    pin_memory: bool = False  # Disabled since not supported on MPS

    # Training settings
    gradient_checkpointing: bool = True
    accumulate_grad_batches: int = (
        8  # Increased from 4 to maintain effective batch size
    )

    # Storage optimization for 1TB SSD
    cache_dataset: bool = False  # Disable caching to save memory
    prefetch_factor: int = 2

    # Conservative settings for local development
    max_audio_length: float = 5.0  # Further reduced for memory efficiency
    max_seq_len: int = 512  # Reduced from 768

    def __post_init__(self):
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"

    @property
    def effective_batch_size(self) -> int:
        return self.train_batch_size * self.accumulate_grad_batches


@dataclass
class LocalPerformanceConfig(LocalConfig):
    """A more aggressive configuration for powerful local machines."""

    train_batch_size: int = 16
    val_batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True


@dataclass
class LocalMaxPerformanceConfig(LocalPerformanceConfig):
    """An even more aggressive configuration for powerful local machines."""

    train_batch_size: int = 24


@dataclass
class OvernightConfig(LocalConfig):  # Inherits from LocalConfig for MPS settings
    """Configuration optimized for overnight training on local MPS."""

    num_epochs: int = 200  # Example: Increased epochs for overnight run
    # Override any other LocalConfig settings as needed for overnight runs
    # For example, you might want to ensure quick_test specific settings are off
    # or adjust learning rate schedule, or ensure full dataset is used.
    # These would typically be handled by the config itself rather than CLI flags for an 'overnight' profile.

    # Ensure paths are appropriate if not overridden by CLI in train_transformer.py
    # data_dir, audio_dir, log_dir, model_dir will be inherited from LocalConfig
    # or overridden by train_transformer.py if CLI args are provided.

    # Example: ensure no quick_test limits are applied from a base class if any existed
    # This is more for illustration, as train_transformer.py handles --quick-test CLI
    # limit_train_batches: Optional[float] = None
    # limit_val_batches: Optional[float] = None

    # You might want a specific model_dir suffix for overnight runs if not using experiment_tag for the main folder
    # model_dir: str = "models/local_transformer_models/overnight/"


@dataclass
class CloudConfig(BaseConfig):
    """Configuration optimized for Google Colab with GPU."""

    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    precision: str = "16-mixed"

    # GPU optimization
    train_batch_size: int = 64
    val_batch_size: int = 128
    num_workers: int = 4  # Conservative for Colab
    pin_memory: bool = True

    # Training settings
    accumulate_grad_batches: int = 4  # Larger effective batch size
    gradient_checkpointing: bool = False  # GPU has more memory

    # Colab-specific paths (will be created if they don't exist)
    data_dir: str = "../datasets/"
    model_dir: str = "model_training/transformer_models/"
    log_dir: str = "logs/"

    # Longer sequences for GPU training
    max_audio_length: float = 12.0
    max_seq_len: int = 1536

    # More aggressive training for cloud resources
    learning_rate: float = 2e-4
    warmup_steps: int = 2000

    @property
    def effective_batch_size(self) -> int:
        return self.train_batch_size * self.accumulate_grad_batches


def get_config(config_type: str = "local") -> BaseConfig:
    """Get configuration based on environment type."""

    config_type_lower = config_type.lower()
    if config_type_lower == "local":
        return LocalConfig()
    elif config_type_lower == "local_performance":
        return LocalPerformanceConfig()
    elif config_type_lower == "local_max_performance":
        return LocalMaxPerformanceConfig()
    elif config_type_lower == "cloud":
        return CloudConfig()
    elif config_type_lower == "overnight_default":  # Added new config type
        return OvernightConfig()
    else:
        raise ValueError(
            f"Unknown config type: {config_type}. Use 'local', 'cloud', or 'overnight_default'."
        )


def auto_detect_config() -> BaseConfig:
    """Automatically detect the best configuration based on available hardware."""

    # Check if running in Google Colab
    try:
        __import__("google.colab")
        return CloudConfig()
    except ImportError:
        pass

    # Check for CUDA
    if torch.cuda.is_available():
        return CloudConfig()

    # Check for Apple Silicon MPS
    if torch.backends.mps.is_available():
        return LocalConfig()

    # Default to CPU local config
    config = LocalConfig()
    config.device = "cpu"
    config.mixed_precision = False
    config.precision = "32"
    return config


def validate_config(config: BaseConfig) -> None:
    """Validate configuration parameters."""

    # Check device availability
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but config specifies CUDA device")

    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available but config specifies MPS device")

    # Validate paths exist
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Validate audio parameters
    assert config.sample_rate > 0, "Sample rate must be positive"
    assert config.max_audio_length > 0, "Max audio length must be positive"
    assert config.n_mels > 0, "Number of mel bins must be positive"

    # Validate model parameters
    assert config.hidden_size > 0, "Hidden size must be positive"
    assert config.num_layers > 0, "Number of layers must be positive"
    assert config.num_heads > 0, "Number of heads must be positive"
    assert (
        config.hidden_size % config.num_heads == 0
    ), "Hidden size must be divisible by num_heads"

    # Validate training parameters
    assert config.learning_rate > 0, "Learning rate must be positive"
    assert 0 <= config.dropout <= 1, "Dropout must be between 0 and 1"
    assert config.train_batch_size > 0, "Batch size must be positive"
