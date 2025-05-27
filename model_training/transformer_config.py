"""
Configuration classes for transformer-based drum transcription training.
Supports both local (M1-Max MacBook Pro) and cloud (Google Colab) environments.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import os


@dataclass
class BaseConfig:
    """Base configuration for transformer training."""
    
    # Model architecture - smaller defaults for memory efficiency
    model_name: str = "audio_spectrogram_transformer"
    hidden_size: int = 384  # Reduced from 768
    num_layers: int = 6     # Reduced from 12
    num_heads: int = 6      # Reduced from 12
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
    
    # Data
    train_batch_size: int = 32
    val_batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Drum classification
    num_drum_classes: int = 9  # kick, snare, hi-hat, crash, ride, high-tom, mid-tom, low-tom, other
    
    # Regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    
    # Checkpointing
    save_top_k: int = 3
    monitor: str = "val_f1"
    mode: str = "max"
    
    # Logging
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0  # How often to run validation (1.0 = every epoch)


@dataclass
class LocalConfig(BaseConfig):
    """Configuration optimized for M1-Max MacBook Pro (64GB RAM, 1TB storage)."""
    
    # Device settings
    device: str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = False # MPS has limited mixed precision support
    precision: str = "32"  # MPS doesn't fully support mixed precision yet in PyTorch Lightning
    
    # Smaller model for memory efficiency
    hidden_size: int = 384  # Reduced from 768
    num_layers: int = 6     # Reduced from 12
    num_heads: int = 4      # Changed from 6 to 4 for compatibility with hidden_size 256
    intermediate_size: int = 1536  # Reduced from 3072
    
    # Memory optimization for 64GB RAM with MPS GPU
    train_batch_size: int = 4   # Conservative default for MPS
    val_batch_size: int = 8    # Conservative default for MPS
    num_workers: int = 2        # Reduced for MPS stability
    pin_memory: bool = False    # Disabled since not supported on MPS
    
    # Training settings
    gradient_checkpointing: bool = True
    accumulate_grad_batches: int = 8  # Increased from 4 to maintain effective batch size
    
    # Storage optimization for 1TB SSD
    cache_dataset: bool = False  # Disable caching to save memory
    prefetch_factor: int = 2
    
    # Paths
    data_dir: str = "datasets/processed/"  # Corrected to point to where _train.pkl etc. are saved
    audio_dir: str = "datasets/processed/"  # Should contain the actual audio files, or be derivable
    log_dir: str = "logs/"
    model_dir: str = "models/local_transformer_models/" # Renamed from model_save_path and changed to a directory
    
    # Conservative settings for local development
    max_audio_length: float = 5.0  # Further reduced for memory efficiency
    max_seq_len: int = 512         # Reduced from 768
    
    # Logging and saving
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0  # How often to run validation (1.0 = every epoch)
    device: str = "mps"  # Device to use for training (e.g., cpu, mps, cuda)
    mixed_precision: bool = False  # Whether to use mixed precision training
    precision: str = "32"  # Precision for training (e.g., 16, 32, bf16)
    gradient_checkpointing: bool = True # Enable gradient checkpointing to save memory
    deterministic_training: bool = True # Added: For reproducibility

    @property
    def effective_batch_size(self) -> int:
        return self.train_batch_size * self.accumulate_grad_batches


@dataclass
class OvernightConfig(LocalConfig): # Inherits from LocalConfig for MPS settings
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
    data_dir: str = "datasets/"
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
    elif config_type_lower == "cloud":
        return CloudConfig()
    elif config_type_lower == "overnight_default": # Added new config type
        return OvernightConfig()
    else:
        raise ValueError(f"Unknown config type: {config_type}. Use 'local', 'cloud', or 'overnight_default'.")


def auto_detect_config() -> BaseConfig:
    """Automatically detect the best configuration based on available hardware."""
    
    # Check if running in Google Colab
    try:
        import google.colab # type: ignore
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
    # Use model_save_path for LocalConfig and model_dir for CloudConfig
    if hasattr(config, 'model_save_path') and isinstance(config.model_save_path, str):
        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    elif hasattr(config, 'model_dir') and isinstance(config.model_dir, str):
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
    assert config.hidden_size % config.num_heads == 0, "Hidden size must be divisible by num_heads"
    
    # Validate training parameters
    assert config.learning_rate > 0, "Learning rate must be positive"
    assert 0 <= config.dropout <= 1, "Dropout must be between 0 and 1"
    assert config.train_batch_size > 0, "Batch size must be positive"


if __name__ == "__main__":
    # Test configuration
    print("Testing configuration classes...")
    
    # Test local config
    local_config = get_config("local")
    # Ensure audio_dir is correctly set for local_config before validation
    local_config.audio_dir = "datasets/processed/" # This might need to be more robust or handled by config itself
    validate_config(local_config)
    print(f"Local config: {local_config.device}, batch_size={local_config.train_batch_size}, audio_dir={local_config.audio_dir}")
    
    # Test cloud config
    cloud_config = get_config("cloud")
    validate_config(cloud_config)
    print(f"Cloud config: {cloud_config.device}, batch_size={cloud_config.train_batch_size}")

    # Test overnight config
    overnight_config = get_config("overnight_default")
    overnight_config.audio_dir = "datasets/processed/" # Example, ensure paths are valid for testing
    validate_config(overnight_config)
    print(f"Overnight config: {overnight_config.device}, epochs={overnight_config.num_epochs}, audio_dir={overnight_config.audio_dir}")
    
    # Test auto-detection
    auto_config = auto_detect_config()
    validate_config(auto_config)
    print(f"Auto-detected config: {auto_config.device}")
    
    print("Configuration tests passed!")