#!/usr/bin/env python3
"""
Setup script for Chart-Hero transformer training environment.
Handles both local (M1-Max) and cloud (Colab) setups.
"""

import subprocess
import sys
import os
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True


def detect_environment():
    """Detect if running on local machine or cloud environment."""
    try:
        import google.colab
        return "colab"
    except ImportError:
        pass
    
    if platform.system() == "Darwin" and platform.processor() == "arm":
        return "m1-mac"
    elif platform.system() == "Linux":
        return "linux"
    elif platform.system() == "Windows":
        return "windows"
    else:
        return "unknown"


def install_pytorch(env_type):
    """Install PyTorch with appropriate backend."""
    print(f"🔧 Installing PyTorch for {env_type}...")
    
    if env_type == "colab":
        # Colab usually has PyTorch pre-installed with CUDA
        cmd = ["pip", "install", "--upgrade", "torch", "torchvision", "torchaudio"]
    elif env_type == "m1-mac":
        # M1 Mac with MPS support
        cmd = ["pip", "install", "torch", "torchvision", "torchaudio"]
    else:
        # CPU-only for other platforms
        cmd = ["pip", "install", "torch", "torchvision", "torchaudio", "--index-url", 
               "https://download.pytorch.org/whl/cpu"]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e}")
        return False


def install_dependencies():
    """Install remaining dependencies."""
    print("🔧 Installing dependencies...")
    
    dependencies = [
        "pytorch-lightning>=2.0.0",
        "transformers>=4.30.0", 
        "timm>=0.9.0",
        "wandb>=0.15.0",
        "librosa>=0.10.0",
        "soundfile",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "joblib",
        "audiomentations"
    ]
    
    try:
        for dep in dependencies:
            print(f"Installing {dep}...")
            subprocess.run(["pip", "install", dep], check=True, capture_output=True)
        print("✅ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "datasets",
        "model_training/transformer_models",
        "logs",
        "colab"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {dir_path}/")
    
    return True


def test_installation():
    """Test if the installation is working."""
    print("🧪 Testing installation...")
    
    try:
        # Test PyTorch
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        
        # Test device availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) available")
        else:
            print("ℹ️  Using CPU backend")
        
        # Test transformer components
        sys.path.insert(0, os.getcwd())
        from model_training.transformer_config import get_config, auto_detect_config
        from model_training.transformer_model import create_model
        
        config = auto_detect_config()
        print(f"✅ Auto-detected config: {config.__class__.__name__}")
        
        # Test model creation
        model = create_model(config)
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Chart-Hero Transformer Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Detect environment
    env_type = detect_environment()
    print(f"🔍 Detected environment: {env_type}")
    
    # Install PyTorch
    if not install_pytorch(env_type):
        sys.exit(1)
    
    # Install other dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("⚠️  Installation completed but tests failed. Please check manually.")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Prepare your EGMD dataset in the datasets/ directory")
    print("2. Run: python model_training/test_transformer_setup.py")
    print("3. Start training: python model_training/train_transformer.py --config auto")
    
    if env_type == "colab":
        print("4. For Colab: Use the notebook in colab/transformer_training_colab.ipynb")


if __name__ == "__main__":
    main()