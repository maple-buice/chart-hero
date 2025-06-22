#!/usr/bin/env python3
"""
Simple focused test to demonstrate positioned progress bars clearly.
"""

import sys
import os
import logging
import signal
import time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model_training.data_preparation import data_preparation

# Timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Test timed out!")

# Configure logging with less verbosity
logging.basicConfig(
    level=logging.WARNING,  # Reduce log noise
    format='%(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_progress_bars_focused():
    """Focused test with minimal files to clearly show progress bars."""
    
    dataset_dir = "/Users/maple/Repos/chart-hero/datasets/e-gmd-v1.0.0"
    output_dir = "/Users/maple/Repos/chart-hero/datasets/quick_tqdm_test"
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False
    
    try:
        print("🎯 Testing positioned progress bars with minimal dataset...")
        print("Watch for clean, positioned progress bars below:\n")
        
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # 60 second timeout
        
        # Create tiny sample
        data_prep = data_preparation(dataset_dir, 'egmd', sample_ratio=0.001)  # ~2-5 files
        
        print(f"\n✅ Created dataset with {len(data_prep.midi_wav_map)} file pairs")
        print("Now processing with positioned progress bars...\n")
        
        # Process with clear progress bars
        result = data_prep.create_audio_set(
            pad_before=0.05,
            pad_after=0.05,
            fix_length=1,  # Very short - 1 second slices
            batching=True,
            dir_path=output_dir,
            num_batches=1,
            memory_limit_gb=8,
            enable_process_parallelization=False
        )
        
        print(f"\n🎉 Test completed! Processed {result} files")
        print("✅ You should have seen clean positioned progress bars above")
        
        # Cancel timeout
        signal.alarm(0)
        return True
        
    except TimeoutException:
        print("⏰ Test timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        signal.alarm(0)  # Cancel timeout

if __name__ == "__main__":
    print("🚀 Quick Positioned Progress Bar Test")
    print("="*50)
    
    success = test_progress_bars_focused()
    
    if success:
        print("\n🎉 SUCCESS: Positioned progress bars working correctly!")
    else:
        print("\n❌ FAILED: Check the output above for issues")
    
    sys.exit(0 if success else 1)
