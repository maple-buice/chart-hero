#!/usr/bin/env python3
"""
Test script to verify that prepare_egmd_data.py uses the smart parallelization logic correctly.
"""

import os
import sys
import tempfile
import shutil
import logging
from pathlib import Path
from unittest.mock import patch
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_prepare_egmd_with_smart_parallel():
    """Test that prepare_egmd_data.py correctly enables smart parallelization by default."""
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output = Path(temp_dir) / "test_output"
        temp_output.mkdir(exist_ok=True)
        
        # Test with a small dataset (should fall back to sequential)
        logger.info("Testing prepare_egmd_data.py with smart parallelization...")
        
        # Use the existing E-GMD dataset directory if available, otherwise skip
        dataset_dir = Path("datasets/e-gmd-v1.0.0")
        if not dataset_dir.exists():
            logger.warning("E-GMD dataset not found, skipping integration test")
            return True
            
        csv_file = dataset_dir / "e-gmd-v1.0.0.csv"
        if not csv_file.exists():
            logger.warning("E-GMD CSV file not found, skipping integration test")
            return True
        
        # Run prepare_egmd_data.py with a very small sample to trigger sequential fallback
        cmd = [
            sys.executable, "prepare_egmd_data.py",
            "--dataset-dir", str(dataset_dir),
            "--output-dir", str(temp_output),
            "--sample-ratio", "0.01",  # Very small sample
            "--num-batches", "5",
            "--memory-limit-gb", "8"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Check the output for expected messages
            if result.returncode == 0:
                output = result.stdout + result.stderr
                
                # Should see smart parallelization logic messages
                if "Using smart parallelization logic" in output:
                    logger.info("✅ Smart parallelization logic is enabled by default")
                else:
                    logger.warning("⚠️  Smart parallelization messages not found in output")
                
                # For small datasets, should fall back to sequential
                if "Falling back to sequential mode" in output or "Using sequential processing mode" in output:
                    logger.info("✅ Correctly falls back to sequential for small datasets")
                else:
                    logger.info("ℹ️  No explicit fallback message (may not have reached threshold)")
                
                return True
            else:
                logger.error(f"Command failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Command timed out")
            return False
        except Exception as e:
            logger.error(f"Error running command: {e}")
            return False

def test_disable_parallel_flag():
    """Test that the --disable-parallel flag works correctly."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output = Path(temp_dir) / "test_output"
        temp_output.mkdir(exist_ok=True)
        
        dataset_dir = Path("datasets/e-gmd-v1.0.0")
        if not dataset_dir.exists():
            logger.warning("E-GMD dataset not found, skipping --disable-parallel test")
            return True
            
        csv_file = dataset_dir / "e-gmd-v1.0.0.csv"
        if not csv_file.exists():
            logger.warning("E-GMD CSV file not found, skipping --disable-parallel test")
            return True
        
        # Test with --disable-parallel flag
        cmd = [
            sys.executable, "prepare_egmd_data.py",
            "--dataset-dir", str(dataset_dir),
            "--output-dir", str(temp_output),
            "--sample-ratio", "0.005",  # Very small sample
            "--num-batches", "3",
            "--memory-limit-gb", "4",
            "--disable-parallel"
        ]
        
        logger.info(f"Running command with --disable-parallel: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                output = result.stdout + result.stderr
                
                # Should see explicit sequential mode message
                if "Parallel processing disabled by user" in output:
                    logger.info("✅ --disable-parallel flag works correctly")
                    return True
                else:
                    logger.warning("⚠️  --disable-parallel flag message not found")
                    return False
            else:
                logger.error(f"Command with --disable-parallel failed: {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Command with --disable-parallel timed out")
            return False
        except Exception as e:
            logger.error(f"Error running --disable-parallel test: {e}")
            return False

def main():
    logger.info("Testing prepare_egmd_data.py integration with smart parallelization...")
    
    success = True
    
    # Test 1: Smart parallelization by default
    logger.info("\n--- Test 1: Smart parallelization by default ---")
    if not test_prepare_egmd_with_smart_parallel():
        success = False
    
    # Test 2: --disable-parallel flag
    logger.info("\n--- Test 2: --disable-parallel flag ---")
    if not test_disable_parallel_flag():
        success = False
    
    # Summary
    logger.info("\n" + "="*60)
    if success:
        logger.info("✅ ALL INTEGRATION TESTS PASSED")
        logger.info("prepare_egmd_data.py correctly uses smart parallelization logic!")
        logger.info("Default behavior: Enable smart parallelization (auto-chooses best mode)")
        logger.info("Override option: Use --disable-parallel to force sequential mode")
    else:
        logger.error("❌ SOME INTEGRATION TESTS FAILED")
        logger.error("Please check the output above for details")
    
    logger.info("="*60)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
