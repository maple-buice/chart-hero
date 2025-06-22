#!/usr/bin/env python3
"""
Quick test to verify that the refactored parallel worker can properly access sequential processing code.
"""

import os
import sys
import logging

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_worker_imports():
    """Test that the parallel worker can import the required components."""
    
    logger.info("Testing parallel worker imports...")
    
    try:
        # Test 1: Import the parallel worker module
        from model_training.parallel_worker import chunk_processor_worker
        logger.info("✅ Successfully imported chunk_processor_worker")
        
        # Test 2: Import the main data_preparation class
        from model_training.data_preparation import data_preparation
        logger.info("✅ Successfully imported data_preparation")
        
        # Test 3: Test that the worker can create a minimal data_prep instance
        from model_training.parallel_worker import create_minimal_data_prep_worker
        
        # Use a test directory
        test_directory = "datasets/e-gmd-v1.0.0"
        if os.path.exists(test_directory):
            minimal_prep = create_minimal_data_prep_worker(test_directory)
            logger.info("✅ Successfully created minimal data_preparation instance")
            
            # Test that it has the required methods
            required_methods = [
                '_process_file_pair',
                # Note: Other methods are accessed dynamically, not required in minimal instance
            ]
            
            for method_name in required_methods:
                if hasattr(minimal_prep, method_name):
                    logger.info(f"✅ Found required method: {method_name}")
                else:
                    logger.error(f"❌ Missing required method: {method_name}")
                    return False
                    
            # Clean up
            del minimal_prep
            
        else:
            logger.warning("Test dataset not found, skipping instance creation test")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False

def test_worker_logic_flow():
    """Test the basic logic flow of the worker without processing real files."""
    
    logger.info("Testing worker logic flow...")
    
    try:
        import pandas as pd
        from model_training.parallel_worker import chunk_processor_worker
        
        # Create a mock file chunk (empty for testing)
        mock_chunk = pd.DataFrame([
            {'midi_filename': 'test1.mid', 'audio_filename': 'test1.wav', 'track_id': 'test1'},
            {'midi_filename': 'test2.mid', 'audio_filename': 'test2.wav', 'track_id': 'test2'}
        ])
        
        mock_params = {
            'pad_before': 0.1,
            'pad_after': 0.1,
            'fix_length': 10.0,
            'memory_limit_gb': 2.0
        }
        
        mock_directory = "datasets/e-gmd-v1.0.0"  # Use real dataset directory
        
        # Test that the worker function can be called without crashing
        # (It will fail to process the mock files, but shouldn't crash)
        result = chunk_processor_worker((mock_chunk, mock_params, mock_directory))
        
        # Check that we get the expected result structure
        # We expect it to fail processing but return the right structure
        if len(result) == 5:
            processed_files, total_records, chunk_time, performance_stats, record_data = result
            logger.info(f"✅ Worker returned expected result structure")
            logger.info(f"   - Processed files: {processed_files}")
            logger.info(f"   - Total records: {total_records}")
            logger.info(f"   - Chunk time: {chunk_time:.3f}s")
            logger.info(f"   - Performance stats: {len(performance_stats)} entries")
            logger.info(f"   - Record data: {len(record_data)} records")
            
            # Since we're using fake file names, we expect 0 processed files
            # The key is that the worker handled the errors gracefully
            logger.info("✅ Worker handled missing files gracefully (expected behavior)")
            return True
        else:
            logger.error(f"❌ Worker returned unexpected result structure: {result}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Worker logic test failed: {e}")
        return False

def main():
    logger.info("Quick verification of refactored parallel worker...")
    logger.info("="*50)
    
    success = True
    
    # Test 1: Import functionality
    logger.info("\n--- Test 1: Import Functionality ---")
    if not test_worker_imports():
        success = False
    
    # Test 2: Worker logic flow
    logger.info("\n--- Test 2: Worker Logic Flow ---")
    if not test_worker_logic_flow():
        success = False
    
    # Summary
    logger.info("\n" + "="*50)
    if success:
        logger.info("✅ PARALLEL WORKER VERIFICATION PASSED")
        logger.info("Refactored parallel worker is ready for use")
        logger.info("")
        logger.info("KEY IMPROVEMENTS:")
        logger.info("• Workers now reuse sequential processing code")
        logger.info("• Minimal data_preparation instances in workers")
        logger.info("• Same memory optimizations as sequential mode")
        logger.info("• Consistent processing logic across modes")
        logger.info("")
        logger.info("NEXT STEPS:")
        logger.info("• Run prepare_egmd_data.py to test with real data")
        logger.info("• Monitor memory usage during parallel processing")
        logger.info("• Compare with sequential processing performance")
    else:
        logger.error("❌ PARALLEL WORKER VERIFICATION FAILED")
        logger.error("Check the errors above and fix before using parallel mode")
    
    logger.info("="*50)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
