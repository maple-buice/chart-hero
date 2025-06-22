#!/usr/bin/env python3
"""
Performance Analysis Test for Data Preparation Pipeline
Tests the optimized data_preparation class with comprehensive performance instrumentation.
"""

import sys
import os
import logging
import time
import psutil
import gc
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure the model_training package is discoverable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model_training.data_preparation import data_preparation

# --- Test Configuration ---
EGMD_DATASET_DIR = "/Users/maple/Repos/chart-hero/datasets/e-gmd-v1.0.0"
PROCESSED_OUTPUT_DIR = "/Users/maple/Repos/chart-hero/datasets/processed_profile"
SAMPLE_RATIO = 0.005  # Small sample for performance testing (0.5% of dataset)
NUM_BATCHES = 3  # Limited batches for focused testing
MEMORY_LIMIT_GB = 8  # Reasonable memory limit for testing
BATCH_SIZE_MULTIPLIER = 1.0  # Standard batch sizing

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'performance_test_{time.strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class PerformanceProfiler:
    """Comprehensive performance profiling utility."""
    
    def __init__(self):
        self.start_time = None
        self.process = psutil.Process()
        self.baseline_memory = None
        self.stage_times = {}
        self.memory_samples = []
        self.performance_metrics = {}
        
    def start_profiling(self):
        """Start performance profiling session."""
        self.start_time = time.perf_counter()
        self.baseline_memory = self.process.memory_info().rss / (1024**3)  # GB
        gc.collect()  # Clear any existing garbage
        logger.info(f"🚀 Performance profiling started - Baseline memory: {self.baseline_memory:.2f}GB")
        
    def stage_start(self, stage_name):
        """Mark the start of a performance stage."""
        self.stage_times[stage_name] = {
            'start': time.perf_counter(),
            'memory_start': self.process.memory_info().rss / (1024**3)
        }
        logger.info(f"📊 Stage '{stage_name}' started")
        
    def stage_end(self, stage_name):
        """Mark the end of a performance stage and log metrics."""
        if stage_name in self.stage_times:
            end_time = time.perf_counter()
            end_memory = self.process.memory_info().rss / (1024**3)
            
            duration = end_time - self.stage_times[stage_name]['start']
            memory_delta = end_memory - self.stage_times[stage_name]['memory_start']
            
            self.stage_times[stage_name].update({
                'end': end_time,
                'duration': duration,
                'memory_end': end_memory,
                'memory_delta': memory_delta
            })
            
            logger.info(f"✅ Stage '{stage_name}' completed: {duration:.3f}s, "
                       f"Memory: {end_memory:.2f}GB ({memory_delta:+.2f}GB)")
            
    def sample_memory(self):
        """Sample current memory usage."""
        current_memory = self.process.memory_info().rss / (1024**3)
        self.memory_samples.append({
            'timestamp': time.perf_counter() - self.start_time,
            'memory_gb': current_memory
        })
        return current_memory
        
    def end_profiling(self):
        """End profiling session and generate report."""
        if self.start_time:
            total_time = time.perf_counter() - self.start_time
            final_memory = self.process.memory_info().rss / (1024**3)
            total_memory_delta = final_memory - self.baseline_memory
            
            self.performance_metrics = {
                'total_time': total_time,
                'baseline_memory': self.baseline_memory,
                'final_memory': final_memory,
                'total_memory_delta': total_memory_delta,
                'peak_memory': max(sample['memory_gb'] for sample in self.memory_samples) if self.memory_samples else final_memory
            }
            
            logger.info(f"🏁 Performance profiling completed:")
            logger.info(f"   Total time: {total_time:.2f}s")
            logger.info(f"   Memory change: {self.baseline_memory:.2f}GB → {final_memory:.2f}GB ({total_memory_delta:+.2f}GB)")
            logger.info(f"   Peak memory: {self.performance_metrics['peak_memory']:.2f}GB")
            
    def generate_report(self):
        """Generate detailed performance report."""
        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Overall metrics
        if self.performance_metrics:
            report.append(f"Total execution time: {self.performance_metrics['total_time']:.3f} seconds")
            report.append(f"Memory usage: {self.performance_metrics['baseline_memory']:.2f}GB → "
                         f"{self.performance_metrics['final_memory']:.2f}GB "
                         f"({self.performance_metrics['total_memory_delta']:+.2f}GB change)")
            report.append(f"Peak memory: {self.performance_metrics['peak_memory']:.2f}GB")
            report.append("")
        
        # Stage-by-stage breakdown
        report.append("STAGE PERFORMANCE BREAKDOWN:")
        report.append("-" * 40)
        
        total_stage_time = 0
        for stage_name, stage_data in self.stage_times.items():
            if 'duration' in stage_data:
                duration = stage_data['duration']
                memory_delta = stage_data['memory_delta']
                total_stage_time += duration
                
                report.append(f"{stage_name:25} | {duration:8.3f}s | {memory_delta:+8.2f}GB")
        
        if total_stage_time > 0 and self.performance_metrics:
            overhead = self.performance_metrics['total_time'] - total_stage_time
            report.append(f"{'OVERHEAD':25} | {overhead:8.3f}s | {'':8}")
        
        # Memory sampling analysis
        if self.memory_samples:
            report.append("")
            report.append("MEMORY USAGE PATTERN:")
            report.append("-" * 40)
            
            # Find memory trends
            memory_values = [s['memory_gb'] for s in self.memory_samples]
            if len(memory_values) > 1:
                memory_trend = "increasing" if memory_values[-1] > memory_values[0] else "decreasing" if memory_values[-1] < memory_values[0] else "stable"
                memory_variance = np.std(memory_values)
                report.append(f"Memory trend: {memory_trend}")
                report.append(f"Memory variance: {memory_variance:.3f}GB")
                report.append(f"Memory samples: {len(memory_values)}")
        
        report.append("=" * 80)
        
        # Log the report
        for line in report:
            logger.info(line)
            
        return "\n".join(report)

def test_data_preparation_performance():
    """Run comprehensive performance test of data_preparation optimizations."""
    
    # Validate paths
    if not os.path.exists(EGMD_DATASET_DIR):
        logger.error(f"❌ E-GMD dataset directory not found: {EGMD_DATASET_DIR}")
        logger.error("Please update EGMD_DATASET_DIR in the script configuration.")
        return False
        
    if not os.path.exists(PROCESSED_OUTPUT_DIR):
        logger.info(f"📁 Creating output directory: {PROCESSED_OUTPUT_DIR}")
        os.makedirs(PROCESSED_OUTPUT_DIR)
    
    # Initialize profiler
    profiler = PerformanceProfiler()
    profiler.start_profiling()
    
    try:
        # Stage 1: Initialization
        profiler.stage_start("initialization")
        logger.info(f"🔧 Initializing data_preparation with sample_ratio={SAMPLE_RATIO}")
        
        data_prep = data_preparation(
            directory_path=EGMD_DATASET_DIR,
            dataset='egmd',
            sample_ratio=SAMPLE_RATIO,
            diff_threshold=1.0
        )
        
        profiler.stage_end("initialization")
        profiler.sample_memory()
        
        # Log dataset info
        total_pairs = len(data_prep.midi_wav_map)
        logger.info(f"📈 Dataset loaded: {total_pairs} MIDI/audio pairs")
        
        if total_pairs == 0:
            logger.error("❌ No valid MIDI/audio pairs found!")
            return False
            
        # Stage 2: Audio Set Creation (Main Performance Test)
        profiler.stage_start("audio_set_creation")
        logger.info(f"🎵 Creating audio set with optimized pipeline...")
        logger.info(f"   Memory limit: {MEMORY_LIMIT_GB}GB")
        logger.info(f"   Batch size multiplier: {BATCH_SIZE_MULTIPLIER}x")
        logger.info(f"   Batching enabled: True")
        
        # Sample memory during processing
        def memory_sampling_callback():
            profiler.sample_memory()
        
        # Run the main performance test
        processed_files = data_prep.create_audio_set(
            pad_before=0.02,
            pad_after=0.02,
            fix_length=5.0,  # Fixed 5-second slices for consistent testing
            batching=True,
            dir_path=PROCESSED_OUTPUT_DIR,
            num_batches=NUM_BATCHES,
            memory_limit_gb=MEMORY_LIMIT_GB,
            batch_size_multiplier=BATCH_SIZE_MULTIPLIER
        )
        
        profiler.stage_end("audio_set_creation")
        profiler.sample_memory()
        
        # Stage 3: Results Analysis
        profiler.stage_start("results_analysis")
        
        # Analyze output files
        output_files = list(Path(PROCESSED_OUTPUT_DIR).glob("batch_*.pkl"))
        total_batches_created = len(output_files)
        
        total_file_size_mb = sum(f.stat().st_size for f in output_files) / (1024**2)
        avg_batch_size_mb = total_file_size_mb / max(1, total_batches_created)
        
        logger.info(f"📊 Processing Results:")
        logger.info(f"   Files processed: {processed_files}/{total_pairs}")
        logger.info(f"   Success rate: {(processed_files/total_pairs)*100:.1f}%")
        logger.info(f"   Batches created: {total_batches_created}")
        logger.info(f"   Total output size: {total_file_size_mb:.1f}MB")
        logger.info(f"   Average batch size: {avg_batch_size_mb:.1f}MB")
        
        # Performance metrics calculation
        if processed_files > 0 and profiler.stage_times.get('audio_set_creation', {}).get('duration'):
            processing_time = profiler.stage_times['audio_set_creation']['duration']
            files_per_second = processed_files / processing_time
            time_per_file = processing_time / processed_files
            
            logger.info(f"⚡ Performance Metrics:")
            logger.info(f"   Processing speed: {files_per_second:.2f} files/second")
            logger.info(f"   Time per file: {time_per_file:.3f} seconds")
            logger.info(f"   Total processing time: {processing_time:.2f} seconds")
        
        profiler.stage_end("results_analysis")
        
        # Stage 4: Memory efficiency analysis
        profiler.stage_start("memory_efficiency_analysis")
        
        # Estimate compression efficiency if we can load a sample batch
        if output_files:
            try:
                sample_batch_path = output_files[0]
                sample_batch = pd.read_pickle(sample_batch_path)
                
                if not sample_batch.empty and 'audio_wav' in sample_batch.columns:
                    # Analyze compression types
                    compressed_count = 0
                    uncompressed_count = 0
                    flac_count = 0
                    float16_count = 0
                    
                    for audio_data in sample_batch['audio_wav'].head(10):  # Sample first 10
                        if isinstance(audio_data, dict) and 'compressed_audio' in audio_data:
                            compressed_count += 1
                            if audio_data.get('format') == 'FLAC':
                                flac_count += 1
                        elif isinstance(audio_data, np.ndarray):
                            if audio_data.dtype == np.float16:
                                float16_count += 1
                            else:
                                uncompressed_count += 1
                    
                    logger.info(f"🗜️  Compression Analysis (sample of 10):")
                    logger.info(f"   FLAC compressed: {flac_count}")
                    logger.info(f"   Float16 compressed: {float16_count}")
                    logger.info(f"   Uncompressed: {uncompressed_count}")
                    logger.info(f"   Total records in sample batch: {len(sample_batch)}")
                    
            except Exception as e:
                logger.warning(f"Could not analyze sample batch: {e}")
        
        profiler.stage_end("memory_efficiency_analysis")
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}", exc_info=True)
        return False
        
    finally:
        # Always end profiling and generate report
        profiler.end_profiling()
        
        # Generate and save detailed report
        report = profiler.generate_report()
        report_file = f"performance_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Detailed performance report saved to: {report_file}")
    
    return True

def main():
    """Main test runner."""
    logger.info("🧪 Starting Data Preparation Performance Analysis")
    logger.info(f"📂 Dataset: {EGMD_DATASET_DIR}")
    logger.info(f"📂 Output: {PROCESSED_OUTPUT_DIR}")
    logger.info(f"🔬 Sample ratio: {SAMPLE_RATIO} ({SAMPLE_RATIO*100:.1f}% of dataset)")
    logger.info(f"💾 Memory limit: {MEMORY_LIMIT_GB}GB")
    
    # System info
    system_memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = os.cpu_count()
    logger.info(f"💻 System: {system_memory_gb:.1f}GB RAM, {cpu_count} CPU cores")
    
    success = test_data_preparation_performance()
    
    if success:
        logger.info("✅ Performance analysis completed successfully!")
    else:
        logger.error("❌ Performance analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
