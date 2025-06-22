# Chart-Hero Parallel Processing Bug - Summary

**Project**: chart-hero (audio/MIDI processing for drum chart generation)
**Issue**: Parallel data preparation pipeline fails with `'float' object has no attribute 'midi_filename'` error

## Problem Description
The E-GMD dataset preparation script (`prepare_egmd_data.py`) works fine in sequential mode but fails in parallel mode. During parallel processing, the worker function receives a list of dictionaries as expected, but when iterating over the chunk, it sometimes encounters `float` objects instead of dictionaries.

## Key Files
- `/Users/maple/Repos/chart-hero/prepare_egmd_data.py` - Main CLI script
- `/Users/maple/Repos/chart-hero/model_training/data_preparation.py` - Core data prep logic with chunking
- `/Users/maple/Repos/chart-hero/model_training/parallel_worker.py` - Parallel worker that processes chunks

## Current State
- Extensive debug logging has been added to `parallel_worker.py` (lines 68-83 show the debugging code)
- The chunking logic in `data_preparation.py` correctly creates lists of dictionaries
- The worker receives the chunk correctly as a list of dicts
- **Bug location**: In the iteration loop (`for i, row_dict in enumerate(file_chunk):`), the first element is a dict but subsequent iterations sometimes yield floats

## Reproduction
Run: `python prepare_egmd_data.py --workers 2` (parallel mode fails)
Run: `python prepare_egmd_data.py --workers 1` (sequential mode works)

## Next Steps Needed
1. **Root cause analysis**: Determine why floats appear during chunk iteration despite chunk being a list of dicts
2. **Fix implementation**: Ensure robust processing of only dict objects
3. **Validation**: Test parallel mode produces correct output without errors

## Key Debugging Question
Why does `enumerate(file_chunk)` sometimes yield floats when `file_chunk` is confirmed to be a list of dictionaries? This is the core mystery that needs solving.

## Technical Context
The bug appears to be in the `chunk_processor_worker` function in `parallel_worker.py`. Debug logs show:
- `file_chunk` is correctly received as a list of dictionaries
- Direct access to `file_chunk[0]` returns a dict as expected
- However, during `for i, row_dict in enumerate(file_chunk):` iteration, `row_dict` sometimes becomes a float

This suggests the issue may be related to how pandas DataFrames are being iterated over or how the chunk data structure is being modified during processing.
