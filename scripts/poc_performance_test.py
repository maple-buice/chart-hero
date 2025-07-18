import time
import warnings
from pathlib import Path

import librosa

# Suppress the UserWarning from librosa about PySoundFile
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


def process_hpss(audio_path: Path):
    """
    Loads an audio file and applies Harmonic-Percussive Source Separation (HPSS).
    Returns the time taken for the operation.
    """
    start_time = time.perf_counter()
    try:
        y, sr = librosa.load(audio_path, sr=None)
        _y_percussive = librosa.effects.percussive(y)
    except Exception as e:
        print(f"Error processing {audio_path} with HPSS: {e}")
        return float("inf")  # Return infinity on error
    end_time = time.perf_counter()
    return end_time - start_time


def process_transient_enhanced_spectrogram(audio_path: Path):
    """
    Loads an audio file and creates a transient-enhanced spectrogram.
    Returns the time taken for the operation.
    """
    start_time = time.perf_counter()
    try:
        y, sr = librosa.load(audio_path, sr=22050)  # Resample for speed

        # 1. Mel Spectrogram
        librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
        )

        # 2. Onset Strength Envelope
        librosa.onset.onset_detect(y=y, sr=sr, units="time", hop_length=512)

        # This is a simplified simulation. A real implementation would align the envelope
        # with the spectrogram frames and multiply. For a PoC, the core computation time is what matters.
        # The multiplication itself is negligible.

    except Exception as e:
        print(f"Error processing {audio_path} with Transient Enhancement: {e}")
        return float("inf")
    end_time = time.perf_counter()
    return end_time - start_time


def main():
    """
    Main function to run the performance benchmark.
    """
    test_files = [
        Path(
            "/Users/maple/Clone Hero/Songs/LEGO Rock Band/Elton John - Crocodile Rock/song.ogg"
        ),
        # Add other files here if needed
    ]

    print("--- Performance Benchmark: HPSS vs. Transient-Enhanced Spectrogram ---")

    for audio_file in test_files:
        if not audio_file.exists():
            print(f"\nWARNING: Test file not found at {audio_file}")
            continue

        print(f"\nTesting: {audio_file.name}")

        # --- HPSS Test ---
        hpss_duration = process_hpss(audio_file)
        print(
            f"  Harmonic-Percussive Source Separation (HPSS): {hpss_duration:.4f} seconds"
        )

        # --- Transient-Enhanced Spectrogram Test ---
        transient_duration = process_transient_enhanced_spectrogram(audio_file)
        print(
            f"  Transient-Enhanced Spectrogram:             {transient_duration:.4f} seconds"
        )

        # --- Conclusion ---
        if hpss_duration < 5:
            print(f"  -> Conclusion: HPSS is viable at {hpss_duration:.2f}s.")
        elif transient_duration < 5:
            print(
                f"  -> Conclusion: HPSS is too slow ({hpss_duration:.2f}s). Transient-Enhancement is viable at {transient_duration:.2f}s."
            )
        else:
            print(
                f"  -> Conclusion: Both methods are too slow. HPSS: {hpss_duration:.2f}s, Transient: {transient_duration:.2f}s"
            )


if __name__ == "__main__":
    main()
