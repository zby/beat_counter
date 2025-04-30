import pytest
import os
import numpy as np
from pathlib import Path

# Import the main functionality from beat_this
# If this fails, the test run will error out immediately.
from beat_this.inference import File2Beats

# Define the path to the test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_AUDIO_FILE = TEST_DATA_DIR / "Besito_a_Besito_10sec.mp3"

# --- beat_this Documentation --- #
# Based on inspection of beat_this.inference module:
#
# File2Beats:
#   - Initialization: File2Beats(checkpoint_path="final0", device="cpu", float16=False, dbn=False)
#       - Loads a pre-trained model (BeatThis) and a post-processor (Postprocessor).
#       - Downloads checkpoint if `checkpoint_path` is not local (e.g., "final0").
#       - `dbn=True` uses madmom DBN post-processing, `dbn=False` uses minimal thresholding.
#   - Call: tracker(audio_path: str) -> tuple[np.ndarray, np.ndarray]
#       - Loads audio using `beat_this.preprocessing.load_audio`.
#       - Resamples audio to 22050 Hz.
#       - Calculates LogMelSpectrogram.
#       - Runs inference using the BeatThis model (handles chunking for long files).
#       - Post-processes frame predictions to get beat/downbeat times.
#       - Returns a tuple: (downbeat_times, beat_times)
#       - Both elements are NumPy arrays containing timestamps in seconds.
#
# ----------------------------- #

@pytest.mark.skipif(not TEST_AUDIO_FILE.exists(), reason="Test audio file not found")
def test_beat_this_runs_and_output():
    """
    Tests if the beat_this tracker (File2Beats) runs on a sample audio file,
    produces the expected output format (tuple of numpy arrays),
    and contains valid timestamp data.
    """
    assert TEST_AUDIO_FILE.exists(), f"Test audio file missing: {TEST_AUDIO_FILE}"

    try:
        # Instantiate the File2Beats class (uses default "final0" checkpoint)
        tracker = File2Beats()

        # Call the tracker instance with the audio file path
        result = tracker(str(TEST_AUDIO_FILE))

        # 1. Check output type (should be a tuple)
        assert isinstance(result, tuple), f"Expected output to be a tuple, got {type(result)}"
        assert len(result) == 2, f"Expected tuple of length 2, got {len(result)}"

        downbeats, beats = result

        # 2. Check elements are NumPy arrays
        assert isinstance(downbeats, np.ndarray), f"Expected downbeats to be numpy array, got {type(downbeats)}"
        assert isinstance(beats, np.ndarray), f"Expected beats to be numpy array, got {type(beats)}"

        # 3. Check array contents (basic validation)
        assert beats.ndim == 1, f"Expected beats array to be 1D, got {beats.ndim}D"
        assert downbeats.ndim == 1, f"Expected downbeats array to be 1D, got {downbeats.ndim}D"

        # Check if arrays contain data (might be empty for very short/silent audio)
        if beats.size > 0:
            assert np.all(beats >= 0), "Beat timestamps should be non-negative"
            assert np.issubdtype(beats.dtype, np.floating), f"Expected beats dtype float, got {beats.dtype}"
            print(f"Detected {len(beats)} beats: {beats[:5]}...") # Optional: print info
        else:
            print("Warning: No beats detected.")

        if downbeats.size > 0:
            assert np.all(downbeats >= 0), "Downbeat timestamps should be non-negative"
            assert np.issubdtype(downbeats.dtype, np.floating), f"Expected downbeats dtype float, got {downbeats.dtype}"
            print(f"Detected {len(downbeats)} downbeats: {downbeats[:5]}...") # Optional: print info
        else:
            print("Warning: No downbeats detected.")

    except Exception as e:
        # Catch potential errors during model download or processing
        pytest.fail(f"beat_this tracker failed to run or produced invalid output: {e}")

# You might need to install additional dependencies required by beat_this
# (e.g., specific versions of numpy, scipy, torch, soxr)
# Check the beat_this repository's requirements.txt or setup.py for details. 