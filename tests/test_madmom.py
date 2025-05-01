import pytest
import os  # Import os for file operations
from pathlib import Path
from beat_detection.core.detector import BeatDetector
from beat_detection.core.factory import get_beat_detector
import numpy as np

# Define the path to the test *fixtures* directory
TEST_FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_AUDIO_FILE = TEST_FIXTURES_DIR / "Besito_a_Besito_10sec.mp3"

# Define parameters used across tests
TEST_TOLERANCE_PERCENT = 10.0
TEST_MIN_MEASURES = 5

# Define fixed output path for madmom results in the output directory
MADMOM_OUTPUT_DIR = Path(__file__).parent / "output" / "madmom"
MADMOM_OUTPUT_BEATS_FILE = MADMOM_OUTPUT_DIR / f"{TEST_AUDIO_FILE.stem}.beats"

# Tests
# -----------------------------------------------------------------------------

def test_madmom_detect_save_load_reconstruct():
    """
    Tests the full madmom process:
    1. Detect beats from an audio file.
    2. Infer beats_per_bar.
    3. Save the simplified RawBeats to a file.
    (Allows exceptions to propagate naturally)
    """
    # --- Setup --- 
    # Ensure the test audio file exists
    if not TEST_AUDIO_FILE.is_file():
        assert False, f"Test audio file not found: {TEST_AUDIO_FILE}"

    # Ensure the output directory exists
    MADMOM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define the fixed output path
    output_beats_file = MADMOM_OUTPUT_BEATS_FILE

    # --- 1. Detect beats & 2. Infer beats_per_bar --- 
    detector: BeatDetector = get_beat_detector("madmom")
    raw_beats = detector.detect(str(TEST_AUDIO_FILE))

    assert raw_beats is not None, "Simplified RawBeats object was not created by madmom."
    assert raw_beats.timestamps.shape[0] > 0, "No raw beats were detected by madmom."

    # Infer beats_per_bar and assert directly
    assert raw_beats.beat_counts.size > 0, "No beat counts detected by madmom, cannot infer beats_per_bar."
    inferred_beats_per_bar = int(np.max(raw_beats.beat_counts[raw_beats.beat_counts > 0]))
    assert inferred_beats_per_bar > 0, "Inferred beats_per_bar by madmom is invalid."
    # Add specific check if known (e.g., assert inferred_beats_per_bar == 4)
    print(f"[Test madmom] Inferred beats_per_bar = {inferred_beats_per_bar}")

    # --- 3. Save RawBeats --- 
    raw_beats.save_to_file(output_beats_file)
    print(f"[Test madmom] Saved simplified raw beats to fixed path: {output_beats_file}")
    assert (
        output_beats_file.is_file()
    ), f"Raw beats file (madmom) was not created at {output_beats_file}"
    assert (
        output_beats_file.stat().st_size > 0
    ), f"Raw beats file (madmom) {output_beats_file} is empty."


# Keep the main block for potentially running tests directly (though pytest is preferred)
if __name__ == "__main__":
    # Note: Running this directly won't use pytest fixtures correctly.
    # Use `pytest tests/test_madmom.py` instead.
    print("Please run these tests using pytest.")
    # Attempting to run might fail due to fixture dependencies.
    pass
