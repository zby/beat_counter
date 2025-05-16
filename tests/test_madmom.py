import pytest
import os  # Import os for file operations
from pathlib import Path
from beat_detection.core import get_beat_detector
from beat_detection.core.beats import BeatCalculationError
import numpy as np

# Define the path to the test *fixtures* directory
TEST_FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_AUDIO_FILE = TEST_FIXTURES_DIR / "Besito_a_Besito_10sec.mp3"

# Define parameters used across tests
TEST_TOLERANCE_PERCENT = 10.0
TEST_MIN_MEASURES = 5
EXPECTED_DURATION = 10.0  # Known duration of the test file

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
    4. Verify clip_length is correctly detected and stored.
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
    detector = get_beat_detector("madmom")
    raw_beats = detector.detect_beats(str(TEST_AUDIO_FILE))

    assert raw_beats is not None, "Simplified RawBeats object was not created by madmom."
    assert raw_beats.timestamps.shape[0] > 0, "No raw beats were detected by madmom."

    # Verify clip_length
    assert hasattr(raw_beats, 'clip_length'), "RawBeats object missing clip_length attribute"
    assert raw_beats.clip_length > 0, f"Invalid clip_length: {raw_beats.clip_length}"
    assert np.isclose(raw_beats.clip_length, EXPECTED_DURATION, rtol=0.01), \
        f"Detected clip_length ({raw_beats.clip_length}) differs from expected duration ({EXPECTED_DURATION})"
    
    # Verify timestamps don't exceed clip_length
    assert np.all(raw_beats.timestamps <= raw_beats.clip_length), \
        f"Some timestamps exceed clip_length: max timestamp {np.max(raw_beats.timestamps)} > clip_length {raw_beats.clip_length}"

    # Verify the last beat is in the latter part of the clip (past 9 seconds)
    assert raw_beats.timestamps[-1] > 9.0, \
        f"Last beat timestamp ({raw_beats.timestamps[-1]}) is too early - should be past 9 seconds"

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

    # --- 4. Load and verify clip_length is preserved ---
    loaded_beats = raw_beats.__class__.load_from_file(output_beats_file)
    assert np.isclose(loaded_beats.clip_length, EXPECTED_DURATION, rtol=0.01), \
        f"Loaded clip_length ({loaded_beats.clip_length}) differs from expected duration ({EXPECTED_DURATION})"


def test_madmom_invalid_audio_file():
    """Test that appropriate errors are raised for invalid audio files."""
    detector = get_beat_detector("madmom")
    
    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        detector.detect_beats("nonexistent.mp3")
    
    # Test empty file (create a temporary empty file)
    empty_file = TEST_FIXTURES_DIR / "empty.mp3"
    try:
        empty_file.touch()
        with pytest.raises(BeatCalculationError, match="Failed to get audio duration"):
            detector.detect_beats(str(empty_file))
    finally:
        if empty_file.exists():
            empty_file.unlink()


def test_madmom_detector_constructor():
    """Test that the detector can be instantiated with various parameters."""
    # Default constructor
    detector = get_beat_detector("madmom")
    assert detector is not None
    
    # With min/max BPM
    detector = get_beat_detector("madmom", min_bpm=90, max_bpm=180)
    assert detector.min_bpm == 90
    assert detector.max_bpm == 180
    
    # With custom fps
    detector = get_beat_detector("madmom", fps=200)
    assert detector.fps == 200
    
    # With custom beats_per_bar
    detector = get_beat_detector("madmom", beats_per_bar=[3, 4])
    assert detector.beats_per_bar == [3, 4]


# Keep the main block for potentially running tests directly (though pytest is preferred)
if __name__ == "__main__":
    # Note: Running this directly won't use pytest fixtures correctly.
    # Use `pytest tests/test_madmom.py` instead.
    print("Please run these tests using pytest.")
    # Attempting to run might fail due to fixture dependencies.
    pass
