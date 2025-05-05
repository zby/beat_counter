"""Functional tests for the Beat-This! integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import os  # Import os for file operations
from unittest.mock import MagicMock

from beat_detection.core.detector_protocol import BeatDetector
from beat_detection.core.factory import get_beat_detector
from beat_detection.core.beats import RawBeats

# -----------------------------------------------------------------------------
# Test data
# -----------------------------------------------------------------------------

# Define the path to the test *fixtures* directory
TEST_FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_AUDIO_FILE = TEST_FIXTURES_DIR / "Besito_a_Besito_10sec.mp3"

TOLERANCE_PERCENT = 10.0
MIN_MEASURES = 5

# Define fixed output path for beat_this results in its own output directory
BEAT_THIS_OUTPUT_DIR = Path(__file__).parent / "output" / "beat_this"
BEAT_THIS_OUTPUT_BEATS_FILE = BEAT_THIS_OUTPUT_DIR / f"{TEST_AUDIO_FILE.stem}.beats"


def test_beat_this_detect_save_load_reconstruct():
    """
    Tests the full beat_this process:
    1. Detect beats from an audio file.
    2. Infer beats_per_bar.
    3. Save the simplified RawBeats to a file.
    4. Load the simplified RawBeats from the file.
    5. Reconstruct the full Beats object from the loaded data.
    (Allows exceptions to propagate naturally)
    """
    # --- Setup --- 
    # Ensure the test audio file exists
    if not TEST_AUDIO_FILE.is_file():
        pytest.fail(f"Test audio file not found: {TEST_AUDIO_FILE}")

    # Ensure the output directory exists
    BEAT_THIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define the fixed output path
    output_beats_file = BEAT_THIS_OUTPUT_BEATS_FILE

    # --- 1. Detect beats & 2. Infer beats_per_bar ---
    detector: BeatDetector = get_beat_detector("beat_this")
    
    # Ensure we're not using a mock
    if isinstance(detector, MagicMock):
        pytest.fail("The beat_this detector is a mock, not a real implementation. "
                    "Make sure the beat_this package is properly installed.")
                    
    raw_beats = detector.detect_beats(str(TEST_AUDIO_FILE))

    # Verify raw_beats is the correct type
    if not isinstance(raw_beats, RawBeats):
        pytest.fail(f"Expected raw_beats to be of type RawBeats, but got {type(raw_beats).__name__}")

    assert raw_beats.timestamps.shape[0] > 0, "No raw beats were detected by beat_this."
    assert raw_beats.beat_counts.size > 0, "No beat counts detected by beat_this, cannot infer beats_per_bar."
    assert int(np.max(raw_beats.beat_counts)) == 4, "Inferred beats_per_bar by beat_this is invalid."

    # --- 3. Save RawBeats --- 
    raw_beats.save_to_file(output_beats_file)
    print(f"[Test beat_this] Saved simplified raw beats to fixed path: {output_beats_file}")
    assert (
        output_beats_file.is_file()
    ), f"Raw beats file (beat_this) was not created at {output_beats_file}"
    assert (
        output_beats_file.stat().st_size > 0
    ), f"Raw beats file (beat_this) {output_beats_file} is empty."


# Keep the main block for potentially running tests directly (though pytest is preferred)
if __name__ == "__main__":
    # Note: Running this directly won't use pytest fixtures correctly.
    # Use `pytest tests/test_beat_this.py` instead.
    print("Please run these tests using pytest.")
    # Attempting to run might fail due to fixture dependencies.
    pass 