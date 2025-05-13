"""Functional tests for the Beat-This! integration."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import numpy as np
import pytest
import os
from unittest.mock import MagicMock

from beat_detection.core.detector_protocol import BeatDetector
from beat_detection.core.factory import get_beat_detector, extract_beats
from beat_detection.core.beats import RawBeats, Beats
from beat_detection.genre_db import GenreDB

# -----------------------------------------------------------------------------
# Test data
# -----------------------------------------------------------------------------

# Define the path to the test *fixtures* directory
TEST_FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_AUDIO_FILE = TEST_FIXTURES_DIR / "Besito_a_Besito_10sec.mp3"
POLKA_AUDIO_FILE = TEST_FIXTURES_DIR / "bavarian-beer-fest-20sec.mp3"

# Known durations of test files
TEST_AUDIO_DURATION = 10.0  # Besito_a_Besito_10sec.mp3
POLKA_AUDIO_DURATION = 20.0  # bavarian-beer-fest-20sec.mp3

TOLERANCE_PERCENT = 10.0
MIN_MEASURES = 5

# Define fixed output path for beat_this results in its own output directory
BEAT_THIS_OUTPUT_DIR = Path(__file__).parent / "output" / "beat_this"
BEAT_THIS_OUTPUT_BEATS_FILE = BEAT_THIS_OUTPUT_DIR / f"{TEST_AUDIO_FILE.stem}.beats"
POLKA_OUTPUT_BEATS_FILE = BEAT_THIS_OUTPUT_DIR / f"{POLKA_AUDIO_FILE.stem}.beats"

# Define genre data for polka music
POLKA_GENRE_CSV = """name,beats_per_bar,bpm_range
Polka,2,120-160
"""

def run_beat_this_detect_save_load_reconstruct():
    """
    Runs the full beat_this process:
    1. Detect beats from an audio file.
    2. Infer beats_per_bar.
    3. Save the simplified RawBeats to a file.
    4. Load the simplified RawBeats from the file.
    5. Reconstruct the full Beats object from the loaded data.
    """
    print("\nRunning beat_this detect, save, load, reconstruct test...")
    
    # --- Setup --- 
    # Ensure the test audio file exists
    assert TEST_AUDIO_FILE.is_file(), f"Test audio file not found: {TEST_AUDIO_FILE}"

    # Ensure the output directory exists
    BEAT_THIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define the fixed output path
    output_beats_file = BEAT_THIS_OUTPUT_BEATS_FILE

    # --- 1. Detect beats & 2. Infer beats_per_bar ---
    detector: BeatDetector = get_beat_detector("beat_this")
    
    # Ensure we're not using a mock
    assert not isinstance(detector, MagicMock), (
        "The beat_this detector is a mock, not a real implementation. "
        "Make sure the beat_this package is properly installed."
    )
                    
    raw_beats = detector.detect_beats(str(TEST_AUDIO_FILE))

    # Verify raw_beats is the correct type and has expected properties
    assert isinstance(raw_beats, RawBeats), f"Expected raw_beats to be of type RawBeats, but got {type(raw_beats).__name__}"
    assert raw_beats.timestamps.shape[0] > 0, "No raw beats were detected by beat_this."
    assert raw_beats.beat_counts.size > 0, "No beat counts detected by beat_this, cannot infer beats_per_bar."
    assert int(np.max(raw_beats.beat_counts)) == 4, "Inferred beats_per_bar by beat_this is invalid."

    # Verify clip_length
    assert hasattr(raw_beats, 'clip_length'), "RawBeats object missing clip_length attribute"
    assert raw_beats.clip_length > 0, f"Invalid clip_length: {raw_beats.clip_length}"
    assert np.isclose(raw_beats.clip_length, TEST_AUDIO_DURATION, rtol=0.01), \
        f"Detected clip_length ({raw_beats.clip_length}) differs from expected duration ({TEST_AUDIO_DURATION})"
    
    # Verify timestamps don't exceed clip_length
    assert np.all(raw_beats.timestamps <= raw_beats.clip_length), \
        f"Some timestamps exceed clip_length: max timestamp {np.max(raw_beats.timestamps)} > clip_length {raw_beats.clip_length}"

    # --- 3. Save RawBeats --- 
    raw_beats.save_to_file(output_beats_file)
    print(f"[Test beat_this] Saved simplified raw beats to fixed path: {output_beats_file}")
    
    assert output_beats_file.is_file(), f"Raw beats file (beat_this) was not created at {output_beats_file}"
    assert output_beats_file.stat().st_size > 0, f"Raw beats file (beat_this) {output_beats_file} is empty."

    # --- 4. Load and verify clip_length is preserved ---
    loaded_beats = raw_beats.__class__.load_from_file(output_beats_file)
    assert np.isclose(loaded_beats.clip_length, TEST_AUDIO_DURATION, rtol=0.01), \
        f"Loaded clip_length ({loaded_beats.clip_length}) differs from expected duration ({TEST_AUDIO_DURATION})"
    
    print("✓ beat_this detect, save, load, reconstruct test passed")

def run_beat_this_polka_beats_per_bar():
    """
    Runs the beat_this process with polka music to verify correct beats_per_bar:
    1. Detect beats from the polka audio file using genre-specific parameters
    2. Specifically verify that beats_per_bar is correctly identified for polka music
    """
    print("\nRunning beat_this polka beats_per_bar test...")
    
    # --- Setup --- 
    # Ensure the test audio file exists
    assert POLKA_AUDIO_FILE.is_file(), f"Test audio file not found: {POLKA_AUDIO_FILE}"

    # Ensure the output directory exists
    BEAT_THIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy the polka file to the output directory
    polka_copy_path = BEAT_THIS_OUTPUT_DIR / POLKA_AUDIO_FILE.name
    shutil.copy2(POLKA_AUDIO_FILE, polka_copy_path)
    print(f"[Test beat_this] Copied polka audio file to: {polka_copy_path}")
    
    # Verify the file was copied correctly
    assert polka_copy_path.is_file(), f"Failed to copy polka file to {polka_copy_path}"
    assert polka_copy_path.stat().st_size == POLKA_AUDIO_FILE.stat().st_size, "File size mismatch after copy"

    # Set up genre database with polka parameters
    genre_db = GenreDB(csv_content=POLKA_GENRE_CSV)
    
    # Get detector parameters for polka music
    detector_kwargs = genre_db.detector_kwargs_for_genre("Polka")
    assert detector_kwargs["beats_per_bar"] == [2], f"Expected beats_per_bar to be [2], got {detector_kwargs['beats_per_bar']}"

    # Get Beats constructor arguments for polka music
    beats_args = genre_db.beats_kwargs_for_genre("Polka")
    
    # --- 1. Detect beats with extract_beats using genre-specific parameters ---
    beats = extract_beats(
        audio_file_path=str(polka_copy_path),  # Use the copied file
        algorithm="beat_this",
        beats_args=beats_args,
        **detector_kwargs
    )
    
    assert isinstance(beats, Beats), f"Expected beats to be of type Beats, got {type(beats).__name__}"
    assert beats.beats_per_bar == 2, f"Expected beats_per_bar to be 2, got {beats.beats_per_bar}"
    assert beats.overall_stats.total_beats > 20, f"Expected more than 40 beats, got {beats.overall_stats.total_beats}"
    
    # Verify clip_length
    assert np.isclose(beats.clip_length, POLKA_AUDIO_DURATION, rtol=0.01), \
        f"Detected clip_length ({beats.clip_length}) differs from expected duration ({POLKA_AUDIO_DURATION})"
    
    # Verify timestamps don't exceed clip_length
    assert np.all(beats.timestamps <= beats.clip_length), \
        f"Some timestamps exceed clip_length: max timestamp {np.max(beats.timestamps)} > clip_length {beats.clip_length}"
    
    print("✓ beat_this polka beats_per_bar test passed")

# Pytest test functions that call the run functions
def test_beat_this_detect_save_load_reconstruct():
    """Pytest wrapper for run_beat_this_detect_save_load_reconstruct."""
    run_beat_this_detect_save_load_reconstruct()

def test_beat_this_polka_beats_per_bar():
    """Pytest wrapper for run_beat_this_polka_beats_per_bar."""
    run_beat_this_polka_beats_per_bar()

def main():
    """Run all tests directly when script is executed."""
    tests = [
        run_beat_this_detect_save_load_reconstruct,
        run_beat_this_polka_beats_per_bar
    ]
    
    failed_tests = []
    
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"\n❌ Test {test.__name__} failed:")
            print(f"Assertion failed: {str(e)}")
            failed_tests.append(test.__name__)
        except Exception as e:
            print(f"\n❌ Test {test.__name__} failed with unexpected error:")
            print(f"Error: {str(e)}")
            failed_tests.append(test.__name__)
    
    if failed_tests:
        print(f"\n❌ {len(failed_tests)} tests failed: {', '.join(failed_tests)}")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main() 