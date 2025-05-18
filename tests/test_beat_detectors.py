"""Functional tests for beat detectors."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import numpy as np
import pytest
import os
from unittest.mock import MagicMock
import torch
import logging

from beat_detection.core.registry import build
from beat_detection.core import extract_beats
from beat_detection.core.detector_protocol import BeatDetector
from beat_detection.core.beats import RawBeats, Beats, BeatCalculationError
from beat_detection.core.detectors.madmom import MADMOM_DEFAULT_FPS
from beat_detection.genre_db import GenreDB

# -----------------------------------------------------------------------------
# Test data and fixtures
# -----------------------------------------------------------------------------

# Define the path to the test fixtures directory
TEST_FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_AUDIO_FILE = TEST_FIXTURES_DIR / "Besito_a_Besito_10sec.mp3"
POLKA_AUDIO_FILE = TEST_FIXTURES_DIR / "bavarian-beer-fest-20sec.mp3"

# Known durations of test files
TEST_AUDIO_DURATION = 10.0  # Besito_a_Besito_10sec.mp3
POLKA_AUDIO_DURATION = 20.0  # bavarian-beer-fest-20sec.mp3

# Common test parameters
EXPECTED_BEATS_PER_BAR = 4
MIN_BEATS = 20
LAST_BEAT_THRESHOLD = 9.0

# Define genre data for polka music
POLKA_GENRE_CSV = """name,beats_per_bar,bpm_range
Polka,2,120-160
"""

# List of detectors to test
DETECTORS = ["beat_this", "madmom"]

# -----------------------------------------------------------------------------
# Test functions
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("detector_name", DETECTORS)
def test_detect_save_load_reconstruct(detector_name: str):
    """
    Tests the full beat detection process for a given detector:
    1. Detect beats from an audio file
    2. Infer beats_per_bar
    3. Save the simplified RawBeats to a file
    4. Load and verify the saved data
    """
    # --- Setup ---
    output_dir = Path(__file__).parent / "output" / detector_name
    output_beats_file = output_dir / f"{TEST_AUDIO_FILE.stem}.beats"
    
    # Ensure the test audio file exists
    assert TEST_AUDIO_FILE.is_file(), f"Test audio file not found: {TEST_AUDIO_FILE}"
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Detect beats & 2. Infer beats_per_bar ---
    detector: BeatDetector = build(detector_name)
    
    # Ensure we're not using a mock for beat_this
    if detector_name == "beat_this":
        assert not isinstance(detector, MagicMock), (
            "The beat_this detector is a mock, not a real implementation. "
            "Make sure the beat_this package is properly installed."
        )
    
    raw_beats = detector.detect_beats(str(TEST_AUDIO_FILE))
    
    # Print detector configuration for debugging
    print(f"\nDetector: {detector_name}")
    print(f"fps: {detector.cfg.fps}")
    print(f"beats_per_bar: {detector.cfg.beats_per_bar}")
    print(f"min_bpm: {detector.cfg.min_bpm}")
    print(f"max_bpm: {detector.cfg.max_bpm}")
    print(f"beats length: {len(raw_beats.timestamps)}")
    print(f"last beat: {raw_beats.timestamps[-1]}")
    print(f"downbeats length: {np.sum(raw_beats.beat_counts == 1)}")
    
    # Verify raw_beats properties
    assert isinstance(raw_beats, RawBeats), f"Expected raw_beats to be of type RawBeats, but got {type(raw_beats).__name__}"
    assert raw_beats.timestamps.shape[0] > 0, f"No raw beats were detected by {detector_name}"
    assert raw_beats.beat_counts.size > 0, f"No beat counts detected by {detector_name}, cannot infer beats_per_bar"
    
    # Verify clip_length
    assert hasattr(raw_beats, 'clip_length'), "RawBeats object missing clip_length attribute"
    assert raw_beats.clip_length > 0, f"Invalid clip_length: {raw_beats.clip_length}"
    assert np.isclose(raw_beats.clip_length, TEST_AUDIO_DURATION, rtol=0.01), \
        f"Detected clip_length ({raw_beats.clip_length}) differs from expected duration ({TEST_AUDIO_DURATION})"
    
    # Verify timestamps don't exceed clip_length
    assert np.all(raw_beats.timestamps <= raw_beats.clip_length), \
        f"Some timestamps exceed clip_length: max timestamp {np.max(raw_beats.timestamps)} > clip_length {raw_beats.clip_length}"
    
    # Verify last beat is after threshold
    assert raw_beats.timestamps[-1] > LAST_BEAT_THRESHOLD, \
        f"Last beat timestamp ({raw_beats.timestamps[-1]}) is too early - should be past {LAST_BEAT_THRESHOLD} seconds"
    
    # Verify beats_per_bar
    inferred_beats_per_bar = int(np.max(raw_beats.beat_counts[raw_beats.beat_counts > 0]))
    assert inferred_beats_per_bar == EXPECTED_BEATS_PER_BAR, \
        f"Inferred beats_per_bar ({inferred_beats_per_bar}) differs from expected ({EXPECTED_BEATS_PER_BAR})"
    
    # --- 3. Save RawBeats ---
    raw_beats.save_to_file(output_beats_file)
    print(f"[Test {detector_name}] Saved simplified raw beats to: {output_beats_file}")
    
    assert output_beats_file.is_file(), f"Raw beats file ({detector_name}) was not created at {output_beats_file}"
    assert output_beats_file.stat().st_size > 0, f"Raw beats file ({detector_name}) {output_beats_file} is empty"
    
    # --- 4. Load and verify ---
    loaded_beats = raw_beats.__class__.load_from_file(output_beats_file)
    assert np.isclose(loaded_beats.clip_length, TEST_AUDIO_DURATION, rtol=0.01), \
        f"Loaded clip_length ({loaded_beats.clip_length}) differs from expected duration ({TEST_AUDIO_DURATION})"

@pytest.mark.parametrize("detector_name", DETECTORS)
def test_invalid_audio_file(detector_name: str):
    """Test that appropriate errors are raised for invalid audio files."""
    detector = build(detector_name)
    
    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        detector.detect_beats("nonexistent.mp3")
    
    # Test empty file
    empty_file = TEST_FIXTURES_DIR / "empty.mp3"
    try:
        empty_file.touch()
        with pytest.raises(BeatCalculationError, match="Failed to get audio duration"):
            detector.detect_beats(str(empty_file))
    finally:
        if empty_file.exists():
            empty_file.unlink()

@pytest.mark.parametrize("detector_name", DETECTORS)
def test_detector_constructor(detector_name: str):
    """Test that the detector can be instantiated with various parameters."""
    # Default constructor
    detector = build(detector_name)
    assert detector is not None
    
    # With min/max BPM
    detector = build(detector_name, min_bpm=90, max_bpm=180)
    assert detector.cfg.min_bpm == 90
    assert detector.cfg.max_bpm == 180
    
    # With custom fps (if supported)
    if detector_name == "madmom":
        detector = build(detector_name, fps=200)
        assert detector.cfg.fps == 200
    
    # With custom beats_per_bar
    detector = build(detector_name, beats_per_bar=[3, 4])
    assert detector.cfg.beats_per_bar == [3, 4]

@pytest.mark.parametrize("detector_name", DETECTORS)
def test_polka_beats_per_bar(detector_name):
    """
    Tests detector with polka music to verify correct beats_per_bar detection.
    Uses genre-specific parameters for polka music which requires 2 beats per bar.
    """
    # --- Setup ---
    output_dir = Path(__file__).parent / "output" / detector_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the polka file to the output directory
    polka_copy_path = output_dir / POLKA_AUDIO_FILE.name
    shutil.copy2(POLKA_AUDIO_FILE, polka_copy_path)
    print(f"[Test {detector_name}] Copied polka audio file to: {polka_copy_path}")
    
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
    
    # Detect beats with genre-specific parameters
    beats = extract_beats(
        audio_file_path=str(polka_copy_path),
        detector_name=detector_name,
        beats_args=beats_args,
        **detector_kwargs
    )
    
    # Verify results
    assert isinstance(beats, Beats), f"Expected beats to be of type Beats, got {type(beats).__name__}"
    assert beats.beats_per_bar == 2, f"Expected beats_per_bar to be 2, got {beats.beats_per_bar}"
    assert beats.overall_stats.total_beats > 20, f"Expected more than 20 beats, got {beats.overall_stats.total_beats}"
    
    # Verify clip_length
    assert np.isclose(beats.clip_length, POLKA_AUDIO_DURATION, rtol=0.01), \
        f"Detected clip_length ({beats.clip_length}) differs from expected duration ({POLKA_AUDIO_DURATION})"
    
    # Verify timestamps don't exceed clip_length
    assert np.all(beats.timestamps <= beats.clip_length), \
        f"Some timestamps exceed clip_length: max timestamp {np.max(beats.timestamps)} > clip_length {beats.clip_length}"

if __name__ == "__main__":
    # Note: Running this directly won't use pytest fixtures correctly.
    # Use `pytest tests/test_beat_detectors.py`
    print("Please run these tests using pytest.") 