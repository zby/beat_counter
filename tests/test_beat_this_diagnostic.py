"""Diagnostic version of beat_this tests to troubleshoot GitHub Actions failures."""

from __future__ import annotations

import sys
import os
import inspect
import traceback
from pathlib import Path

import numpy as np
import pytest

from beat_detection.core.detector_protocol import BeatDetector
from beat_detection.core.factory import get_beat_detector

# -----------------------------------------------------------------------------
# Test data with additional diagnostics
# -----------------------------------------------------------------------------

# Define the path to the test *fixtures* directory
TEST_FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_AUDIO_FILE = TEST_FIXTURES_DIR / "Besito_a_Besito_10sec.mp3"

TOLERANCE_PERCENT = 10.0
MIN_MEASURES = 5

# Define fixed output path for beat_this results in its own output directory
BEAT_THIS_OUTPUT_DIR = Path(__file__).parent / "output" / "beat_this"
BEAT_THIS_OUTPUT_BEATS_FILE = BEAT_THIS_OUTPUT_DIR / f"{TEST_AUDIO_FILE.stem}.beats"

def print_diagnostic(message):
    """Print diagnostic message with clear formatting."""
    print(f"\n[DIAGNOSTIC] {message}\n", file=sys.stderr, flush=True)

def test_beat_this_diagnostic():
    """Diagnostic version of the beat_this test to debug GitHub Actions failures."""
    try:
        # --- Diagnose paths ---
        print_diagnostic(f"Current working directory: {os.getcwd()}")
        print_diagnostic(f"Test file location: {__file__}")
        print_diagnostic(f"TEST_FIXTURES_DIR: {TEST_FIXTURES_DIR}")
        print_diagnostic(f"TEST_AUDIO_FILE: {TEST_AUDIO_FILE}")
        print_diagnostic(f"TEST_AUDIO_FILE exists: {TEST_AUDIO_FILE.is_file()}")
        
        # --- Look for alternative test files ---
        mp3_files = list(Path(os.getcwd()).glob("**/*.mp3"))
        print_diagnostic(f"Found MP3 files in workspace: {mp3_files}")
        print_diagnostic(f"Using alternative audio file: {TEST_AUDIO_FILE}")
        
        # --- Setup ---
        # Ensure the output directory exists
        BEAT_THIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # --- Diagnose detector ---
        print_diagnostic("Initializing beat_this detector...")
        detector: BeatDetector = get_beat_detector("beat_this")
        print_diagnostic(f"Detector type: {type(detector)}")
        print_diagnostic(f"Detector methods: {dir(detector)}")
        print_diagnostic(f"detect_beats method: {inspect.signature(detector.detect_beats)}")
        
        # --- Run detector with safeguards ---
        print_diagnostic(f"Detecting beats in file: {TEST_AUDIO_FILE}")
        raw_beats = detector.detect_beats(str(TEST_AUDIO_FILE))
        print_diagnostic(f"Raw beats type: {type(raw_beats)}")
        
        if raw_beats is None:
            print_diagnostic("ERROR: raw_beats is None")
            return
            
        # --- Diagnose raw_beats object ---
        print_diagnostic(f"Raw beats attributes: {dir(raw_beats)}")
        
        # Check if timestamps exists and what it actually is
        has_timestamps = hasattr(raw_beats, "timestamps")
        print_diagnostic(f"Has timestamps attribute: {has_timestamps}")
        
        if has_timestamps:
            timestamps = raw_beats.timestamps
            print_diagnostic(f"Timestamps type: {type(timestamps)}")
            print_diagnostic(f"Timestamps repr: {repr(timestamps)}")
            
            # If it's a MagicMock, it won't have shape attribute that works with '>'
            if hasattr(timestamps, "shape"):
                print_diagnostic(f"Timestamps shape: {timestamps.shape}")
                print_diagnostic(f"Timestamps shape[0]: {timestamps.shape[0]}")
            else:
                print_diagnostic("Timestamps doesn't have shape attribute")
        
        # Check beat_counts in a similar way
        has_beat_counts = hasattr(raw_beats, "beat_counts")
        print_diagnostic(f"Has beat_counts attribute: {has_beat_counts}")
        
        if has_beat_counts:
            beat_counts = raw_beats.beat_counts
            print_diagnostic(f"Beat counts type: {type(beat_counts)}")
            print_diagnostic(f"Beat counts repr: {repr(beat_counts)}")
            
            if hasattr(beat_counts, "size"):
                print_diagnostic(f"Beat counts size: {beat_counts.size}")
            else:
                print_diagnostic("Beat counts doesn't have size attribute")
        
    except Exception as e:
        print_diagnostic(f"Exception during test: {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)
        raise
        
if __name__ == "__main__":
    # Run the diagnostic test directly
    test_beat_this_diagnostic() 