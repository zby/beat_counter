"""
Tests for the registry module.

These tests verify the functionality of the beat detector registry and extraction pipeline.
"""
import pytest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

from beat_detection.core import get_beat_detector, extract_beats
from beat_detection.core.registry import _DETECTORS
from beat_detection.core.detector_protocol import BeatDetector
from beat_detection.core.beats import Beats, RawBeats
import numpy as np

# Additional tests for the registry that weren't covered in test_registry.py
def test_detector_registry_contents():
    """Verify that the detector registry contains expected implementations."""
    # The registry should have at least these two entries
    assert "madmom" in _DETECTORS
    assert "beat_this" in _DETECTORS

# Tests for extract_beats functionality (moved to pipeline module)
@patch("beat_detection.core.pipeline.get_detector")
def test_extract_beats_with_mocked_detector(mock_get_detector):
    """Test extract_beats with a mocked detector."""
    # Create a mock detector that returns a RawBeats object with enough beats for 5 measures (at least 20 beats)
    # Generate 20 regular beats with proper counts for 4/4 time
    timestamps = np.arange(0.5, 10.5, 0.5)  # 20 beats at 0.5s intervals
    # Generate counts that cycle through 1,2,3,4 (for 4/4 time)
    beat_counts = np.array([(i % 4) + 1 for i in range(20)])
    
    mock_detector = MagicMock(spec=BeatDetector)
    mock_detector.detect_beats.return_value = RawBeats(
        timestamps=timestamps,
        beat_counts=beat_counts,
        clip_length=10.5
    )
    mock_get_detector.return_value = mock_detector
    
    # Create a temporary file to use as the audio file
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
        # Call extract_beats
        result = extract_beats(
            audio_file_path=temp_file.name,
            algorithm="mock_detector"
        )
        
        # Check that get_detector was called with the correct algorithm
        mock_get_detector.assert_called_once_with(algorithm="mock_detector")
        
        # Check that detect_beats was called with the correct file path
        mock_detector.detect_beats.assert_called_once_with(temp_file.name)
        
        # Check that the result is a Beats object
        assert isinstance(result, Beats)
        assert result.beat_data.shape[0] == 20
        
        # Check that the .beats file was created
        beats_file = Path(temp_file.name).with_suffix(".beats")
        assert beats_file.exists()
        
        # Clean up
        beats_file.unlink()
        stats_file = Path(str(beats_file).replace(".beats", "._beat_stats"))
        if stats_file.exists():
            stats_file.unlink()
