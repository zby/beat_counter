"""
Tests for the beat detector factory.
"""
import pytest
from unittest.mock import patch, MagicMock
import os
import importlib  # Import importlib at the top
import numpy as np

from beat_detection.core.factory import get_beat_detector, DETECTOR_REGISTRY, extract_beats # Import extract_beats
from beat_detection.core.beat_this_detector import BeatThisDetector
from beat_detection.core.madmom_detector import MadmomBeatDetector
from beat_detection.core.beats import Beats
from beat_detection.core.beats import RawBeats


def test_get_beat_detector_default():
    """Test that the default detector is MadmomBeatDetector."""
    detector = get_beat_detector()
    assert isinstance(detector, MadmomBeatDetector)


def test_get_beat_detector_madmom():
    """Test getting a MadmomBeatDetector explicitly."""
    detector = get_beat_detector("madmom")
    assert isinstance(detector, MadmomBeatDetector)


def test_get_beat_detector_beat_this():
    """Test getting a BeatThisDetector."""
    detector = get_beat_detector("beat_this")
    assert isinstance(detector, BeatThisDetector)


def test_get_beat_detector_invalid():
    """Test that an invalid algorithm name raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported beat detection algorithm"):
        get_beat_detector("invalid_algorithm")


def test_get_beat_detector_kwargs():
    """Test that kwargs are passed to the detector constructor."""
    # Create a detector with a custom parameter
    detector = get_beat_detector("madmom", fps=200)
    assert detector.fps == 200


def test_detector_registry():
    """Test that the detector registry contains the expected detectors."""
    assert "madmom" in DETECTOR_REGISTRY
    assert "beat_this" in DETECTOR_REGISTRY
    assert DETECTOR_REGISTRY["madmom"] == MadmomBeatDetector
    assert DETECTOR_REGISTRY["beat_this"] == BeatThisDetector


# --- Tests for extract_beats ---

@pytest.fixture
def mock_audio_file(tmp_path):
    """Create a dummy audio file."""
    audio_path = tmp_path / "test_audio.wav"
    audio_path.touch()
    return str(audio_path)

# Helper function to create test RawBeats objects
def create_test_raw_beats(timestamps=None, beat_counts=None, clip_length=None) -> RawBeats:
    """Creates a RawBeats object with provided data or default test data."""
    if timestamps is None or beat_counts is None:
        # Create a sequence of 20 beats with regular 0.5s intervals
        # Use a 4/4 time signature pattern (1,2,3,4,1,2,3,4,...)
        num_beats = 20
        interval = 0.5
        beats_per_bar = 4
        
        timestamps = np.arange(num_beats) * interval
        beat_counts = np.array([(i % beats_per_bar) + 1 for i in range(num_beats)])
    else:
        timestamps = np.array(timestamps, dtype=float)
        beat_counts = np.array(beat_counts, dtype=int)
    
    # Default clip_length to last timestamp + interval if not provided
    if clip_length is None:
        clip_length = float(timestamps[-1] + 0.5) if len(timestamps) > 0 else 10.0
        
    return RawBeats(
        timestamps=timestamps,
        beat_counts=beat_counts,
        clip_length=clip_length
    )

@pytest.fixture
def mock_extract_dependencies():
    """Fixture to prepare dependencies for extract_beats tests using real RawBeats."""
    mock_detector_instance = MagicMock()
    
    # Create a real RawBeats object with enough beats to pass validation
    raw_beats = create_test_raw_beats()  # Use the default 20 beats
    
    mock_detector_instance.detect_beats.return_value = raw_beats

    # Only patch the get_beat_detector function and RawBeats.save_to_file method
    with patch('beat_detection.core.factory.get_beat_detector', return_value=mock_detector_instance) as mock_get_detector, \
         patch.object(RawBeats, 'save_to_file') as mock_save_to_file:
        
        # Use the real Beats class, no need to mock it
        yield {
            "mock_get_detector": mock_get_detector,
            "mock_detector_instance": mock_detector_instance,
            "raw_beats": raw_beats,
            "mock_save_to_file": mock_save_to_file
        }


def test_extract_beats_default_output(mock_extract_dependencies, tmp_path, mock_audio_file):
    """Test extract_beats with the default output path."""
    # Get dependencies from fixture
    mock_get_detector = mock_extract_dependencies["mock_get_detector"]
    mock_detector_instance = mock_extract_dependencies["mock_detector_instance"]
    raw_beats = mock_extract_dependencies["raw_beats"]
    mock_save_to_file = mock_extract_dependencies["mock_save_to_file"]

    # Call extract_beats with min_measures=1 to reduce validation requirements
    result = extract_beats(mock_audio_file, algorithm="mock_alg", beats_args={"min_measures": 1})

    # Assertions
    assert isinstance(result, Beats)
    assert result.beats_per_bar == 4  # From our test data
    assert result.min_measures == 1   # From our beats_args
    assert result.clip_length == raw_beats.clip_length  # Verify clip_length is preserved

    # Assert that save_to_file was called on the RawBeats object
    mock_save_to_file.assert_called_once()
    saved_path = mock_save_to_file.call_args[0][0]
    expected_output_path = tmp_path / "test_audio.beats"
    assert str(saved_path) == str(expected_output_path)

    # Verify detector interactions
    mock_get_detector.assert_called_once_with(algorithm="mock_alg")
    mock_detector_instance.detect_beats.assert_called_once_with(mock_audio_file)


def test_extract_beats_specified_output(mock_extract_dependencies, tmp_path, mock_audio_file):
    """Test extract_beats with a specified output path."""
    # Get dependencies from fixture
    mock_get_detector = mock_extract_dependencies["mock_get_detector"]
    mock_detector_instance = mock_extract_dependencies["mock_detector_instance"]
    raw_beats = mock_extract_dependencies["raw_beats"]
    mock_save_to_file = mock_extract_dependencies["mock_save_to_file"]

    # Set up test
    specified_output_path = tmp_path / "custom_output.txt"
    beats_arguments = {"tolerance_percent": 5.0, "min_measures": 1}  # Add min_measures=1

    # Call extract_beats
    result = extract_beats(
        mock_audio_file,
        output_path=str(specified_output_path),
        algorithm="mock_alg",
        beats_args=beats_arguments
    )

    # Assertions
    assert isinstance(result, Beats)
    assert result.tolerance_percent == 5.0  # From beats_arguments
    assert result.min_measures == 1         # From beats_arguments
    assert result.clip_length == raw_beats.clip_length  # Verify clip_length is preserved

    # Assert save_to_file called with specified path
    mock_save_to_file.assert_called_once_with(str(specified_output_path))

    # Assert detector calls
    mock_get_detector.assert_called_once_with(algorithm="mock_alg")
    mock_detector_instance.detect_beats.assert_called_once_with(mock_audio_file)


def test_extract_beats_passes_kwargs_to_detector(mock_extract_dependencies, tmp_path, mock_audio_file):
    """Test that extra kwargs are passed to get_beat_detector."""
    # Get dependencies from fixture
    mock_get_detector = mock_extract_dependencies["mock_get_detector"]
    mock_detector_instance = mock_extract_dependencies["mock_detector_instance"]
    raw_beats = mock_extract_dependencies["raw_beats"]
    mock_save_to_file = mock_extract_dependencies["mock_save_to_file"]

    # Set specific test parameters
    detector_kwargs = {"fps": 100, "extra_arg": "test"}

    # Call extract_beats with min_measures=1
    result = extract_beats(
        mock_audio_file, 
        algorithm="mock_alg", 
        beats_args={"min_measures": 1}, 
        **detector_kwargs
    )

    # Assertions
    assert isinstance(result, Beats)
    assert result.min_measures == 1  # From beats_args
    assert result.clip_length == raw_beats.clip_length  # Verify clip_length is preserved

    # Assert save_to_file was called
    mock_save_to_file.assert_called_once()

    # Assert that kwargs were passed to get_beat_detector
    mock_get_detector.assert_called_once_with(algorithm="mock_alg", **detector_kwargs)
    mock_detector_instance.detect_beats.assert_called_once_with(mock_audio_file)
