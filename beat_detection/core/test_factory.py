"""
Tests for the beat detector factory.
"""
import pytest
from unittest.mock import patch, MagicMock

from beat_detection.core.factory import get_beat_detector, DETECTOR_REGISTRY
from beat_detection.core.beat_this_detector import BeatThisDetector
from beat_detection.core.madmom_detector import MadmomBeatDetector


def test_get_beat_detector_default():
    """Test that the default detector is MadmomBeatDetector."""
    detector = get_beat_detector()
    assert isinstance(detector, MadmomBeatDetector)


def test_get_beat_detector_madmom():
    """Test getting a MadmomBeatDetector explicitly."""
    detector = get_beat_detector("madmom")
    assert isinstance(detector, MadmomBeatDetector)


# Create a patch at module level to mock the BeatThisDetector before it's imported
@patch('beat_detection.core.beat_this_detector.BeatThisDetector')
def test_get_beat_detector_beat_this(mock_detector):
    """Test getting a BeatThisDetector."""
    # Configure the mock to return a MagicMock instance when instantiated
    mock_instance = MagicMock()
    mock_detector.return_value = mock_instance
    
    # Reload the factory module to use our mock
    import importlib
    import beat_detection.core.factory
    importlib.reload(beat_detection.core.factory)
    
    # Call the factory function
    detector = beat_detection.core.factory.get_beat_detector("beat_this")
    
    # Verify the mock was called
    assert mock_detector.called
    assert detector is mock_instance


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
    # We don't directly import BeatThisDetector to avoid ImportError
    assert DETECTOR_REGISTRY["beat_this"].__name__ == "BeatThisDetector"
