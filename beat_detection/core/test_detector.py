"""
Tests for the beat detector factory (now part of detector.py).
"""
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Type, Any
import inspect

from beat_detection.core.factory import get_beat_detector, DETECTOR_REGISTRY, BeatDetector
from beat_detection.core.madmom_detector import MadmomBeatDetector
# Import BeatThisDetector conditionally or mock it if necessary
try:
    from beat_detection.core.beat_this_detector import BeatThisDetector
    _BEAT_THIS_AVAILABLE = True
except ImportError:
    BeatThisDetector = MagicMock() # Use MagicMock if BeatThisDetector is not available
    _BEAT_THIS_AVAILABLE = False

# Define a simple MockDetector for testing patching
class MockDetector(BeatDetector):
    def __init__(self, *args, **kwargs):
        print(f"MockDetector initialized with args: {args}, kwargs: {kwargs}")
        # Store kwargs if needed for assertion
        self.init_kwargs = kwargs

    def detect(self, audio_path):
        # Simple mock implementation
        print(f"MockDetector detecting beats for: {audio_path}")
        return MagicMock() # Return a mock object for RawBeats


def test_get_beat_detector_default():
    """Test that the default detector is MadmomBeatDetector."""
    detector = get_beat_detector()
    assert isinstance(detector, MadmomBeatDetector)


def test_get_beat_detector_madmom():
    """Test getting a MadmomBeatDetector explicitly."""
    detector = get_beat_detector("madmom")
    assert isinstance(detector, MadmomBeatDetector)


# Conditionally run test if BeatThisDetector is available
@pytest.mark.skipif(not _BEAT_THIS_AVAILABLE, reason="BeatThisDetector not installed or found")
def test_get_beat_detector_beat_this():
    """Test getting a BeatThisDetector."""
    detector = get_beat_detector("beat_this")
    # Check if it's the real one or the MagicMock we assigned if import failed
    assert isinstance(detector, BeatThisDetector)


def test_get_beat_detector_invalid():
    """Test that an invalid algorithm name raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported beat detection algorithm"):
        get_beat_detector("invalid_algorithm")


def test_get_beat_detector_kwargs():
    """Test that kwargs are passed to the detector constructor."""
    detector = get_beat_detector("madmom", fps=200)
    assert isinstance(detector, MadmomBeatDetector) # Check type first
    assert detector.fps == 200


def test_detector_registry():
    """Test that the detector registry contains the expected detectors."""
    assert "madmom" in DETECTOR_REGISTRY
    assert DETECTOR_REGISTRY["madmom"] == MadmomBeatDetector
    if _BEAT_THIS_AVAILABLE:
        assert "beat_this" in DETECTOR_REGISTRY
        assert DETECTOR_REGISTRY["beat_this"] == BeatThisDetector
    else:
        assert "beat_this" not in DETECTOR_REGISTRY


# Test that get_beat_detector uses the DETECTOR_REGISTRY
# Patch the registry within the factory module where get_beat_detector uses it
@patch("beat_detection.core.factory.DETECTOR_REGISTRY", {
    "mock_detector": MockDetector
})
def test_get_beat_detector_mocked_registry():
    """Test getting a detector using a patched registry."""
    # Call the factory function (imported directly from factory)
    detector = get_beat_detector("mock_detector")

    # Verify the mock detector class was used
    assert isinstance(detector, MockDetector)


# Test filtering of kwargs with a patched registry
# Patch the registry within the factory module where get_beat_detector uses it
@patch("beat_detection.core.factory.DETECTOR_REGISTRY", {
    "mock_detector_with_params": MockDetector # Use the same mock class
})
def test_get_beat_detector_filtered_kwargs_mocked_registry():
    """Test getting a detector with filtered kwargs using a patched registry."""
    # Call the factory function with valid and invalid kwargs for MockDetector
    valid_kwarg = list(inspect.signature(MockDetector.__init__).parameters.keys())[-1] # Assuming last is a valid kwarg
    kwargs_to_pass = {valid_kwarg: "test_value", "extra_param": 123}

    # Expect a warning about the extra param
    with pytest.warns(UserWarning, match="Ignoring extraneous keyword arguments"):
        detector = get_beat_detector("mock_detector_with_params", **kwargs_to_pass)

    # Verify the mock detector class was used and received only valid kwargs
    assert isinstance(detector, MockDetector)
    assert valid_kwarg in detector.init_kwargs
    assert "extra_param" not in detector.init_kwargs
    assert detector.init_kwargs[valid_kwarg] == "test_value"
