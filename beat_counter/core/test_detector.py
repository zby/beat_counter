"""
Tests for the beat detector factory (now part of registry.py).
"""
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Type, Any
import inspect

from beat_counter.core.registry import build, _DETECTORS
from beat_counter.core.detector_protocol import BeatDetector
from beat_counter.core.detectors.madmom import MadmomBeatDetector
from beat_counter.core.detectors.beat_this import BeatThisDetector
from beat_counter.core.detectors.base import DetectorConfig

# Define a simple MockDetector for testing patching
class MockDetector(BeatDetector):
    def __init__(self, cfg: DetectorConfig, **kwargs):
        self.cfg = cfg
        print(f"MockDetector initialized with cfg: {cfg}, kwargs: {kwargs}")
        # Store kwargs if needed for assertion
        self.init_kwargs = kwargs

    def detect_beats(self, audio_path):
        # Simple mock implementation
        print(f"MockDetector detecting beats for: {audio_path}")
        return MagicMock() # Return a mock object for RawBeats


def test_get_beat_detector_default():
    """Test that the default detector is MadmomBeatDetector."""
    detector = build("madmom")
    assert isinstance(detector, MadmomBeatDetector)


def test_get_beat_detector_madmom():
    """Test getting a MadmomBeatDetector explicitly."""
    detector = build("madmom")
    assert isinstance(detector, MadmomBeatDetector)


def test_get_beat_detector_beat_this():
    """Test getting a BeatThisDetector."""
    detector = build("beat_this")
    assert isinstance(detector, BeatThisDetector)


def test_get_beat_detector_invalid():
    """Test that an invalid algorithm name raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported beat detection algorithm"):
        build("invalid_algorithm")


def test_get_beat_detector_kwargs():
    """Test that kwargs are passed to the detector constructor."""
    detector = build("madmom", fps=200)
    assert isinstance(detector, MadmomBeatDetector) # Check type first
    assert detector.cfg.fps == 200


def test_detector_registry():
    """Test that the detector registry contains the expected detectors."""
    assert "madmom" in _DETECTORS
    assert _DETECTORS["madmom"] == MadmomBeatDetector
    assert "beat_this" in _DETECTORS
    assert _DETECTORS["beat_this"] == BeatThisDetector


# Test that build uses the DETECTOR_REGISTRY
# Patch the registry within the registry module where build uses it
@patch('beat_counter.core.registry._DETECTORS', {
    "mock_detector": MockDetector
})
def test_get_beat_detector_mocked_registry():
    """Test getting a detector using a patched registry."""
    # Call the factory function (imported directly from registry)
    detector = build("mock_detector")

    # Verify the mock detector class was used
    assert isinstance(detector, MockDetector)


# Test kwargs passing
@patch('beat_counter.core.registry._DETECTORS', {
    "mock_detector_with_params": MockDetector # Use the same mock class
})
def test_get_beat_detector_filtered_kwargs_mocked_registry():
    """Test that kwargs are passed to the detector constructor."""
    # Call the function with kwargs
    kwargs_to_pass = {"valid_kwarg": "test_value", "extra_param": 123}

    detector = build("mock_detector_with_params", **kwargs_to_pass)

    # Verify the mock detector class was used
    assert isinstance(detector, MockDetector)
    # In the new implementation, only recognized kwargs are passed to the config, 
    # and the rest are passed to the detector constructor
    assert "valid_kwarg" in detector.init_kwargs
    assert detector.init_kwargs["valid_kwarg"] == "test_value"
    assert "extra_param" in detector.init_kwargs
    assert detector.init_kwargs["extra_param"] == 123
