"""
Tests for detector imports and registration.
"""

import inspect
import pytest

# Explicitly import to ensure registration
from beat_counter.core.detectors import madmom, beat_this
from beat_counter.core.registry import _DETECTORS, build

def test_registry_contains_detectors():
    """Test that the registry contains all expected detectors."""
    # Make sure the registry has expected detectors
    assert "madmom" in _DETECTORS
    assert "beat_this" in _DETECTORS

def test_all_detectors_instantiable():
    """Test that all registered detectors can be instantiated."""
    for name in _DETECTORS:
        detector = build(name)
        # Check that we got an instance of a class, not the class itself
        assert not inspect.isclass(detector)
        # Verify it's an instance of the registered detector class
        assert isinstance(detector, _DETECTORS[name])
        # Verify it has the detect_beats method
        assert hasattr(detector, "detect_beats")
        assert callable(detector.detect_beats) 