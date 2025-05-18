"""
Tests for the beat detection registry.
"""

import pytest
import warnings
from unittest.mock import patch, MagicMock

from beat_detection.core.registry import register, build, get
from beat_detection.core.detectors.base import DetectorConfig


# Test detector class
class MockDetector:
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.kwargs = kwargs


def test_register():
    """Test detector registration."""
    # Create a local registry for testing
    with patch('beat_detection.core.registry._DETECTORS', {}):
        # Register the test detector
        register_name = "test_detector"
        decorated = register(register_name)(MockDetector)
        
        from beat_detection.core.registry import _DETECTORS
        assert register_name in _DETECTORS
        assert _DETECTORS[register_name] == MockDetector
        assert decorated == MockDetector


def test_build_with_config():
    """Test building a detector with an explicit config object."""
    with patch('beat_detection.core.registry._DETECTORS', {"test_detector": MockDetector}):
        # Create a config
        config = DetectorConfig(min_bpm=90, max_bpm=180)
        
        # Build detector with config
        detector = build("test_detector", config=config, extra_arg="value")
        
        # Check detector was created with our config and extra args
        assert detector.cfg is config
        assert detector.kwargs == {"extra_arg": "value"}


def test_build_with_kwargs():
    """Test building a detector from kwargs to create a config."""
    with patch('beat_detection.core.registry._DETECTORS', {"test_detector": MockDetector}):
        # Build detector with kwargs that should go into config
        detector = build("test_detector", min_bpm=90, max_bpm=180, extra_arg="value")
        
        # Check detector was created with a proper config and extra args
        assert isinstance(detector.cfg, DetectorConfig)
        assert detector.cfg.min_bpm == 90
        assert detector.cfg.max_bpm == 180
        assert detector.kwargs == {"extra_arg": "value"}


def test_get_deprecated():
    """Test that get() calls build() and emits a deprecation warning."""
    with patch('beat_detection.core.registry.build') as mock_build, \
         warnings.catch_warnings(record=True) as recorded_warnings:
        
        # Configure mock
        mock_build.return_value = "mock detector"
        
        # Call get() and capture warning
        result = get("test_detector", min_bpm=90, extra_arg="value")
        
        # Verify build was called with correct args
        mock_build.assert_called_once_with("test_detector", min_bpm=90, extra_arg="value")
        
        # Verify result is from build()
        assert result == "mock detector"
        
        # Verify deprecation warning was issued
        assert len(recorded_warnings) > 0
        assert issubclass(recorded_warnings[0].category, DeprecationWarning)
        assert "get() is deprecated" in str(recorded_warnings[0].message)


def test_build_algorithm_not_found():
    """Test that build() raises an error for unknown algorithms."""
    with patch('beat_detection.core.registry._DETECTORS', {}):
        with pytest.raises(ValueError) as excinfo:
            build("nonexistent")
        
        assert "Unsupported beat detection algorithm" in str(excinfo.value) 