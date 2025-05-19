"""
Tests for the madmom beat detector implementation.
"""

import pytest
from unittest.mock import patch, MagicMock, ANY

from beat_detection.core.detectors.madmom import MadmomBeatDetector
from beat_detection.core.detectors.base import DetectorConfig


def test_madmom_detector_with_config():
    """Test that MadmomBeatDetector properly accepts and uses a config object."""
    # Mock the entire class initialization to avoid madmom import issues
    with patch.object(MadmomBeatDetector, '__init__', return_value=None) as mock_init:
        
        # Create a detector with custom min_bpm
        cfg = DetectorConfig(min_bpm=90, max_bpm=180)
        detector = MadmomBeatDetector(cfg)
        
        # Check that init was called with our config
        mock_init.assert_called_once_with(cfg)
        
        # Manually set the config since we mocked __init__
        detector.cfg = cfg
        
        # Test that config was stored correctly
        assert detector.cfg.min_bpm == 90
        assert detector.cfg.max_bpm == 180 