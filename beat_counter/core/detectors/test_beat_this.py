"""
Tests for the Beat-This! detector implementation.
"""

import pytest
from unittest.mock import patch, MagicMock, ANY
import numpy as np
from pathlib import Path

from beat_counter.core.detectors.beat_this import BeatThisDetector, BEAT_THIS_FPS
from beat_counter.core.detectors.base import DetectorConfig


def test_beat_this_detector_with_config():
    """Test that BeatThisDetector properly accepts and uses a config object."""
    with patch('beat_this.inference.File2Beats') as mock_file2beats, \
         patch('beat_counter.core.detectors.beat_this.CustomBeatTrackingProcessor') as mock_processor, \
         patch('torch.cuda.is_available', return_value=False):
        
        # Create detector with custom min_bpm
        cfg = DetectorConfig(fps=100, min_bpm=90, max_bpm=180)
        detector = BeatThisDetector(cfg)
        
        # Test that config was stored
        assert detector.cfg is cfg
        assert detector.cfg.min_bpm == 90
        assert detector.cfg.max_bpm == 180
        
        # Test that the config values were passed to the CustomBeatTrackingProcessor
        mock_processor.assert_called_once_with(
            fps=BEAT_THIS_FPS,
            beats_per_bar=cfg.beats_per_bar, 
            min_bpm=cfg.min_bpm,
            max_bpm=cfg.max_bpm,
            transition_lambda=100
        ) 

def test_beat_this_detector_default_config():
    """Test BeatThisDetector with default config values."""
    with patch('beat_this.inference.File2Beats') as mock_file2beats, \
         patch('beat_counter.core.detectors.beat_this.CustomBeatTrackingProcessor') as mock_processor, \
         patch('torch.cuda.is_available', return_value=False):
        
        cfg = DetectorConfig(fps=100)
        detector = BeatThisDetector(cfg)
        
        assert detector.cfg is cfg
        mock_processor.assert_called_once_with(
            fps=BEAT_THIS_FPS,
            beats_per_bar=cfg.beats_per_bar, 
            min_bpm=cfg.min_bpm, 
            max_bpm=cfg.max_bpm,
            transition_lambda=100 
        )