"""
Tests for the base detector module.
"""

import pytest
from beat_counter.core.detectors.base import DetectorConfig, BaseBeatDetector
import beat_counter.utils.constants as constants


def test_config_defaults():
    """Test that the config defaults match expectations."""
    cfg = DetectorConfig()
    assert cfg.min_bpm == 60
    assert cfg.max_bpm == 240
    assert cfg.fps is None
    assert cfg.beats_per_bar == constants.SUPPORTED_BEATS_PER_BAR


def test_config_params():
    """Test that the config accepts and stores custom values."""
    cfg = DetectorConfig(min_bpm=90, max_bpm=180, fps=50, beats_per_bar=[3, 4, 5])
    assert cfg.min_bpm == 90
    assert cfg.max_bpm == 180
    assert cfg.fps == 50
    assert cfg.beats_per_bar == [3, 4, 5]


class DummyDetector(BaseBeatDetector):
    """Test implementation of BaseBeatDetector."""
    pass


def test_detector_accepts_config():
    """Test that BaseBeatDetector properly accepts and stores the config."""
    d = DummyDetector(DetectorConfig(min_bpm=90))
    assert d.cfg.min_bpm == 90
    assert d.cfg.max_bpm == 240  # Default value 