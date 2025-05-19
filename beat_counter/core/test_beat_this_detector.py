# tests/core/test_beat_this_detector.py
# -----------------------------------------------------------------------------
# Minimal test-suite for the BeatThisDetector wrapper.
# -----------------------------------------------------------------------------

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from beat_counter.core.detectors.beat_this import BeatThisDetector
from beat_counter.core.detectors.base import DetectorConfig
from beat_counter.core.beats import RawBeats


# -----------------------------------------------------------------------------
# Helper – create a detector with a fully mocked File2Beats implementation
# -----------------------------------------------------------------------------

def _make_mock_detector(beats: np.ndarray, downbeats: np.ndarray) -> BeatThisDetector:
    """Instantiate BeatThisDetector while monkey-patching File2Beats."""

    class DummyProcessor:
        def __init__(self, *_, **__):
            pass

        def __call__(self, _audio_path: str):  # noqa: D401 – mimic __call__ interface
            return beats, downbeats

    with patch('beat_counter.core.detectors.beat_this.File2Beats', DummyProcessor):
        # Create a default config for the detector
        cfg = DetectorConfig()
        return BeatThisDetector(cfg)


# -----------------------------------------------------------------------------
# Unit tests
# -----------------------------------------------------------------------------

def test_beats_to_counts():
    """_beats_to_counts maps beats→counts with 1 on each downbeat."""

    beats = np.array([0.0, 1.0, 2.0, 3.0])
    downbeats = np.array([0.0])

    _, counts = BeatThisDetector._beats_to_counts(beats, downbeats)
    np.testing.assert_array_equal(counts, np.array([1, 2, 3, 4]))


def test_detect_returns_rawbeats(tmp_path: Path):
    """detect() returns RawBeats with timestamps and inferred counts."""

    audio_file = tmp_path / "dummy.wav"
    audio_file.touch()

    beats = np.array([0.0, 1.0])
    downbeats = np.array([0.0])

    detector = _make_mock_detector(beats, downbeats)
    
    # Mock _get_audio_duration to return a fixed duration
    with patch.object(detector, '_get_audio_duration', return_value=2.0):
        raw = detector.detect_beats(audio_file)

    assert isinstance(raw, RawBeats)
    assert np.array_equal(raw.timestamps, beats)
    assert np.array_equal(raw.beat_counts, np.array([1, 2]))  # Inferred counts
    assert raw.clip_length == 2.0  # Verify clip_length from mock


def test_detect_file_not_found():
    """detect() raises FileNotFoundError for missing files."""

    detector = _make_mock_detector(np.array([]), np.array([]))
    with pytest.raises(FileNotFoundError):
        detector.detect_beats("missing.wav") 