"""
Tests for pipeline functionality.
"""
import os
import pytest
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch

from beat_detection.core.pipeline import extract_beats, process_batch
from beat_detection.core.beats import RawBeats, Beats


class FakeDetector:
    """Mock detector for testing."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def detect_beats(self, audio_path):
        """Return fixed beats for testing."""
        # Create enough beats for a valid test (5 measures with 4 beats each = 20 beats)
        # Generate timestamps with consistent intervals (0.5s) and simple pattern of beat counts
        timestamps = np.array([0.5 + i*0.5 for i in range(25)])
        
        # Create repeating pattern of 1,2,3,4 for beat counts (for a 4/4 time signature)
        beat_counts = np.array([(i % 4) + 1 for i in range(25)])
        
        return RawBeats(timestamps=timestamps, beat_counts=beat_counts, clip_length=13.0)


@pytest.fixture
def audio_file(tmp_path):
    """Create a fake audio file for testing."""
    audio_path = tmp_path / "test.wav"
    audio_path.touch()  # Create an empty file
    return str(audio_path)


# Use function-level patching to avoid issues with imported names
@patch("beat_detection.core.pipeline.get_detector")
def test_extract_beats_uses_detector(mock_get_detector, audio_file):
    """Test that extract_beats uses the detector from registry."""
    # Set up mock
    fake_detector = FakeDetector()
    mock_get_detector.return_value = fake_detector
    
    # Call function
    result = extract_beats(audio_file, algorithm="test_algo")
    
    # Check that registry.get was called with the right args
    mock_get_detector.assert_called_once_with(algorithm="test_algo")
    
    # Check that the result is a Beats object with expected values
    assert isinstance(result, Beats)
    assert result.beat_data.shape[0] == 25  # 25 beats
    assert result.beat_data[0, 0] == 0.5    # First timestamp
    
    # Check that .beats file was created
    beats_file = Path(audio_file).with_suffix(".beats")
    assert beats_file.exists()
    
    # Check that ._beat_stats file was created
    stats_file = Path(str(beats_file).replace(".beats", "._beat_stats"))
    assert stats_file.exists()


@patch("beat_detection.core.pipeline.get_detector")
@patch("beat_detection.core.pipeline.find_audio_files")
def test_process_batch(mock_find_files, mock_get_detector, tmp_path):
    """Test batch processing of audio files."""
    # Create some test files
    file1 = tmp_path / "test1.wav"
    file2 = tmp_path / "test2.wav"
    file1.touch()
    file2.touch()
    
    # Set up mock for find_audio_files
    mock_find_files.return_value = [file1, file2]
    
    # Set up mock for get_detector
    fake_detector = FakeDetector()
    mock_get_detector.return_value = fake_detector
    
    # Call function with no_progress=True to avoid tqdm output in tests
    results = process_batch(tmp_path, algorithm="test_algo", no_progress=True)
    
    # Check results
    assert len(results) == 2
    assert results[0][0] == "test1.wav"
    assert results[1][0] == "test2.wav"
    assert isinstance(results[0][1], Beats)
    assert isinstance(results[1][1], Beats)
    
    # Check that files were created
    assert (tmp_path / "test1.beats").exists()
    assert (tmp_path / "test2.beats").exists()
    assert (tmp_path / "test1._beat_stats").exists()
    assert (tmp_path / "test2._beat_stats").exists() 