"""Tests for beat statistics functionality."""

import json
import pathlib
import tempfile
import pytest
from beat_detection.utils.reporting import (
    get_beat_statistics_dict,
    save_beat_statistics
)
from beat_detection.core.detector import BeatStatistics

@pytest.fixture
def sample_beat_stats():
    """Create a sample BeatStatistics object for testing."""
    return BeatStatistics(
        tempo_bpm=120.5,
        mean_interval=0.5,
        median_interval=0.498,
        std_interval=0.02,
        min_interval=0.45,
        max_interval=0.55,
        irregularity_percent=5.2
    )

@pytest.fixture
def sample_irregular_beats():
    """Create sample irregular beats for testing."""
    return [2, 5, 8]

def test_get_beat_statistics_dict(sample_beat_stats, sample_irregular_beats):
    """Test the get_beat_statistics_dict function."""
    # Test without optional parameters
    stats_dict = get_beat_statistics_dict(sample_beat_stats, sample_irregular_beats)
    
    assert stats_dict["tempo_bpm"] == 120.5
    assert stats_dict["mean_interval"] == 0.5
    assert stats_dict["irregular_beats_count"] == 3
    assert stats_dict["irregular_beat_indices"] == [2, 5, 8]
    assert "filename" not in stats_dict
    assert "detected_meter" not in stats_dict
    assert "duration" not in stats_dict
    
    # Test with all optional parameters
    stats_dict = get_beat_statistics_dict(
        sample_beat_stats,
        sample_irregular_beats,
        filename="test.mp3",
        detected_meter=4,
        duration=180.5
    )
    
    assert stats_dict["filename"] == "test.mp3"
    assert stats_dict["detected_meter"] == 4
    assert stats_dict["duration"] == 180.5

def test_save_beat_statistics(sample_beat_stats, sample_irregular_beats):
    """Test saving beat statistics in JSON format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        stats_file = pathlib.Path(temp_dir) / "test_beat_stats.json"
        
        # Save with all parameters
        save_beat_statistics(
            sample_beat_stats,
            sample_irregular_beats,
            stats_file,
            filename="test.mp3",
            detected_meter=4,
            duration=180.5
        )
        
        # Read and verify the JSON file
        with open(stats_file, 'r') as f:
            saved_stats = json.load(f)
        
        assert saved_stats["tempo_bpm"] == 120.5
        assert saved_stats["mean_interval"] == 0.5
        assert saved_stats["irregular_beats_count"] == 3
        assert saved_stats["irregular_beat_indices"] == [2, 5, 8]
        assert saved_stats["filename"] == "test.mp3"
        assert saved_stats["detected_meter"] == 4
        assert saved_stats["duration"] == 180.5

def test_rounding_of_values(sample_beat_stats, sample_irregular_beats):
    """Test that numeric values are properly rounded in the output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        stats_file = pathlib.Path(temp_dir) / "test_beat_stats.json"
        
        # Save with values that need rounding
        save_beat_statistics(
            sample_beat_stats,
            sample_irregular_beats,
            stats_file,
            duration=180.5678
        )
        
        # Read and verify rounding
        with open(stats_file, 'r') as f:
            saved_stats = json.load(f)
        
        assert saved_stats["tempo_bpm"] == 120.5  # Rounded to 1 decimal
        assert saved_stats["mean_interval"] == 0.5  # Rounded to 3 decimals
        assert saved_stats["irregularity_percent"] == 5.2  # Rounded to 1 decimal
        assert saved_stats["duration"] == 180.568  # Rounded to 3 decimals 