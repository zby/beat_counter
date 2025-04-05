"""
Tests for the serialization of Beats objects.
"""

import json
import numpy as np
import pytest

from beat_detection.core.beats import Beats, BeatInfo, BeatStatistics, BeatCalculationError

# Helper function to create a sample Beats object for testing
def create_sample_beats_for_serialization() -> Beats:
    """Creates a predictable Beats object for serialization tests."""
    timestamps = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    downbeats = np.array([0, 4, 8]) # Meter 4
    meter = 4
    
    # Use from_timestamps to ensure consistency with creation logic
    # Need meter * min_consistent_measures = 4 * 1 = 4 beats minimum. We have 10.
    beats_obj = Beats.from_timestamps(
        timestamps=timestamps,
        downbeat_indices=downbeats,
        meter=meter,
        tolerance_percent=15.0, 
        min_consistent_measures=1 # Lower requirement for this test case
    )
    return beats_obj

def test_beat_info_to_dict():
    """Test the to_dict method of BeatInfo."""
    beat_info = BeatInfo(
        timestamp=1.23, 
        index=5, 
        is_downbeat=True, 
        is_irregular_interval=False, 
        beat_count=0  # 0 indicates irregular/undetermined
    )
    expected_dict = {
        "timestamp": 1.23,
        "index": 5,
        "is_downbeat": True,
        "is_irregular_interval": False,
        "is_irregular": True, # Property should be True because beat_count is 0
        "beat_count": 0,
    }
    assert beat_info.to_dict() == expected_dict

def test_beats_to_dict_structure():
    """Test the overall structure and keys of the dictionary produced by Beats.to_dict()."""
    beats_obj = create_sample_beats_for_serialization()
    beats_dict = beats_obj.to_dict()
    
    # Check top-level keys
    expected_top_keys = {
        "meter", "tolerance_percent", "tolerance_interval", 
        "min_consistent_measures", "start_regular_beat_idx", 
        "end_regular_beat_idx", "overall_stats", "regular_stats", "beat_list"
    }
    assert set(beats_dict.keys()) == expected_top_keys
    
    # Check stats keys (assuming BeatStatistics.to_dict works)
    expected_stat_keys = {
        'mean_interval', 'median_interval', 'std_interval',
        'min_interval', 'max_interval', 'irregularity_percent',
        'tempo_bpm', 'total_beats'
    }
    assert isinstance(beats_dict["overall_stats"], dict)
    assert set(beats_dict["overall_stats"].keys()) == expected_stat_keys
    assert isinstance(beats_dict["regular_stats"], dict)
    assert set(beats_dict["regular_stats"].keys()) == expected_stat_keys
    
    # Check beat_list structure
    assert isinstance(beats_dict["beat_list"], list)
    assert len(beats_dict["beat_list"]) == len(beats_obj.beat_list)
    if beats_dict["beat_list"]:
        # Check keys of the first beat in the list
        expected_beat_info_keys = {
            "timestamp", "index", "is_downbeat", "is_irregular_interval",
            "is_irregular", "beat_count"
        }
        assert isinstance(beats_dict["beat_list"][0], dict)
        assert set(beats_dict["beat_list"][0].keys()) == expected_beat_info_keys

def test_beats_to_dict_values():
    """Test specific values converted by Beats.to_dict()."""
    beats_obj = create_sample_beats_for_serialization()
    beats_dict = beats_obj.to_dict()
    
    assert beats_dict["meter"] == beats_obj.meter
    assert beats_dict["tolerance_percent"] == beats_obj.tolerance_percent
    assert beats_dict["tolerance_interval"] == beats_obj.tolerance_interval
    assert beats_dict["min_consistent_measures"] == beats_obj.min_consistent_measures
    assert beats_dict["start_regular_beat_idx"] == beats_obj.start_regular_beat_idx
    assert beats_dict["end_regular_beat_idx"] == beats_obj.end_regular_beat_idx
    
    # Check a specific beat
    # Example: Check the 5th beat (index 4), which is the second downbeat
    beat_info_obj = beats_obj.beat_list[4] 
    beat_info_dict = beats_dict["beat_list"][4]
    assert beat_info_dict["timestamp"] == beat_info_obj.timestamp
    assert beat_info_dict["index"] == beat_info_obj.index
    assert beat_info_dict["is_downbeat"] == beat_info_obj.is_downbeat
    assert beat_info_dict["beat_count"] == beat_info_obj.beat_count
    assert beat_info_dict["is_irregular"] == beat_info_obj.is_irregular

def test_json_serialization():
    """Test that the dictionary from to_dict can be serialized by the json library."""
    beats_obj = create_sample_beats_for_serialization()
    beats_dict = beats_obj.to_dict()
    
    try:
        json_string = json.dumps(beats_dict, indent=4)
        # Optionally, try loading it back to ensure it's valid JSON
        loaded_dict = json.loads(json_string)
        assert loaded_dict == beats_dict
    except TypeError as e:
        pytest.fail(f"Beats.to_dict() produced unserializable data: {e}")

# Optional: Add tests for from_dict methods if/when they are implemented
# def test_beat_info_from_dict():
#     ...
# def test_beats_from_dict():
#     ... 