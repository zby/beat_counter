"""
Tests for the serialization and deserialization of the Beats class.
"""

import pytest
import json
from pathlib import Path
import numpy as np

from beat_detection.core.beats import Beats, BeatCalculationError, BeatStatistics

# Assuming create_test_beats is accessible, either defined here or imported
# For simplicity, let's import it from the other test file
# (In a real scenario, you might move helpers to a common conftest.py)
from beat_detection.core.test_beats import create_test_beats


def test_save_and_load_beats(tmp_path: Path):
    """Test saving a Beats object to JSON and loading it back."""
    # 1. Create a sample Beats object
    original_beats = create_test_beats(
        beats_per_bar=3, num_beats=15, interval=0.6, tolerance=15.0, min_measures=2
    )
    file_path = tmp_path / "test_beats.json"

    # 2. Save the object
    original_beats.save_to_file(file_path)
    assert file_path.is_file()

    # 3. Load the object
    loaded_beats = Beats.load_from_file(file_path)

    # 4. Compare original and loaded objects
    assert isinstance(loaded_beats, Beats)
    assert loaded_beats.beats_per_bar == original_beats.beats_per_bar
    assert loaded_beats.tolerance_percent == original_beats.tolerance_percent
    assert loaded_beats.min_measures == original_beats.min_measures
    assert np.isclose(
        loaded_beats.tolerance_interval, original_beats.tolerance_interval
    )
    assert loaded_beats.start_regular_beat_idx == original_beats.start_regular_beat_idx
    assert loaded_beats.end_regular_beat_idx == original_beats.end_regular_beat_idx

    # Compare stats (using np.isclose for float comparisons)
    assert np.isclose(
        loaded_beats.overall_stats.mean_interval,
        original_beats.overall_stats.mean_interval,
    )
    assert np.isclose(
        loaded_beats.overall_stats.median_interval,
        original_beats.overall_stats.median_interval,
    )
    assert (
        loaded_beats.overall_stats.total_beats
        == original_beats.overall_stats.total_beats
    )
    assert np.isclose(
        loaded_beats.regular_stats.tempo_bpm, original_beats.regular_stats.tempo_bpm
    )
    assert (
        loaded_beats.regular_stats.total_beats
        == original_beats.regular_stats.total_beats
    )

    # Compare beat_data array
    assert isinstance(loaded_beats.beat_data, np.ndarray)
    np.testing.assert_allclose(loaded_beats.beat_data, original_beats.beat_data)


def test_load_beats_file_not_found(tmp_path: Path):
    """Test loading from a non-existent file path."""
    file_path = tmp_path / "non_existent_beats.json"
    with pytest.raises(FileNotFoundError, match=f"Beats file not found: {file_path}"):
        Beats.load_from_file(file_path)


def test_load_beats_invalid_json(tmp_path: Path):
    """Test loading from a file with invalid JSON content."""
    file_path = tmp_path / "invalid_beats.json"
    file_path.write_text("this is not valid json{")

    with pytest.raises(
        BeatCalculationError, match="Error decoding JSON from Beats file"
    ):
        Beats.load_from_file(file_path)


def test_load_beats_missing_key(tmp_path: Path):
    """Test loading from a JSON file missing a required key."""
    file_path = tmp_path / "missing_key_beats.json"
    # Create valid data, then remove a key
    original_beats = create_test_beats(min_measures=1)  # Use simpler object
    data = original_beats.to_dict()
    del data["regular_stats"]  # Remove a required key

    with file_path.open("w") as f:
        json.dump(data, f)

    # Adjust regex to match the double single quotes in the KeyError representation
    with pytest.raises(
        BeatCalculationError, match="Missing expected key ''regular_stats''"
    ):
        Beats.load_from_file(file_path)


def test_load_beats_incorrect_type(tmp_path: Path):
    """Test loading from JSON where a value has an incorrect type."""
    file_path = tmp_path / "incorrect_type_beats.json"
    original_beats = create_test_beats(min_measures=1)
    data = original_beats.to_dict()
    data["beats_per_bar"] = "not_an_integer"  # Corrupt the type

    with file_path.open("w") as f:
        json.dump(data, f)

    # Match the exact beginning of the error message
    with pytest.raises(
        BeatCalculationError,
        match=r"Type or value error reconstructing Beats object from",
    ):
        Beats.load_from_file(file_path)


# Helper function to create a sample Beats object for testing
def create_sample_beats_for_serialization() -> Beats:
    """Creates a predictable Beats object for serialization tests."""
    timestamps = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    beats_per_bar = 4
    num_beats = len(timestamps)
    # Generate counts based on beats_per_bar
    beat_counts = np.array([(i % beats_per_bar) + 1 for i in range(num_beats)])

    # Use from_timestamps to ensure consistency with creation logic
    # Need beats_per_bar * min_measures = 4 * 1 = 4 beats minimum. We have 10.
    beats_obj = Beats.from_timestamps(
        timestamps=timestamps,
        beats_per_bar=beats_per_bar,
        beat_counts=beat_counts,  # Add counts
        tolerance_percent=15.0,
        min_measures=1,  # Lower requirement for this test case
    )
    return beats_obj


def test_beats_to_dict_structure():
    """Test the overall structure and keys of the dictionary produced by Beats.to_dict()."""
    beats_obj = create_sample_beats_for_serialization()
    beats_dict = beats_obj.to_dict()

    # Check top-level keys
    expected_top_keys = {
        "beats_per_bar",
        "tolerance_percent",
        "tolerance_interval",
        "min_measures",
        "start_regular_beat_idx",
        "end_regular_beat_idx",
        "overall_stats",
        "regular_stats",
        "beat_list",
    }
    assert set(beats_dict.keys()) == expected_top_keys

    # Check stats keys (assuming BeatStatistics.to_dict works)
    expected_stat_keys = {
        "mean_interval",
        "median_interval",
        "std_interval",
        "min_interval",
        "max_interval",
        "irregularity_percent",
        "tempo_bpm",
        "total_beats",
    }
    assert isinstance(beats_dict["overall_stats"], dict)
    assert set(beats_dict["overall_stats"].keys()) == expected_stat_keys
    assert isinstance(beats_dict["regular_stats"], dict)
    assert set(beats_dict["regular_stats"].keys()) == expected_stat_keys

    # Check beat_list structure
    assert isinstance(beats_dict["beat_list"], list)
    assert len(beats_dict["beat_list"]) == beats_obj.beat_data.shape[0]
    if beats_dict["beat_list"]:
        # Check keys of the first beat entry in the list
        expected_beat_entry_keys = {"timestamp", "count"}
        assert isinstance(beats_dict["beat_list"][0], dict)
        assert set(beats_dict["beat_list"][0].keys()) == expected_beat_entry_keys


def test_beats_to_dict_values():
    """Test specific values converted by Beats.to_dict()."""
    beats_obj = create_sample_beats_for_serialization()
    beats_dict = beats_obj.to_dict()

    assert beats_dict["beats_per_bar"] == beats_obj.beats_per_bar
    assert beats_dict["tolerance_percent"] == beats_obj.tolerance_percent
    assert beats_dict["tolerance_interval"] == beats_obj.tolerance_interval
    assert beats_dict["min_measures"] == beats_obj.min_measures
    assert beats_dict["start_regular_beat_idx"] == beats_obj.start_regular_beat_idx
    assert beats_dict["end_regular_beat_idx"] == beats_obj.end_regular_beat_idx

    # Check a specific beat entry in the serialized list
    # Example: Check the 5th beat entry (index 4), which corresponds to the 5th row in beat_data
    original_timestamp = beats_obj.beat_data[4, 0]
    original_count = beats_obj.beat_data[4, 1]
    beat_entry_dict = beats_dict["beat_list"][4]
    assert np.isclose(beat_entry_dict["timestamp"], original_timestamp)
    assert beat_entry_dict["count"] == int(original_count)


def test_json_serialization():
    """Test that the dictionary from to_dict can be serialized by the json library."""
    beats_obj = create_sample_beats_for_serialization()
    beats_dict = beats_obj.to_dict()

    try:
        # Using np.encoder.JSONEncoder for robustness, although to_dict should convert types
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
