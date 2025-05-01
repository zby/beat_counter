import json
import numpy as np
import pytest
from pathlib import Path

# Assuming RawBeats is importable from this location
# Removed: from beat_detection.core.beats import RawBeats
from beat_detection.core.beats import RawBeats # Import from beats.py now

# Define default parameters for fixtures and tests
# Removed: DEFAULT_BPB = 4

@pytest.fixture
def sample_raw_beats() -> RawBeats: # No longer needs params
    """Provides a sample RawBeats instance for testing."""
    timestamps = np.array([0.5, 1.0, 1.5, 2.0])
    # Counts no longer tied to a specific BPB
    counts = np.array([1, 2, 1, 2])
    return RawBeats(timestamps=timestamps, beat_counts=counts) # Removed params


@pytest.fixture
def temp_beat_file(tmp_path: Path) -> Path:
    """Provides a temporary file path for saving/loading tests."""
    return tmp_path / "test_raw_beats.json"


def test_raw_beats_init_validation(): # Removed fixture dependency
    """Test the __init__ / __post_init__ validation logic."""
    # Mismatched lengths (Not shapes anymore)
    with pytest.raises(ValueError, match="Timestamp count.*does not match beat count"):
        RawBeats(
            timestamps=np.array([1.0, 2.0]),
            beat_counts=np.array([1, 2, 3]),
        )

    # Wrong dimensions (timestamps) - Need valid counts shape
    with pytest.raises(ValueError, match="Timestamps must be a 1D numpy array"):
        RawBeats(
            timestamps=np.array([[1.0], [2.0]]),
            beat_counts=np.array([1, 2]), # Keep counts 1D to trigger timestamp error
        )

    # Wrong dimensions (beat_counts) - Need valid timestamps shape
    with pytest.raises(ValueError, match="Beat counts must be a 1D numpy array"):
        RawBeats(
            timestamps=np.array([1.0, 2.0]),
            beat_counts=np.array([[1], [2]]), # Make counts 2D
        )

    # Non-increasing timestamps
    with pytest.raises(ValueError, match="Timestamps must be strictly increasing"):
        RawBeats(
            timestamps=np.array([1.0, 0.9, 2.0]),
            beat_counts=np.array([1, 2, 3]),
        )

    # Removed: Invalid beats_per_bar test

    # No error expected for valid input
    RawBeats(
        timestamps=np.array([1.0, 2.0]),
        beat_counts=np.array([1, 2]),
    )


def test_raw_beats_save_and_load(
    sample_raw_beats: RawBeats, temp_beat_file: Path
):
    """Test saving and then loading a RawBeats object."""
    sample_raw_beats.save_to_file(temp_beat_file)
    assert temp_beat_file.is_file()

    # Verify JSON content
    with temp_beat_file.open("r") as f:
        data = json.load(f)
        assert "beats_per_bar" not in data # Verify removed
        assert "timestamps" in data
        assert "beat_counts" in data
        # Check actual data values
        np.testing.assert_array_equal(data["timestamps"], sample_raw_beats.timestamps.tolist())
        np.testing.assert_array_equal(data["beat_counts"], sample_raw_beats.beat_counts.astype(int).tolist())


    loaded_beats = RawBeats.load_from_file(temp_beat_file)

    assert isinstance(loaded_beats, RawBeats)
    np.testing.assert_array_equal(
        loaded_beats.timestamps, sample_raw_beats.timestamps
    )
    np.testing.assert_array_equal(
        loaded_beats.beat_counts, sample_raw_beats.beat_counts
    )
    # Removed: assert loaded_beats.beats_per_bar == sample_raw_beats.beats_per_bar


def test_raw_beats_save_creates_directory(
    sample_raw_beats: RawBeats, tmp_path: Path
):
    """Test that save_to_file creates parent directories if needed."""
    nested_path = tmp_path / "nested" / "dir" / "beats.json"
    assert not nested_path.parent.exists()
    sample_raw_beats.save_to_file(nested_path)
    assert nested_path.is_file()


def test_raw_beats_load_file_not_found(temp_beat_file: Path):
    """Test loading a non-existent file raises FileNotFoundError."""
    assert not temp_beat_file.exists()
    with pytest.raises(FileNotFoundError):
        RawBeats.load_from_file(temp_beat_file)


def test_raw_beats_load_invalid_json(temp_beat_file: Path):
    """Test loading a file with invalid JSON raises ValueError."""
    temp_beat_file.write_text("this is not json")
    with pytest.raises(ValueError, match="Error decoding JSON"):
        RawBeats.load_from_file(temp_beat_file)


def test_raw_beats_load_missing_keys(temp_beat_file: Path):
    """Test loading JSON missing required keys raises ValueError."""
    # Test missing 'timestamps'
    data_missing_timestamps = {
        # Removed "beats_per_bar": DEFAULT_BPB,
        "beat_counts": [1, 2]
    }
    with temp_beat_file.open("w") as f:
        json.dump(data_missing_timestamps, f)
    with pytest.raises(ValueError, match=r"is missing required keys: \['timestamps'\]\."):
        RawBeats.load_from_file(temp_beat_file)

    # Test missing 'beat_counts'
    data_missing_counts = {
        # Removed "beats_per_bar": DEFAULT_BPB,
        "timestamps": [1.0, 2.0],
    }
    with temp_beat_file.open("w") as f:
        json.dump(data_missing_counts, f)
    with pytest.raises(ValueError, match=r"is missing required keys: \['beat_counts'\]\."):
         RawBeats.load_from_file(temp_beat_file)

    # Removed: Test missing beats_per_bar key

# Removed: test_raw_beats_load_invalid_param_type (bpb removed)

def test_raw_beats_load_mismatched_lengths(temp_beat_file: Path):
    """Test loading JSON with mismatched array lengths raises ValueError."""
    data = {
        # Removed "beats_per_bar": DEFAULT_BPB,
        "timestamps": [1.0, 2.0, 3.0], # Length 3
        "beat_counts": [1, 2]          # Length 2
    }
    with temp_beat_file.open("w") as f:
        json.dump(data, f)
    # Match the error raised directly by load_from_file for mismatched lengths
    with pytest.raises(ValueError, match="Mismatched lengths in.*timestamps vs.*counts"):
        RawBeats.load_from_file(temp_beat_file)


def test_raw_beats_load_empty_arrays(temp_beat_file: Path): # Removed fixture dependency
    """Test loading JSON with empty but valid arrays."""
    data = {
        # Removed: **sample_raw_beats_params,
        "timestamps": [],
        "beat_counts": []
    }
    with temp_beat_file.open("w") as f:
        json.dump(data, f)

    loaded_beats = RawBeats.load_from_file(temp_beat_file)
    assert isinstance(loaded_beats, RawBeats)
    np.testing.assert_array_equal(loaded_beats.timestamps, np.array([]))
    np.testing.assert_array_equal(loaded_beats.beat_counts, np.array([]))
    # Removed: assert loaded_beats.beats_per_bar == sample_raw_beats_params["beats_per_bar"]

def test_raw_beats_attributes(sample_raw_beats: RawBeats):
    """Test accessing attributes."""
    assert hasattr(sample_raw_beats, 'timestamps')
    assert hasattr(sample_raw_beats, 'beat_counts')
    assert not hasattr(sample_raw_beats, 'beats_per_bar') # Verify removal

    # Check types
    assert isinstance(sample_raw_beats.timestamps, np.ndarray)
    assert isinstance(sample_raw_beats.beat_counts, np.ndarray)

    # Check values (redundant with save/load but good for direct check)
    np.testing.assert_array_equal(sample_raw_beats.timestamps, np.array([0.5, 1.0, 1.5, 2.0]))
    np.testing.assert_array_equal(sample_raw_beats.beat_counts, np.array([1, 2, 1, 2])) 