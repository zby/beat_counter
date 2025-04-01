"""
Tests for the Beats data structure and its core logic.
"""

import numpy as np
import pytest

from beat_detection.core.beats import Beats, BeatInfo, BeatStatistics, BeatCalculationError

# Helper function to create a standard Beats object for testing
# Note: Using simple, predictable values
def create_test_beats(meter=4, num_beats=20, interval=0.5, tolerance=10.0, min_measures=2) -> Beats:
    """Creates a predictable Beats object for logic tests."""
    timestamps = np.arange(num_beats) * interval
    # Simple downbeats every 'meter' beats
    downbeat_indices = np.arange(0, num_beats, meter)
    
    # Ensure enough beats for the default min_consistent_measures in from_timestamps
    # Note: from_timestamps itself raises error if not enough, this helper just ensures
    #       the *inputs* to from_timestamps are sufficient for the *helper's* default params.
    required = meter * min_measures 
    if num_beats < required:
        # Adjust num_beats if the defaults don't meet the minimum requirement
        timestamps = np.arange(required) * interval
        downbeat_indices = np.arange(0, required, meter)
        num_beats = required # Update num_beats to match

    # This might still raise BeatCalculationError if constraints aren't met,
    # which is expected behavior for some tests.
    return Beats.from_timestamps(
        timestamps=timestamps,
        downbeat_indices=downbeat_indices,
        meter=meter,
        tolerance_percent=tolerance,
        min_consistent_measures=min_measures
    )

# Test Cases

def test_beat_creation_and_properties():
    """Test basic creation and property access."""
    beats = create_test_beats(meter=4, num_beats=16, interval=0.5, min_measures=3) 
    assert len(beats.beat_list) == 16
    assert beats.meter == 4
    assert beats.stats.total_beats == 16
    assert beats.tolerance_percent == 10.0 # Default tolerance from helper
    assert beats.min_consistent_measures == 3
    assert np.isclose(beats.stats.median_interval, 0.5)
    assert np.isclose(beats.stats.tempo_bpm, 120.0)
    assert len(beats.timestamps) == 16
    assert np.array_equal(beats.downbeat_indices, [0, 4, 8, 12])

def test_beat_counting_regular():
    """Test beat counting for a regular sequence."""
    # Need meter * min_measures beats = 4 * 2 = 8 beats minimum
    beats = create_test_beats(meter=4, num_beats=8, interval=0.5, min_measures=2)
    # Timestamps: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5
    # Downbeats: indices 0, 4
    # Expected counts: 1, 2, 3, 4, 1, 2, 3, 4
    expected_counts = [1, 2, 3, 4, 1, 2, 3, 4]
    assert len(beats.beat_list) == 8
    for i, beat_info in enumerate(beats.beat_list):
        assert beat_info.beat_count == expected_counts[i]
        # Test time-based lookup as well (check time just after the beat starts)
        assert beats.get_beat_count_at_time(beat_info.timestamp + 0.01) == expected_counts[i]

def test_downbeat_detection():
    """Test downbeat identification."""
    # Need meter * min_measures beats = 3 * 2 = 6 beats minimum
    beats = create_test_beats(meter=3, num_beats=9, interval=0.6, min_measures=2)
    # Downbeats at indices 0, 3, 6
    expected_downbeats = [True, False, False, True, False, False, True, False, False]
    assert len(beats.beat_list) == 9
    for i, beat_info in enumerate(beats.beat_list):
        assert beat_info.is_downbeat == expected_downbeats[i]
        assert beats.is_downbeat_at_time(beat_info.timestamp + 0.01) == expected_downbeats[i]
    assert np.array_equal(beats.downbeat_indices, [0, 3, 6])

def test_irregular_interval_beats():
    """Test identification of beats with irregular intervals."""
    # Create timestamps with a jump
    timestamps = np.array([0.5, 1.0, 1.5, 2.5, 3.0, 3.5]) # Irregular interval between index 2 and 3 (1.5 -> 2.5)
    downbeats = np.array([0, 3]) # Meter doesn't strictly matter here, focus on interval
    meter = 3 
    # Need 3 * 1 = 3 beats minimum. We have 6.
    beats = Beats.from_timestamps(timestamps, downbeats, meter, tolerance_percent=10.0, min_consistent_measures=1)
    
    # Median interval is 0.5. Tolerance interval is 0.05.
    # Intervals: 0.5, 0.5, 1.0, 0.5, 0.5
    # Irregularities: F, F, F, T, F, F (irregular_interval[i] corresponds to beat i)
    expected_irregular_interval = [False, False, False, True, False, False]
    
    assert len(beats.beat_list) == 6
    irregular_indices = []
    for i, beat_info in enumerate(beats.beat_list):
        assert beat_info.is_irregular_interval == expected_irregular_interval[i]
        if beat_info.is_irregular_interval:
            irregular_indices.append(i)
            
    # Beat 3 should be irregular due to interval
    assert irregular_indices == [3]
    assert beats.beat_list[3].is_irregular == True
    assert beats.is_irregular_at_time(beats.beat_list[3].timestamp + 0.01) == True
    # Check a regular one
    assert beats.beat_list[2].is_irregular == False 
    assert beats.is_irregular_at_time(beats.beat_list[2].timestamp + 0.01) == False 

def test_irregular_count_beats():
    """Test identification of beats with irregular counts (exceeding meter)."""
    # Create timestamps where downbeats are further apart than the meter suggests
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]) # 7 beats
    downbeats = np.array([0, 5]) # Downbeats at index 0 and 5
    meter = 4 # But downbeats imply meter 5
    # Need 4 * 1 = 4 beats minimum. We have 7.
    beats = Beats.from_timestamps(timestamps, downbeats, meter, tolerance_percent=10.0, min_consistent_measures=1)
    
    # Expected counts: 1, 2, 3, 4, 5(irregular), 1, 2
    # Expected display counts: 1, 2, 3, 4, 1, 1, 2 
    # Expected irregular count flag: F, F, F, F, T, F, F
    expected_irregular_count = [False, False, False, False, True, False, False]
    expected_display_counts = [1, 2, 3, 4, 1, 1, 2]

    assert len(beats.beat_list) == 7
    irregular_indices = []
    for i, beat_info in enumerate(beats.beat_list):
        assert beat_info.is_irregular_count == expected_irregular_count[i], f"Beat {i}"
        assert beat_info.beat_count == expected_display_counts[i], f"Beat {i}"
        if beat_info.is_irregular_count:
            irregular_indices.append(i)
            
    # Beat 4 should be irregular due to count
    assert irregular_indices == [4]
    assert beats.beat_list[4].is_irregular == True
    assert beats.is_irregular_at_time(beats.beat_list[4].timestamp + 0.01) == True
    # Check a regular one
    assert beats.beat_list[3].is_irregular == False
    assert beats.is_irregular_at_time(beats.beat_list[3].timestamp + 0.01) == False

def test_beat_info_access():
    """Test accessing BeatInfo objects at specific times."""
    # Need 4 * 2 = 8 beats minimum
    beats = create_test_beats(meter=4, num_beats=8, interval=0.5, min_measures=2)
    # Time 1.6s is between beat 3 (1.5s) and beat 4 (2.0s)
    # Should return info for beat 3 (index 3)
    beat_info = beats.get_beat_info_at_time(1.6)
    assert beat_info is not None
    assert beat_info.index == 3
    assert beat_info.timestamp == 1.5
    assert beat_info.beat_count == 4 # 1-based count for beat index 3
    assert not beat_info.is_downbeat
    assert not beat_info.is_irregular
    
    # Time before first beat
    assert beats.get_beat_info_at_time(beats.beat_list[0].timestamp - 0.1) is None
    
    # Time exactly on a beat
    beat_info_exact = beats.get_beat_info_at_time(2.0) # Beat 4 (index 4)
    assert beat_info_exact is not None
    assert beat_info_exact.index == 4
    assert beat_info_exact.is_downbeat
    assert beat_info_exact.beat_count == 1

def test_filtering_regular_irregular():
    """Test filtering beats into regular/irregular lists."""
    # Combine interval and count irregularities
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5]) # Irregular interval 2.0->3.0; Irregular count at index 4
    downbeats = np.array([0, 5]) # Meter 4, downbeats imply meter 5
    meter = 4
    # Need 4 * 1 = 4 beats minimum. We have 7.
    beats = Beats.from_timestamps(timestamps, downbeats, meter, tolerance_percent=10.0, min_consistent_measures=1)
    # Irregular Interval: Beat 5 (index 5, timestamp 3.0) is irregular (interval 1.0 vs median 0.5)
    # Irregular Count: Beat 4 (index 4, timestamp 2.0) is irregular (count 5 > meter 4)
    # Beat List Length: 7
    # Expected irregular indices: 4, 5
    
    assert len(beats.beat_list) == 7
    assert np.array_equal(sorted(beats.irregular_beat_indices), [4, 5])
    
    regular_beats = beats.get_regular_beats()
    irregular_beats = beats.get_irregular_beats()
    
    assert len(regular_beats) == 5
    assert len(irregular_beats) == 2
    assert [b.index for b in regular_beats] == [0, 1, 2, 3, 6]
    assert [b.index for b in irregular_beats] == [4, 5]
    
    # Test downbeat filtering
    # Downbeats: indices 0, 5.
    # Beat 5 is irregular. Beat 0 is regular.
    regular_downbeats = beats.get_regular_downbeats()
    irregular_downbeats = beats.get_irregular_downbeats()
    
    assert len(regular_downbeats) == 1
    assert len(irregular_downbeats) == 1
    assert regular_downbeats[0].index == 0
    assert irregular_downbeats[0].index == 5

def test_edge_cases_creation():
    """Test creation with edge cases like 0 or 1 beat (now mostly validation errors)."""
    # Test 0 beats - should fail validation if min_consistent_measures > 0
    with pytest.raises(BeatCalculationError, match="Insufficient number of beats"):
        Beats.from_timestamps(np.array([]), np.array([]), 4, min_consistent_measures=1)

    # Test 1 beat - should fail validation if min_consistent_measures > 0
    with pytest.raises(BeatCalculationError, match="Insufficient number of beats"):
        Beats.from_timestamps(np.array([0.5]), np.array([0]), 4, min_consistent_measures=1)
        
    # Test case that *passes* validation but has only 1 beat interval (num_beats=2)
    # Requires min_consistent_measures * meter <= 2
    try:
        beats_2 = Beats.from_timestamps(np.array([0.5, 1.0]), np.array([0]), 2, min_consistent_measures=1)
        assert len(beats_2.beat_list) == 2
        assert beats_2.stats.total_beats == 2
    except BeatCalculationError as e:
        pytest.fail(f"Two-beat case failed unexpectedly: {e}")

def test_validation_errors():
    """Test specific validation errors raised by from_timestamps."""
    timestamps = np.arange(20) * 0.5
    downbeats = np.arange(0, 20, 4)
    
    # Invalid meter
    with pytest.raises(BeatCalculationError, match="Invalid meter provided: 0"):
        Beats.from_timestamps(timestamps, downbeats, 0)
    with pytest.raises(BeatCalculationError, match="Invalid meter provided: -1"):
        Beats.from_timestamps(timestamps, downbeats, -1)
        
    # Insufficient beats for min_consistent_measures
    with pytest.raises(BeatCalculationError, match="Insufficient number of beats \\(20\\) for analysis with meter 4. Requires at least 24 beats \\(6 measures\\)."):
        Beats.from_timestamps(timestamps, downbeats, 4, min_consistent_measures=6)
        
    # Non-positive median interval
    # Corrected bad_timestamps (again) to ensure median interval is 0
    bad_timestamps = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 1.5]) 
    # Need enough beats to pass initial validation (meter * min_measures)
    # Here meter=4, min_measures=1 -> need 4 beats. bad_timestamps has 6.
    # Using just bad_timestamps is sufficient if min_consistent_measures=1
    bad_downbeats = np.array([0, 3]) # Example downbeats for this short array
    with pytest.raises(BeatCalculationError, match="Median interval is 0.0000"):
        # Use bad_timestamps directly, assuming min_consistent_measures=1 is sufficient for the test setup
        Beats.from_timestamps(bad_timestamps, bad_downbeats, 4, min_consistent_measures=1) 