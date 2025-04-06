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
    # Simple downbeats every 'meter' beats (used only to generate counts now)
    
    # Generate corresponding beat counts (cycling 1 to meter)
    beat_counts = np.array([(i % meter) + 1 for i in range(num_beats)])

    # Ensure enough beats for the default min_measures in from_timestamps
    # Note: from_timestamps itself raises error if not enough, this helper just ensures
    #       the *inputs* to from_timestamps are sufficient for the *helper's* default params.
    required = meter * min_measures 
    if num_beats < required:
        # Adjust num_beats if the defaults don't meet the minimum requirement
        timestamps = np.arange(required) * interval
        num_beats = required # Update num_beats to match
        # Regenerate counts for the adjusted num_beats
        beat_counts = np.array([(i % meter) + 1 for i in range(num_beats)])

    # This might still raise BeatCalculationError if constraints aren't met,
    # which is expected behavior for some tests.
    return Beats.from_timestamps(
        timestamps=timestamps,
        meter=meter,
        beat_counts=beat_counts, # Pass the generated counts
        tolerance_percent=tolerance,
        min_measures=min_measures
    )

# Test Cases

def test_beat_creation_and_properties():
    """Test basic Beats object creation and properties."""
    beats = create_test_beats() # Use the helper
    
    assert isinstance(beats, Beats)
    assert beats.meter == 4
    assert len(beats.beat_list) == 20 # Default num_beats in helper
    assert isinstance(beats.overall_stats, BeatStatistics)
    assert isinstance(beats.regular_stats, BeatStatistics)
    assert np.isclose(beats.overall_stats.median_interval, 0.5)
    assert beats.overall_stats.total_beats == 20
    # Regular section might not be the full 20 if min_measures affects it
    # Default min_measures is 2, meter 4 -> needs 8 beats. Helper ensures this.
    # The sequence finder should find the whole sequence [0..19] as regular.
    assert beats.start_regular_beat_idx == 0
    assert beats.end_regular_beat_idx == 20
    assert beats.regular_stats.total_beats == 20
    assert np.isclose(beats.regular_stats.median_interval, 0.5)
    
    # Test properties
    assert len(beats.timestamps) == 20
    # Check downbeats derived from beat_counts (indices 0, 4, 8, 12, 16)
    expected_downbeat_indices = [0, 4, 8, 12, 16]
    actual_downbeat_indices = [b.index for b in beats.beat_list if b.is_downbeat]
    assert actual_downbeat_indices == expected_downbeat_indices
    # No irregular intervals expected in default helper
    assert len(beats.irregular_beat_indices) == 0 

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
        count, _, _ = beats.get_info_at_time(beat_info.timestamp + 0.01)
        assert count == expected_counts[i]

def test_downbeat_detection():
    """Test downbeat identification."""
    # Need meter * min_measures beats = 3 * 2 = 6 beats minimum
    beats = create_test_beats(meter=3, num_beats=9, interval=0.6, min_measures=2)
    # Expected downbeats based on beat_counts [1,2,3,1,2,3,1,2,3]: indices 0, 3, 6
    expected_downbeats = [True, False, False, True, False, False, True, False, False]
    assert len(beats.beat_list) == 9
    actual_downbeat_indices = []
    for i, beat_info in enumerate(beats.beat_list):
        assert beat_info.is_downbeat == expected_downbeats[i]
        # Test using get_info_at_time and checking if count == 1 for downbeats
        count, _, _ = beats.get_info_at_time(beat_info.timestamp + 0.01)
        assert (count == 1) == expected_downbeats[i]
        if beat_info.is_downbeat:
             actual_downbeat_indices.append(i)
    # Check the collected indices match expectation
    assert actual_downbeat_indices == [0, 3, 6]

def test_irregular_interval_beats():
    """Test identification of beats with irregular intervals."""
    # Create timestamps with a jump
    timestamps = np.array([0.5, 1.0, 1.5, 2.5, 3.0, 3.5]) # Irregular interval between index 2 and 3 (1.5 -> 2.5)
    meter = 3 
    # Need 3 * 1 = 3 beats minimum. We have 6.
    # Assume regular counting based on meter 3: [1, 2, 3, 1, 2, 3]
    beat_counts_irr_interval = np.array([1, 2, 3, 1, 2, 3]) 
    expected_irregular_interval = [False, False, False, True, False, False]
    
    # Pass args by keyword to avoid misinterpretation
    beats = Beats.from_timestamps(timestamps, meter, beat_counts_irr_interval, tolerance_percent=10.0, min_measures=1)
    
    # Median interval is 0.5. Tolerance interval is 0.05.
    # Intervals: 0.5, 0.5, 1.0, 0.5, 0.5
    # Irregularities: F, F, F, T, F, F (irregular_interval[i] corresponds to beat i)
    # We need to provide beat_counts now
    # Assume regular counting based on meter 3, despite downbeats: [1, 2, 3, 1, 2, 3]
    beat_counts_irr_interval = np.array([1, 2, 3, 1, 2, 3]) 
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
    # Check if beat is irregular using get_info_at_time
    count, _, _ = beats.get_info_at_time(beats.beat_list[3].timestamp + 0.01)
    assert count == 0  # Irregular beats have count 0
    # Check a regular one
    assert beats.beat_list[2].is_irregular == False 
    # Check if beat is regular using get_info_at_time
    count, _, _ = beats.get_info_at_time(beats.beat_list[2].timestamp + 0.01)
    assert count > 0  # Regular beats have count > 0

    # Check that overall irregularity reflects the FINAL count of irregular beats (1 out of 6)
    irregularity_percent = beats.overall_stats.irregularity_percent
    expected_percent = (1 / 6) * 100
    assert np.isclose(irregularity_percent, expected_percent, atol=0.01)

def test_irregular_count_beats():
    """Test identification of beats with irregular counts (exceeding meter)."""
    # Create timestamps where downbeats are further apart than the meter suggests
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]) # 7 beats
    meter = 4 # But downbeats imply meter 5
    # Need 4 * 1 = 4 beats minimum. We have 7.
    # Provide the beat counts that would result from the downbeats (1,2,3,4,5,1,2)
    beat_counts_irr = np.array([1, 2, 3, 4, 5, 1, 2])

    # Pass args by keyword
    beats = Beats.from_timestamps(timestamps, meter, beat_counts_irr, tolerance_percent=10.0, min_measures=1)
    
    # Expected original counts: 1, 2, 3, 4, 5(irregular), 1, 2
    # Expected display counts: 1, 2, 3, 4, 0, 1, 2 
    # Expected irregular flag: F, F, F, F, T, F, F
    expected_irregular = [False, False, False, False, True, False, False]
    expected_display_counts = [1, 2, 3, 4, 0, 1, 2]

    assert len(beats.beat_list) == 7
    irregular_indices = []
    for i, beat_info in enumerate(beats.beat_list):
        assert beat_info.is_irregular == expected_irregular[i], f"Beat {i}"
        assert beat_info.beat_count == expected_display_counts[i], f"Beat {i}"
        if beat_info.is_irregular:
            irregular_indices.append(i)
            
    # Beat 4 should be irregular due to count
    assert irregular_indices == [4]
    assert beats.beat_list[4].is_irregular == True
    # Check if beat is irregular using get_info_at_time
    count, _, _ = beats.get_info_at_time(beats.beat_list[4].timestamp + 0.01)
    assert count == 0  # Irregular beats have count 0
    # Check a regular one
    assert beats.beat_list[3].is_irregular == False
    # Check if beat is regular using get_info_at_time
    count, _, _ = beats.get_info_at_time(beats.beat_list[3].timestamp + 0.01)
    assert count > 0  # Regular beats have count > 0
    
    # Check that overall irregularity reflects the FINAL count of irregular beats (1 out of 7)
    irregularity_percent = beats.overall_stats.irregularity_percent
    expected_percent = (1 / 7) * 100
    assert np.isclose(irregularity_percent, expected_percent, atol=0.01)

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

def test_get_info_at_time():
    """Test the get_info_at_time method returns correct count and time since beat."""
    # Create test beats with 4/4 time signature, 8 beats at 0.5s intervals
    beats = create_test_beats(meter=4, num_beats=8, interval=0.5, min_measures=2)
    
    # Test at exact beat times
    for i in range(8):
        time_point = i * 0.5  # Beat times: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5
        expected_count = (i % 4) + 1  # Expected counts: 1, 2, 3, 4, 1, 2, 3, 4
        count, time_since, beat_idx = beats.get_info_at_time(time_point)
        assert count == expected_count
        assert time_since == 0.0  # Exactly on the beat, so time_since should be 0
        assert beat_idx == i  # Beat index should match i
    
    # Test between beats
    count, time_since, beat_idx = beats.get_info_at_time(1.25)  # Between beats at 1.0 and 1.5
    assert count == 3  # Should be beat count 3
    assert abs(time_since - 0.25) < 1e-6  # Should be 0.25 seconds after the beat
    assert beat_idx == 2  # Should be beat at index 2 (timestamp 1.0)
    
    # Test time before first beat
    count, time_since, beat_idx = beats.get_info_at_time(-0.1)  # Before first beat
    assert count == 0  # No beat count before first beat
    assert time_since == 0.0  # No prior beat
    assert beat_idx == -1  # No valid beat index
    
    # Test with irregular beats
    # Create a beat sequence with irregular beats at the beginning
    irregular_beats = create_test_beats(meter=4, num_beats=8, interval=0.5, min_measures=2)
    irregular_beats.start_regular_beat_idx = 4  # Make first 4 beats irregular
    
    # Check irregular beats return count 0
    for i in range(4):
        time_point = i * 0.5  # First 4 beats: 0.0, 0.5, 1.0, 1.5
        count, time_since, beat_idx = irregular_beats.get_info_at_time(time_point)
        assert count == 0  # Irregular beats should have count 0
        assert abs(time_since - 0.0) < 1e-6  # Time since should still be 0 at the exact beat time
        assert beat_idx == i  # Beat index should still be correct
    
    # Check regular beats still return proper counts
    for i in range(4, 8):
        time_point = i * 0.5  # Last 4 beats: 2.0, 2.5, 3.0, 3.5
        expected_count = (i % 4) + 1  # Expected counts: 1, 2, 3, 4
        count, time_since, beat_idx = irregular_beats.get_info_at_time(time_point)
        assert count == expected_count
        assert abs(time_since - 0.0) < 1e-6
        assert beat_idx == i

def test_filtering_regular_irregular():
    """Test filtering beats into regular/irregular lists."""
    # Combine interval and count irregularities
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5]) # Irregular interval 2.0->3.0; Irregular count at index 4
    meter = 4
    # Need 4 * 1 = 4 beats minimum. We have 7.
    # Provide plausible beat counts, e.g., reflecting the (now removed) downbeats
    beat_counts_filter = np.array([1, 2, 3, 4, 5, 1, 2]) # Use counts that cause both types of irregularity
    # Pass args by keyword
    beats = Beats.from_timestamps(timestamps, meter, beat_counts=beat_counts_filter, tolerance_percent=10.0, min_measures=1)
    # Irregular Interval: Beat 5 (index 5, timestamp 3.0) is irregular (interval 1.0 vs median 0.5)
    # Irregular Count: Beat 4 (index 4, timestamp 2.0) has count 5 > meter 4 -> becomes 0 in BeatInfo
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
    """Test edge cases in beat creation."""
    
    # Test with zero beats (should fail due to insufficient beats)
    with pytest.raises(BeatCalculationError, match="Insufficient number of beats"):
        Beats.from_timestamps(np.array([]), meter=4, beat_counts=np.array([]), min_measures=1)
        
    # Test with one beat (should fail due to insufficient beats)
    with pytest.raises(BeatCalculationError, match="Insufficient number of beats"):
        Beats.from_timestamps(np.array([0.5]), meter=4, beat_counts=np.array([1]), min_measures=1)
        
    # Test with meter=1 (should fail)
    with pytest.raises(BeatCalculationError, match="Meter must be 2 or greater"):
        Beats.from_timestamps(np.arange(4) * 0.5, meter=1, beat_counts=np.array([1,1,1,1]), min_measures=1)
        
    # Test with negative tolerance (should fail)
    with pytest.raises(BeatCalculationError, match="Invalid tolerance_percent"):
        Beats.from_timestamps(np.arange(8) * 0.5, meter=4, beat_counts=(np.arange(8) % 4) + 1, tolerance_percent=-5.0)

    # Test case with non-strictly increasing timestamps
    timestamps_not_increasing = np.array([0.0, 0.5, 0.5, 0.8, 0.7, 1.0]) # Duplicate at index 1->2, decrease at 3->4
    counts_not_increasing = np.array([1, 2, 2, 3, 3, 4]) # Match length
    # Expect the timestamp validation error
    with pytest.raises(BeatCalculationError, match=r"Timestamps must be strictly increasing. Error found after index 1"):
        Beats.from_timestamps(timestamps_not_increasing, meter=4, beat_counts=counts_not_increasing, min_measures=1)

# === New/Modified Tests for Automatic Regular Section Detection ===

def test_regular_section_detection_full():
    """Test regular section detection when the whole track is regular."""
    beats = create_test_beats(meter=4, num_beats=16, interval=0.5, min_measures=3)
    assert beats.start_regular_beat_idx == 0
    assert beats.end_regular_beat_idx == 16
    assert beats.regular_stats.total_beats == 16
    assert np.isclose(beats.regular_stats.median_interval, 0.5)

def test_regular_section_detection_intro_outro():
    """Test detection with irregular intro and outro."""
    timestamps = np.array([0.0, 1.0, # Irregular intro
                          1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, # Regular middle (8 beats)
                          6.0, 6.5]) # Irregular outro
    meter_io = 4
    min_measures_io = 2 # Requires 8 beats
    # Intervals: 1.0(i), 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5(r), 1.0(i), 0.5. Median=0.5. Tol=0.05.
    # Beat Counts: 0, 0, 1,2,3,4, 1,2,3,4, 0, 0
    # Provide counts: (Indices: 0, 1,  2,3,4,5, 6,7,8,9, 10,11)
    counts_io = np.array([0, 0,  1,2,3,4, 1,2,3,4,  0, 0])
    # Check sequence finder (req 8 beats):
    # BeatInfo Counts: [0, 0, 1,2,3,4, 1,2,3,4, 0, 0]
    # Seq [2..9]: Len 8. Intervals 0.5 (7 times). Irreg=0%. Valid.
    # Pass args by keyword
    beats = Beats.from_timestamps(timestamps, meter=meter_io, beat_counts=counts_io, tolerance_percent=10.0, min_measures=min_measures_io)
    assert beats.start_regular_beat_idx == 2
    assert beats.end_regular_beat_idx == 10 # Exclusive index for beat 9
    assert beats.regular_stats.total_beats == 8
    assert np.isclose(beats.regular_stats.median_interval, 0.5)
    assert beats.regular_stats.irregularity_percent == 0.0

def test_regular_section_detection_insufficient():
    """Test failure when no sufficiently long regular section exists."""
    timestamps = np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, 5.0]) # Irregular gaps
    meter_insuf = 3
    min_measures_insuf = 2 # Requires 6 beats
    # Intervals: 1.0(i), 0.5, 0.5, 0.5(r), 1.0(i), 0.5, 1.0(i). Median=0.5. Tol=0.05
    # Provide beat counts based on description
    counts_insuf = np.array([0, 1, 2, 3, 1, 2, 0, 0])
    # Beat Counts: 0, 1,2,3, 1,2,0, 0
    # Check sequence finder (req 6):
    # Seq [1..3]: Len 3. Invalid.
    # Seq [4..5]: Len 2. Invalid.
    with pytest.raises(BeatCalculationError, match="Could not determine a stable regular section: Longest regular sequence found .* is shorter than required"):
        # Pass args by keyword, including counts
        Beats.from_timestamps(timestamps, meter=meter_insuf, beat_counts=counts_insuf, tolerance_percent=10.0, min_measures=min_measures_insuf)

def test_regular_section_with_count_irregularities():
    """Test that beats with irregular counts break the regular sequence."""
    timestamps = np.arange(12) * 0.5 # Regular intervals
    meter = 4
    min_measures = 1 # Requires 4 beats
    # Beat Counts: 1,2,3,4, 5(irr), 1,2,3,4, 5(irr), 1,2
    counts_irr_c = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2])
    # Check sequence finder (req 4):
    # Seq [0..3]: Len 4. Irreg=0%. Valid.
    # Seq [5..8]: Len 4. Irreg=0%. Valid.
    # Seq [10..11]: Len 2. Invalid.
    # Longest valid is length 4. Should pick the first one found: [0..3]
    # Pass args by keyword
    beats = Beats.from_timestamps(timestamps, meter=meter, beat_counts=counts_irr_c, tolerance_percent=10.0, min_measures=min_measures)
    assert beats.start_regular_beat_idx == 0
    assert beats.end_regular_beat_idx == 4
    assert beats.regular_stats.total_beats == 4
    assert np.isclose(beats.regular_stats.median_interval, 0.5) 

def test_regular_sequence_starts_from_downbeat():
    """Test that the longest regular sequence always starts from a downbeat."""
    # Create a sequence of beats with the following structure:
    # - First 3 beats: regular intervals but NOT starting with a downbeat
    # - Next 6 beats: regular intervals starting with a downbeat
    # - Last 3 beats: irregular
    
    # Create timestamps with regular intervals
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.7])  # Last beat irregular

    # Create beat counts where the first beat is NOT a downbeat
    # First 3 beats: 2,3,4 (not starting with a downbeat)
    # Next 6 beats: 1,2,3,4,1,2 (starting with a downbeat at position 3)
    # Last 3 beats: 3,4,1 (with irregular interval for the last beat)
    beat_counts = np.array([2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1])
    
    beats = Beats.from_timestamps(
        timestamps=timestamps,
        meter=4,
        beat_counts=beat_counts,
        tolerance_percent=10.0,
        min_measures=2  # Require at least 2 complete measures
    )
    
    # Regular sequence should start at index 3 (beat_count = 1) and end at index 8
    assert beats.start_regular_beat_idx == 3
    
    # Even though the first 3 beats have regular intervals, they don't start with a downbeat,
    # so the algorithm should prefer the sequence starting at index 3
    
    # Create another test case with multiple potential downbeat-starting sequences
    # but different lengths
    timestamps2 = np.array([
        0.0, 0.5, 1.0, 1.5,  # 4 regular beats starting with downbeat
        2.0, 2.7, 3.2, 3.7,  # 4 irregular beats
        4.0, 4.5, 5.0, 5.5, 6.0, 6.5  # 6 regular beats starting with downbeat
    ])
    
    beat_counts2 = np.array([
        1, 2, 3, 4,          # First sequence (starts with downbeat)
        1, 2, 3, 4,          # Irregular sequence (but still has downbeat)
        1, 2, 3, 4, 1, 2     # Longest sequence (starts with downbeat)
    ])
    
    beats2 = Beats.from_timestamps(
        timestamps=timestamps2,
        meter=4,
        beat_counts=beat_counts2,
        tolerance_percent=10.0,
        min_measures=1  # Only require 1 measure for this test
    )
    
    # The algorithm should choose the longest sequence starting with a downbeat
    assert beats2.start_regular_beat_idx == 8  # Start of the third sequence (longest)
    assert beats2.end_regular_beat_idx == 14   # End of the third sequence (inclusive)