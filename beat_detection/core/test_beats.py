"""
Tests for the Beats data structure and its core logic.
"""

import numpy as np
import pytest

from beat_detection.core.beats import Beats, BeatStatistics, BeatCalculationError

# Helper function to create a standard Beats object for testing
# Note: Using simple, predictable values
def create_test_beats(beats_per_bar=4, num_beats=20, interval=0.5, tolerance=10.0, min_measures=2) -> Beats:
    """Creates a predictable Beats object for logic tests."""
    timestamps = np.arange(num_beats) * interval
    # Simple downbeats every 'beats_per_bar' beats (used only to generate counts now)
    
    # Generate corresponding beat counts (cycling 1 to beats_per_bar)
    beat_counts = np.array([(i % beats_per_bar) + 1 for i in range(num_beats)])

    # Ensure enough beats for the default min_measures in from_timestamps
    # Note: from_timestamps itself raises error if not enough, this helper just ensures
    #       the *inputs* to from_timestamps are sufficient for the *helper's* default params.
    required = beats_per_bar * min_measures 
    if num_beats < required:
        # Adjust num_beats if the defaults don't meet the minimum requirement
        timestamps = np.arange(required) * interval
        num_beats = required # Update num_beats to match
        # Regenerate counts for the adjusted num_beats
        beat_counts = np.array([(i % beats_per_bar) + 1 for i in range(num_beats)])

    # This might still raise BeatCalculationError if constraints aren't met,
    # which is expected behavior for some tests.
    return Beats.from_timestamps(
        timestamps=timestamps,
        beats_per_bar=beats_per_bar,
        beat_counts=beat_counts, # Pass the generated counts
        tolerance_percent=tolerance,
        min_measures=min_measures
    )

# Test Cases

def test_beat_creation_and_properties():
    """Test basic Beats object creation and properties."""
    beats = create_test_beats() # Use the helper
    
    assert isinstance(beats, Beats)
    assert beats.beats_per_bar == 4
    assert beats.overall_stats.total_beats == 20 # Default num_beats in helper
    assert isinstance(beats.overall_stats, BeatStatistics)
    assert isinstance(beats.regular_stats, BeatStatistics)
    assert np.isclose(beats.overall_stats.median_interval, 0.5)
    assert beats.overall_stats.total_beats == 20
    assert beats.start_regular_beat_idx == 0
    assert beats.end_regular_beat_idx == 20
    assert beats.regular_stats.total_beats == 20
    assert np.isclose(beats.regular_stats.median_interval, 0.5)
    
    # Test properties
    assert isinstance(beats.beat_data, np.ndarray)
    assert beats.beat_data.shape == (20, 2)
    assert len(beats.timestamps) == 20
    assert len(beats.counts) == 20
    assert np.all(beats.timestamps == beats.beat_data[:, 0])
    assert np.all(beats.counts == beats.beat_data[:, 1])
    
    expected_downbeat_indices = [0, 4, 8, 12, 16]
    # Access counts directly from beat_data
    actual_downbeat_indices = np.where(beats.beat_data[:, 1] == 1)[0].tolist()
    assert actual_downbeat_indices == expected_downbeat_indices
    assert len(beats.irregular_beat_indices) == 0 
    assert isinstance(beats.irregular_beat_indices, np.ndarray) # Check type

def test_beat_counting_regular():
    """Test beat counting for a regular sequence."""
    beats = create_test_beats(beats_per_bar=4, num_beats=8, interval=0.5, min_measures=2)
    expected_counts = [1, 2, 3, 4, 1, 2, 3, 4]
    assert beats.overall_stats.total_beats == 8
    # Check counts directly
    np.testing.assert_array_equal(beats.counts, expected_counts)
    
    # Test get_info_at_time consistency
    for i, ts in enumerate(beats.timestamps):
        count, _, _ = beats.get_info_at_time(ts + 0.01)
        assert count == expected_counts[i]
        # Check if it correctly identifies downbeats
        assert (count == 1) == (expected_counts[i] == 1)

def test_irregular_beats_marked_as_zero_count():
    """Test that beats outside the regular interval sequence or with invalid input counts are marked with count 0."""
    # Scenario 1: Irregular interval causes sequence break
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0]) # Irregular interval 2.5 -> 3.5 (1.0s vs median 0.5s)
    beats_per_bar = 4
    counts_input = np.array([1, 2, 3, 4, 1, 2, 1, 2]) # Counts are sequentially valid initially
    # Expected regular sequence: 0-5 (len 6). min_measures=1 requires 4 beats.
    beats = Beats.from_timestamps(timestamps, beats_per_bar, counts_input, tolerance_percent=10.0, min_measures=1)
    
    assert beats.overall_stats.total_beats == 8
    # Regular section should be 0-5 (exclusive end 6)
    assert beats.start_regular_beat_idx == 0
    assert beats.end_regular_beat_idx == 6 
    
    # Beat 6 (ts 3.5) follows an irregular interval and breaks the sequence.
    # Beat 7 (ts 4.0) also follows an irregular interval.
    # Although input counts were valid, they fall outside the *identified* longest regular sequence.
    # The from_timestamps processes counts first (all valid initially), then finds sequence 0-5.
    # get_info_at_time should return 0 for beats outside this.
    
    # Check beat 6 (index 6)
    count_irr_interval, _, idx_irr_interval = beats.get_info_at_time(3.5 + 0.01) 
    assert count_irr_interval == 0 # Outside regular section
    assert idx_irr_interval == 6
    
    # Check overall irregularity percentage (based on processed counts == 0) - should be 0 initially
    assert beats.overall_stats.irregularity_percent == 0.0 
    # Check irregular_beat_indices property (based on processed counts == 0) - should be empty
    assert len(beats.irregular_beat_indices) == 0 
    # Note: Irregularity is now mainly about the regular section bounds identified,
    # not flags on individual beats or overall stats directly reflecting interval issues.

    # Scenario 2: Irregular input count
    timestamps_c = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    beats_per_bar_c = 4
    beat_counts_irr_c = np.array([1, 2, 3, 4, 5, 1, 2]) # Invalid count 5 at index 4
    beats_c = Beats.from_timestamps(timestamps_c, beats_per_bar_c, beat_counts_irr_c, tolerance_percent=10.0, min_measures=1)
    
    # Expected processed counts in beat_data: [1, 2, 3, 4, 0, 1, 2]
    expected_processed_counts = [1, 2, 3, 4, 0, 1, 2]
    np.testing.assert_array_equal(beats_c.counts, expected_processed_counts)
    assert beats_c.overall_stats.total_beats == 7
    # The irregular count (index 4 -> 0) breaks the sequence finding.
    # Longest regular sequence should be index 0-3 (len 4). min_measures=1 requires 4.
    assert beats_c.start_regular_beat_idx == 0
    assert beats_c.end_regular_beat_idx == 4
    
    # Check irregular_beat_indices property (where count == 0)
    np.testing.assert_array_equal(beats_c.irregular_beat_indices, [4]) 
    
    # Check get_info_at_time for the irregular beat (index 4)
    count_irr, _, idx_irr = beats_c.get_info_at_time(2.0 + 0.01)
    # Should be 0 both because count is 0 AND it's outside (>= end_regular_beat_idx)
    assert count_irr == 0 
    assert idx_irr == 4
    
    # Check get_info_at_time for a beat after the irregularity (index 5)
    count_after, _, idx_after = beats_c.get_info_at_time(2.5 + 0.01)
    assert count_after == 0 # Outside regular section
    assert idx_after == 5
    
    # Check overall irregularity percentage (based on count==0)
    irregularity_percent = beats_c.overall_stats.irregularity_percent
    expected_percent = (1 / 7) * 100 # One beat has count 0
    assert np.isclose(irregularity_percent, expected_percent, atol=0.01)

def test_get_info_at_time():
    """Test the get_info_at_time method returns correct count and time since beat."""
    beats = create_test_beats(beats_per_bar=4, num_beats=8, interval=0.5, min_measures=2)
    # Expected counts [1, 2, 3, 4, 1, 2, 3, 4]
    # Regular section 0-7 (exclusive end 8)
    assert beats.start_regular_beat_idx == 0
    assert beats.end_regular_beat_idx == 8
    
    for i in range(8):
        time_point = i * 0.5
        expected_count = (i % 4) + 1
        count, time_since, beat_idx = beats.get_info_at_time(time_point)
        assert count == expected_count # Should get actual count as it's within regular section
        assert time_since == 0.0
        assert beat_idx == i
    
    count, time_since, beat_idx = beats.get_info_at_time(1.25) # Between beat 2 (1.0s) and 3 (1.5s)
    assert beat_idx == 2
    assert count == 3 # Beat 2 is regular
    assert abs(time_since - 0.25) < 1e-6
    
    count, time_since, beat_idx = beats.get_info_at_time(-0.1) # Before first beat
    assert count == 0
    assert time_since == 0.0
    assert beat_idx == -1
    
    # Test with irregular beats (outside identified regular section)
    timestamps = np.array([0.0, 0.2, 0.7, 1.2, 1.7, 2.2, 2.7, 3.2, 3.7, 4.0, 4.2]) # Irregular intervals at start/end
    # Input counts - assume some are marked 0 initially due to external logic, others valid
    counts = np.array([0, 0, 1, 2, 3, 4, 1, 2, 3, 0, 0]) 
    # min_measures=1 requires 4 beats. Median interval approx 0.5.
    # Sequence finder looks for first '1', finds it at index 2.
    # Checks interval 0.7->1.2 (0.5, ok), 1.2->1.7 (0.5, ok), ..., 3.2->3.7 (0.5, ok)
    # Checks interval 3.7->4.0 (0.3, irregular). Sequence ends at index 8 (ts 3.7).
    # Longest sequence: indices 2 to 8 (inclusive), length 7.
    beats = Beats.from_timestamps(timestamps, 4, counts, min_measures=1, tolerance_percent=10.0) 
    
    assert beats.start_regular_beat_idx == 2
    assert beats.end_regular_beat_idx == 9 # Exclusive index
    
    # Check a beat before the regular section (index 1, ts 0.2)
    count, time_since, beat_idx = beats.get_info_at_time(0.2 + 0.01)
    assert beat_idx == 1
    assert count == 0 # Count is 0 because it's outside regular section start
    assert abs(time_since - 0.01) < 1e-6
    
    # Check a beat after the regular section (index 9, ts 4.0)
    count, time_since, beat_idx = beats.get_info_at_time(4.0 + 0.01)
    assert beat_idx == 9
    assert count == 0 # Count is 0 because it's outside regular section end
    assert abs(time_since - 0.01) < 1e-6
    
    # Check a beat within the regular section (index 5, ts 2.2, count 4)
    count, time_since, beat_idx = beats.get_info_at_time(2.2 + 0.01)
    assert beat_idx == 5
    assert count == 4 # Should have its correct count (non-zero, within bounds)
    assert abs(time_since - 0.01) < 1e-6
    
    # Check a beat within the regular section BUT with count 0 (index 9, ts 4.0 - wait, this is OUTSIDE section)
    # Let's modify the example slightly: Regular section includes a beat with count 0.
    timestamps_z = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]) # All regular intervals
    counts_z = np.array([1, 2, 0, 4, 1, 2, 3, 4]) # Irregular count 0 at index 2
    # Sequence finding: Starts at 0 (count 1). Index 1 (count 2) is ok. Index 2 (count 0) breaks sequence. Longest seq 0-1.
    # Let's try again: Start at 4 (count 1). Index 5 (count 2) ok. Index 6 (count 3) ok. Index 7 (count 4) ok. Longest seq 4-7.
    beats_z = Beats.from_timestamps(timestamps_z, 4, counts_z, min_measures=1)
    assert beats_z.start_regular_beat_idx == 4
    assert beats_z.end_regular_beat_idx == 8
    
    # Check the beat with count 0 (index 2, ts 1.0) - it's outside the found regular section
    count_z, time_since_z, beat_idx_z = beats_z.get_info_at_time(1.0 + 0.01)
    assert beat_idx_z == 2
    assert count_z == 0 # Outside regular section
    assert abs(time_since_z - 0.01) < 1e-6
    
    # Conclusion: get_info_at_time returns 0 if EITHER index is outside start/end OR the stored count is 0.

def test_filtering_regular_irregular_mixed():
    """Test filtering regular and irregular beats correctly based on final beat_data and regular section."""
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0]) # Irregular interval 2.5 -> 3.5
    beats_per_bar = 4
    counts_input = np.array([1, 2, 3, 4, 1, 5, 1, 2]) # Irregular count 5 at index 5
    # Processed counts: [1, 2, 3, 4, 1, 0, 1, 2]
    # Intervals: [0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5]. Median 0.5. Tolerance 0.05.
    # Sequence finder: 
    # Start 0 (count 1). OK until index 4 (count 1). 
    # Check index 5 (count 0): Breaks sequence. Sequence 0-4 (len 5).
    # Check index 6 (count 1): Start new potential sequence. Interval 2.5->3.5=1.0 (irregular). Breaks immediately. 
    # Longest is 0-4.
    # min_measures=1 requires 4 beats. Found sequence 0-4 (len 5) is valid.
    beats = Beats.from_timestamps(timestamps, beats_per_bar, counts_input, tolerance_percent=10.0, min_measures=1)
    
    assert beats.start_regular_beat_idx == 0
    assert beats.end_regular_beat_idx == 5 # Exclusive index
    
    # Irregular indices based on property (where processed count == 0)
    expected_irregular_indices_prop = np.array([5])
    np.testing.assert_array_equal(beats.irregular_beat_indices, expected_irregular_indices_prop)
    
    # Filter beat_data manually for "regular" beats (within bounds AND count > 0)
    is_regular_mask = np.zeros(beats.beat_data.shape[0], dtype=bool)
    is_regular_mask[beats.start_regular_beat_idx:beats.end_regular_beat_idx] = True
    is_regular_mask &= (beats.counts > 0)
    
    regular_indices_manual = np.where(is_regular_mask)[0]
    irregular_indices_manual = np.where(~is_regular_mask)[0]
    
    # Beats 0, 1, 2, 3, 4 are within bounds and have count > 0.
    expected_regular_indices = [0, 1, 2, 3, 4]
    # Beats 5 (count 0), 6 (outside bounds), 7 (outside bounds) are irregular.
    expected_irregular_indices = [5, 6, 7]
  
    np.testing.assert_array_equal(regular_indices_manual, expected_regular_indices)
    np.testing.assert_array_equal(irregular_indices_manual, expected_irregular_indices)
    
    # Check downbeats among the regular beats
    regular_beat_data = beats.beat_data[is_regular_mask]
    regular_downbeat_indices = regular_indices_manual[regular_beat_data[:, 1] == 1]
    expected_regular_downbeat_indices = [0, 4]
    np.testing.assert_array_equal(regular_downbeat_indices, expected_regular_downbeat_indices)

def test_edge_cases_creation():
    """Test edge cases in beat creation."""
    
    # Test with zero beats - Now returns a degenerate object, doesn't raise here.
    # Beats.from_timestamps(np.array([]), 4, np.array([]), min_measures=5)
    # Instead, test that insufficient beats *relative to min_measures* raises.
    
    # Test with few beats failing min_measures requirement
    # The error raised is about the *found* sequence being too short, not the initial count,
    # and it's wrapped by the outer exception.
    # Check only for the exception type
    with pytest.raises(BeatCalculationError):
        # Requires 4*1 = 4 beats, only providing 3. Finds sequence len 3, then fails.
        Beats.from_timestamps(np.array([1.0, 1.5, 2.0]), 4, np.array([1, 2, 3]), min_measures=1)

    # Test with one beat, but min_measures allows it (e.g., min_measures=0 or 1 beat < 4*1)
    # Should NOT raise error here anymore, but return degenerate object.
    try:
        # Requires 4*0=0 beats. Should pass.
        beats_one = Beats.from_timestamps(np.array([1.0]), 4, np.array([1]), min_measures=0) 
        assert beats_one.overall_stats.total_beats == 1
        assert beats_one.start_regular_beat_idx == 0
        assert beats_one.end_regular_beat_idx == 1 # Degenerate section
    except BeatCalculationError as e:
        pytest.fail(f"Should not raise for 1 beat when min_measures=0: {e}")
        
    # Test with two beats, min_measures=1 (requires 4 beats) -> Fails
    # Also raises the "Longest regular sequence... shorter" error after finding len 2 sequence,
    # and it's wrapped by the outer exception.
    # Check only for the exception type
    with pytest.raises(BeatCalculationError):
        Beats.from_timestamps(np.array([1.0, 1.5]), 4, np.array([1, 2]), min_measures=1) 
        
    # Test with enough beats but non-increasing timestamps
    with pytest.raises(BeatCalculationError, match="Timestamps must be strictly increasing"):
        Beats.from_timestamps(np.array([0.0, 1.0, 0.8, 1.5, 2.0]), 4, np.array([1,2,3,4,1]), min_measures=1)

    # Test with invalid tolerance
    with pytest.raises(BeatCalculationError, match="Invalid tolerance_percent"):
        Beats.from_timestamps(np.arange(10) * 0.5, 4, np.ones(10), tolerance_percent=-5.0)
        
    # Test with invalid beats_per_bar
    with pytest.raises(BeatCalculationError, match="Invalid beats_per_bar"):
        Beats.from_timestamps(np.arange(10) * 0.5, 1, np.ones(10))
        
    # Test with mismatched timestamp and counts arrays
    with pytest.raises(BeatCalculationError, match="Timestamp count .* does not match beat_counts count"):
        Beats.from_timestamps(np.arange(5)*0.5, 4, np.array([1,2,3,4]))

def test_regular_section_detection_full():
    """Test when the entire sequence is regular."""
    beats = create_test_beats(num_beats=16, min_measures=2) # 4 measures of 4/4
    assert beats.start_regular_beat_idx == 0
    assert beats.end_regular_beat_idx == 16
    assert beats.regular_stats.total_beats == 16
    assert np.isclose(beats.regular_stats.median_interval, beats.overall_stats.median_interval)

def test_regular_section_detection_intro_outro():
    """Test finding a regular section with irregular start and end."""
    # Regular part: 0.7 to 3.7 (7 beats, indices 2-8)
    timestamps = np.array([0.0, 0.2, 0.7, 1.2, 1.7, 2.2, 2.7, 3.2, 3.7, 4.0, 4.2]) 
    counts = np.array([0, 0, 1, 2, 3, 4, 1, 2, 3, 0, 0]) # Counts match the intended structure
    # min_measures=1 requires 4 beats. Found section 2-8 (len 7) is ok.
    beats = Beats.from_timestamps(timestamps, 4, counts, min_measures=1, tolerance_percent=10.0) 
    
    assert beats.start_regular_beat_idx == 2
    assert beats.end_regular_beat_idx == 9 # Exclusive index of last regular beat + 1
    assert beats.regular_stats.total_beats == 7 # 9 - 2 = 7 beats
    # Regular median should be based on intervals within 0.7 to 3.7 (all 0.5s)
    assert np.isclose(beats.regular_stats.median_interval, 0.5) 
    # Overall median might be skewed by 0.2, 0.3 intervals
    # For this specific dataset, the median of [0.2, 0.2, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] is 0.5.
    assert np.isclose(beats.overall_stats.median_interval, 0.5) # Check it IS 0.5

def test_regular_section_detection_insufficient():
    """Test when the longest regular section is shorter than min_measures."""
    # Regular section 0.7-2.2 (indices 2-5, length 4)
    timestamps = np.array([0.0, 0.2, 0.7, 1.2, 1.7, 2.2, 2.8, 3.5, 4.0]) 
    counts = np.array([0, 0, 1, 2, 3, 4, 0, 0, 0])
    beats_per_bar = 4
    min_measures = 2 # Requires 4*2 = 8 beats
    
    # Sequence finder finds 2-5 (len 4). This is < required 8.
    # Check only for the exception type
    with pytest.raises(BeatCalculationError):
        Beats.from_timestamps(timestamps, beats_per_bar, counts, min_measures=min_measures, tolerance_percent=10.0)

def test_regular_section_with_count_irregularities():
    """Test regular section finding when input counts are irregular (become 0)."""
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    counts_input = np.array([1, 2, 3, 4, 0, 1, 2, 3, 4]) # Count 0 at index 4
    beats_per_bar = 4
    min_measures = 1 # Requires 4 beats
    
    # Sequence finding:
    # Starts at 0 (count 1). OK until index 3 (count 4).
    # Index 4 (count 0) breaks sequence. Sequence 0-3 (len 4).
    # Starts again at index 5 (count 1). OK until index 8 (count 4). Sequence 5-8 (len 4).
    # Longest is length 4. Ambiguous, implementation takes the first one found? Let's assume so.
    # Fix: Pass arguments correctly using keywords
    beats = Beats.from_timestamps(timestamps=timestamps, beat_counts=counts_input, beats_per_bar=beats_per_bar, min_measures=min_measures)

    # Assuming the implementation picks the first longest sequence
    assert beats.start_regular_beat_idx == 0
    assert beats.end_regular_beat_idx == 4 # Exclusive
    assert beats.regular_stats.total_beats == 4
    
    # Verify the processed counts
    np.testing.assert_array_equal(beats.counts, [1, 2, 3, 4, 0, 1, 2, 3, 4])

def test_regular_sequence_starts_from_downbeat():
    """Verify the _find_longest_regular_sequence_static requires starting with count 1."""
    # Data where the longest *potentially* regular sequence doesn't start with 1
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    counts = np.array([2, 3, 4, 1, 2, 3, 4, 1]) # Starts with 2, then regular
    beats_per_bar = 4
    
    # Create beat_data array directly for static method testing
    beat_data = np.stack((timestamps, counts), axis=1)
    
    # Calculate median and tolerance
    intervals = np.diff(beat_data[:, 0])
    median_interval = np.median(intervals) # Should be 0.5
    tolerance = median_interval * 0.1 # 10% tolerance
    
    # Sequence finding should start at index 3 (the first '1')
    start, end, _ = Beats._find_longest_regular_sequence_static(
        beat_data, 10.0, beats_per_bar, median_interval, tolerance
    )
    
    assert start == 3
    # The sequence is 1, 2, 3, 4, 1 (indices 3, 4, 5, 6, 7). End index is 7.
    assert end == 7 # Inclusive end index (Corrected from 6)
    
    # Test case where *no* sequence starts with 1
    counts_no_downbeat = np.array([2, 3, 4, 2, 3, 4, 2, 3])
    beat_data_no_downbeat = np.stack((timestamps, counts_no_downbeat), axis=1)
    
    with pytest.raises(BeatCalculationError, match="No regular sequence starting with beat count '1' found"):
        Beats._find_longest_regular_sequence_static(
            beat_data_no_downbeat, 10.0, beats_per_bar, median_interval, tolerance
        )

# === Tests for _find_longest_regular_sequence_static ===
# Removed fixture simple_beat_list_factory

# Helper to create beat_data for static method tests
def create_beat_data(timestamps: list[float], counts: list[int]) -> np.ndarray:
     if not timestamps or not counts or len(timestamps) != len(counts):
         raise ValueError("Invalid input for create_beat_data")
     # Ensure float for timestamps, int for counts, then combine
     ts_arr = np.array(timestamps, dtype=float)
     cnt_arr = np.array(counts, dtype=int) # Static method expects counts
     return np.stack((ts_arr, cnt_arr), axis=1)

@pytest.fixture
def default_static_params():
    # Provides common median and tolerance for static tests
    median_interval = 0.5
    tolerance_interval = 0.05 # 10% of 0.5
    tolerance_percent = 10.0
    beats_per_bar = 4
    return beats_per_bar, median_interval, tolerance_interval, tolerance_percent

def test_find_longest_regular_sequence_correct_counts(default_static_params):
    """Test basic correct sequence detection."""
    beats_per_bar, median, tolerance, tol_pct = default_static_params
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    counts =     [1,   2,   3,   4,   1,   2,   3,   4]
    beat_data = create_beat_data(timestamps, counts)
    
    start, end, _ = Beats._find_longest_regular_sequence_static(
        beat_data, tol_pct, beats_per_bar, median, tolerance
    )
    assert start == 0
    assert end == 7

def test_find_longest_regular_sequence_incorrect_count_breaks(default_static_params):
    """Test that an incorrect count breaks the sequence."""
    beats_per_bar, median, tolerance, tol_pct = default_static_params
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    counts =     [1,   2,   3,   5,   1,   2,   3,   4] # Incorrect count 5 at index 3
    beat_data = create_beat_data(timestamps, counts)
    
    # Sequence 0-2 breaks (len 3). Sequence 4-7 starts later (len 4).
    # Should find the longest: 4-7
    start, end, _ = Beats._find_longest_regular_sequence_static(
        beat_data, tol_pct, beats_per_bar, median, tolerance
    )
    assert start == 4 # Corrected from 0
    assert end == 7 # Corrected from 2

def test_find_longest_regular_sequence_incorrect_wrap_breaks(default_static_params):
    """Test that incorrect wrap (e.g., 4 -> 2) breaks the sequence."""
    beats_per_bar, median, tolerance, tol_pct = default_static_params
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    counts =     [1,   2,   3,   4,   2,   3,   4,   1] # Incorrect count 2 after 4 at index 4
    beat_data = create_beat_data(timestamps, counts)
    
    # Sequence should be 0-3
    start, end, _ = Beats._find_longest_regular_sequence_static(
        beat_data, tol_pct, beats_per_bar, median, tolerance
    )
    assert start == 0
    assert end == 3

def test_find_longest_regular_sequence_zero_count_breaks(default_static_params):
    """Test that a zero count breaks the sequence."""
    beats_per_bar, median, tolerance, tol_pct = default_static_params
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    counts =     [1,   2,   3,   0,   1,   2,   3,   4] # Zero count at index 3
    beat_data = create_beat_data(timestamps, counts)
    
    # Sequence 0-2 is broken by count 0. 
    # Sequence 4-7 starts later. Sequence 0-2 has length 3. Sequence 4-7 has length 4.
    # Should find 4-7.
    start, end, _ = Beats._find_longest_regular_sequence_static(
        beat_data, tol_pct, beats_per_bar, median, tolerance
    )
    assert start == 4
    assert end == 7

def test_find_longest_regular_sequence_non_downbeat_start_ignored(default_static_params):
    """Test that sequences must start with count 1."""
    beats_per_bar, median, tolerance, tol_pct = default_static_params
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    counts =     [2,   3,   4,   1,   2,   3,   4,   1] # Starts with 2
    beat_data = create_beat_data(timestamps, counts)
    
    # First valid start is index 3. Sequence 3-7. Length 5.
    # Should find 3-7.
    start, end, _ = Beats._find_longest_regular_sequence_static(
        beat_data, tol_pct, beats_per_bar, median, tolerance
    )
    assert start == 3
    assert end == 7 # Inclusive index (Corrected from 6)

def test_find_longest_regular_sequence_multiple_candidates(default_static_params):
    """Test selection when multiple sequences of the same max length exist."""
    beats_per_bar, median, tolerance, tol_pct = default_static_params
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    counts =     [1,   2,   3,   4,   0,   0,   1,   2,   3,   4,   0,   0 ] 
    # Sequence 1: 0-3 (len 4)
    # Sequence 2: 6-9 (len 4)
    beat_data = create_beat_data(timestamps, counts)

    # Implementation should return the *first* longest sequence found
    start, end, _ = Beats._find_longest_regular_sequence_static(
        beat_data, tol_pct, beats_per_bar, median, tolerance
    )
    assert start == 0
    assert end == 3

def test_find_longest_regular_sequence_no_valid_sequence(default_static_params):
    """Test case where no regular sequence can be found."""
    beats_per_bar, median, tolerance, tol_pct = default_static_params
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0]
    counts =     [2,   3,   0,   2,   3] # No count 1
    beat_data = create_beat_data(timestamps, counts)
    
    with pytest.raises(BeatCalculationError, match="No regular sequence starting with beat count '1' found"):
        Beats._find_longest_regular_sequence_static(
            beat_data, tol_pct, beats_per_bar, median, tolerance
        )

def test_find_longest_regular_sequence_only_downbeats(default_static_params):
    """Test sequence of only downbeats (should be length 1)."""
    beats_per_bar, median, tolerance, tol_pct = default_static_params
    timestamps = [0.0, 0.5, 1.0, 1.5]
    counts =     [1,   1,   1,   1] 
    beat_data = create_beat_data(timestamps, counts)
    
    # First sequence starts at 0, but breaks at index 1 (expected 2, got 1). Length 1.
    # Second sequence starts at 1, breaks at index 2. Length 1.
    # Third sequence starts at 2, breaks at index 3. Length 1.
    # Fourth sequence starts at 3. Length 1.
    # First longest is 0-0.
    start, end, _ = Beats._find_longest_regular_sequence_static(
        beat_data, tol_pct, beats_per_bar, median, tolerance
    )
    assert start == 0
    assert end == 0