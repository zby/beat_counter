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
    assert beats.overall_stats.total_beats == 16
    assert beats.tolerance_percent == 10.0 # Default tolerance from helper
    assert beats.min_consistent_measures == 3
    assert np.isclose(beats.overall_stats.median_interval, 0.5)
    assert np.isclose(beats.overall_stats.tempo_bpm, 120.0)
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
    # Pass args by keyword to avoid misinterpretation
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
    
    # Check that irregularity_percent in stats reflects both interval and count irregularities
    # In this case, we have 1 irregular interval out of 6 beats = 16.67%
    assert np.isclose(beats.overall_stats.irregularity_percent, 16.67, atol=0.01)

def test_irregular_count_beats():
    """Test identification of beats with irregular counts (exceeding meter)."""
    # Create timestamps where downbeats are further apart than the meter suggests
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]) # 7 beats
    downbeats = np.array([0, 5]) # Downbeats at index 0 and 5
    meter = 4 # But downbeats imply meter 5
    # Need 4 * 1 = 4 beats minimum. We have 7.
    # Pass args by keyword
    beats = Beats.from_timestamps(timestamps, downbeats, meter, tolerance_percent=10.0, min_consistent_measures=1)
    
    # Expected counts: 1, 2, 3, 4, 5(irregular), 1, 2
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
    # Pass args by keyword
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
    """Test edge cases in beat creation."""
    # Test minimum number of beats (now stricter due to regular section finding)
    # Meter=2, min_measures=1 => requires 2 beats.
    # The sequence finder requires at least one interval, so min 2 beats.
    timestamps_2 = np.array([0.5, 1.0])
    downbeats_2 = np.array([0])
    try:
        # Pass args by keyword
        beats_2 = Beats.from_timestamps(timestamps_2, downbeats_2, meter=2, tolerance_percent=10.0, min_consistent_measures=1)
        assert len(beats_2.beat_list) == 2
        assert beats_2.overall_stats.total_beats == 2
        assert beats_2.start_regular_beat_idx == 0 # The whole sequence is regular
        assert beats_2.end_regular_beat_idx == 2
        assert beats_2.regular_stats.total_beats == 2
        assert np.isclose(beats_2.regular_stats.median_interval, 0.5)
    except BeatCalculationError as e:
        pytest.fail(f"Two-beat case failed unexpectedly: {e}")
        
    # Test insufficient beats for regular section finding, even if overall beats is enough
    # Meter=4, min_measures=2 => requires 8 beats.
    timestamps_short_regular = np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]) # Irregular start and end
    downbeats_short_regular = np.array([1, 5]) # Downbeats inside potentially regular section
    # Intervals: 1.0(irr), 0.5, 0.5, 0.5, 0.5, 0.5, 0.5(reg), 1.0(irr). Median=0.5, Tol=0.05
    # Beat counts (m=4, d=[1,5]): 0, 1, 2, 3, 4, 1, 2, 3, 0
    # Regular sequence check (req 8 beats):
    # Start 1: [1,2,3,4,5,6,7]. Intervals: 0.5,0.5,0.5,0.5,0.5,0.5. Irreg=0%. Len=7. Need 8. Invalid.
    with pytest.raises(BeatCalculationError, match="Could not determine a stable regular section: Longest regular sequence found .* is shorter than required"):
         # Pass args by keyword
         Beats.from_timestamps(timestamps_short_regular, downbeats_short_regular, meter=4, tolerance_percent=10.0, min_consistent_measures=2)

    # Test with slightly varying intervals that SHOULD be considered irregular
    timestamps_all_irr = np.array([0.0, 0.6, 1.1, 1.7, 2.2, 2.8, 3.3, 3.9]) # Intervals: 0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6
    downbeats_all_irr = np.array([0, 4])
    meter_irr=4
    min_measures_irr=1 # Requires 4 beats
    tolerance_irr=10.0
    # Median interval is 0.6. Tolerance interval = 0.6 * 0.10 = 0.06.
    # Allowed range [0.54, 0.66].
    # Intervals: 0.6(R), 0.5(I), 0.6(R), 0.5(I), 0.6(R), 0.5(I), 0.6(R).
    # Beat counts: 1,2,3,4,1,2,3,4 (all regular)
    # Sequence check (req 4 beats, 10% irreg tolerance):
    # Seq 0..7: Len 8. Intervals=7. Irregular intervals at index 1, 3, 5 (relative to seq). 3 irregular.
    # Irregularity % = (3 / 7) * 100 = 42.8% > 10%. Invalid.
    # Seq 0..0: Len 1. Valid, but too short.
    # Seq 0..1: Len 2. 1 Int (0.6). Irreg 0%. Valid, too short.
    # Seq 0..2: Len 3. 2 Ints (0.6, 0.5). 1 Irreg (0.5). Irreg 50%. Invalid.
    # ... No sequence of length >= 4 will have <= 10% irregularity.
    # Test now expects failure
    with pytest.raises(BeatCalculationError, match="Could not determine a stable regular section: Longest regular sequence found .* is shorter than required"):
        Beats.from_timestamps(timestamps_all_irr, downbeats_all_irr, meter=meter_irr, tolerance_percent=tolerance_irr, min_consistent_measures=min_measures_irr)

    # Test case where only a short section in the middle is regular
    timestamps_mid_reg = np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.0]) # Irreg start/end
    downbeats_mid_reg = np.array([1, 5])
    meter_mid = 4
    min_measures_mid = 1 # Requires 4 beats
    # Intervals: 1.0(irr), 0.5, 0.5, 0.5, 0.5, 0.5, 1.0(irr), 0.5. Median=0.5, Tol=0.05
    # Beat Counts (m=4): 0, 1, 2, 3, 4, 1, 2, 0, 0
    # Need 4*1=4 beats.
    # Seq [1..6]: Len 6. Intervals: 0.5,0.5,0.5,0.5,0.5 (5 intervals). Irreg=0%. Valid.
    # Pass args by keyword
    beats_mid = Beats.from_timestamps(timestamps_mid_reg, downbeats_mid_reg, meter=meter_mid, tolerance_percent=10.0, min_consistent_measures=min_measures_mid)
    assert beats_mid.start_regular_beat_idx == 1
    assert beats_mid.end_regular_beat_idx == 7 # Exclusive index for beat 6
    assert beats_mid.regular_stats.total_beats == 6
    assert np.isclose(beats_mid.regular_stats.median_interval, 0.5)

    # Test case with only irregular counts breaking sequence
    timestamps_reg_interval = np.arange(12) * 0.5 # 12 beats, regular interval
    downbeats_bad_count = np.array([0, 5, 10]) # Downbeats imply meter 5
    meter = 4
    # Counts: 1,2,3,4, 0(5), 1,2,3,4, 0(5), 1,2
    # Need 4*1=4 beats.
    # Seq [0..3]: Len 4. Irreg=0%. Valid.
    # Seq [5..8]: Len 4. Irreg=0%. Valid.
    # Seq [10..11]: Len 2. Invalid.
    # Longest valid is length 4. Pick first one? Yes, current logic finds first longest.
    # Pass args by keyword
    beats_bad_count = Beats.from_timestamps(timestamps_reg_interval, downbeats_bad_count, meter=meter, tolerance_percent=10.0, min_consistent_measures=1)
    assert beats_bad_count.start_regular_beat_idx == 0
    assert beats_bad_count.end_regular_beat_idx == 4
    assert beats_bad_count.regular_stats.total_beats == 4

    # Test non-positive median interval
    # Use timestamps that guarantee median is 0
    bad_timestamps = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 1.0]) # Intervals: 0, 0, 0, 0, 0.5 -> Median 0
    bad_downbeats = np.array([0, 3]) 
    meter_bad = 4
    min_measures_bad = 1 # Req 4 beats. Have 6.
    with pytest.raises(BeatCalculationError, match="Median interval is 0.0000"):
        # Pass args by keyword
        Beats.from_timestamps(bad_timestamps, bad_downbeats, meter=meter_bad, tolerance_percent=10.0, min_consistent_measures=min_measures_bad)

    # Test non-positive median interval determined *within* sequence finder (overall median is ok)
    # The sequence finder now uses the overall median, so this specific case isn't testable separately.
    # The previous test already covers overall non-positive median.
    # Removing the duplicate test case based on timestamps_zero_median_overall.
    # timestamps_zero_median_overall = np.array([0.5, 0.5, 0.5, 1.0, 1.5, 2.0])
    # downbeats_zero_median_overall = np.array([0, 3])
    # with pytest.raises(BeatCalculationError, match="Median interval is 0.0000"):
    #     Beats.from_timestamps(timestamps_zero_median_overall, downbeats_zero_median_overall, 3, min_consistent_measures=1)

    # Test failure during regular section calculation (e.g., non-positive median in regular part)
    # This specific check (regular_median_interval <= 0) might be harder to trigger if the overall passed
    timestamps_bad_reg_median = np.array([0.0, 0.5, 1.0, 1.5, 1.5, 1.5, 2.0, 2.5, 3.0, 3.5])
    downbeats_bad_reg_median = np.array([0, 6])
    # Overall Intervals: 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5. Overall Median=0.5.
    # Beat Counts (m=4): 1,2,3,4, 0(5), 0(6), 1,2,3,4.
    # Need 4*1=4 beats. Tol=0.05.
    # Seq [0..3]: Len 4. Intervals 0.5,0.5,0.5. Irreg=0%. Valid.
    # Seq [6..9]: Len 4. Intervals 0.5,0.5,0.5. Irreg=0%. Valid.
    # Found regular sequence [0..3]. Regular Intervals = [0.5, 0.5, 0.5]. Median=0.5. OK.
    # Let's try to make the *regular* section have bad stats. Need the regular finder to pick it.
    timestamps_tricky = np.array([0.0, 1.0, 1.5, 2.0, 2.5, # Irregular start
                                 3.0, 3.0, 3.0, 3.0,       # Regular section with 0 intervals
                                 3.5, 4.0, 4.5, 5.5])      # Irregular end
    downbeats_tricky = np.array([1, 5, 10])
    meter=4
    # Overall Intervals: 1.0(i), 0.5, 0.5, 0.5, 0.5(r), 0.0, 0.0, 0.0, 0.5(r), 0.5(r), 0.5(r), 1.0(i). Median=0.5. Tol=0.05.
    # Beat Counts (m=4, d=[1,5,10]): 0, 1,2,3,4, 1,2,3,4, 0(5), 1,2,3, 0(4)
    # Required beats = 4 * 1 = 4.
    # Seq [1..4]: Len 4. Ints: 0.5,0.5,0.5. Irreg 0%. Valid.
    # Seq [5..8]: Len 4. Ints: 0.0,0.0,0.0. Irreg 100%. Invalid.
    # Seq [10..12]: Len 3. Invalid.
    # Regular section found: [1..4]. Regular intervals: [0.5, 0.5, 0.5]. Median=0.5. OK.
    # The `regular_median_interval <= 0` check seems hard to hit now. Removing specific test for it.

# === Remove Obsolete Tests ===

# def test_find_longest_regular_sequence():
#    ... removed ...

# def test_find_longest_regular_sequence_edge_cases():
#    ... removed ...

# def test_find_longest_regular_sequence_with_irregular_counts():
#    ... removed ...

# def test_find_longest_regular_sequence_with_threshold():
#    ... removed ...

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
    downbeats = np.array([2, 6]) # Downbeats within the regular section
    meter_io = 4
    min_measures_io = 2 # Requires 8 beats
    # Intervals: 1.0(i), 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5(r), 1.0(i), 0.5. Median=0.5. Tol=0.05.
    # Beat Counts: 0, 0, 1,2,3,4, 1,2,3,4, 0, 0
    # Check sequence finder:
    # Seq [2..9]: Len 8. Intervals 0.5 (8 times). Irreg=0%. Valid.
    # Pass args by keyword
    beats = Beats.from_timestamps(timestamps, downbeats, meter=meter_io, tolerance_percent=10.0, min_consistent_measures=min_measures_io)
    assert beats.start_regular_beat_idx == 2
    assert beats.end_regular_beat_idx == 10 # Exclusive index for beat 9
    assert beats.regular_stats.total_beats == 8
    assert np.isclose(beats.regular_stats.median_interval, 0.5)
    assert beats.regular_stats.irregularity_percent == 0.0

def test_regular_section_detection_insufficient():
    """Test failure when no sufficiently long regular section exists."""
    timestamps = np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, 5.0]) # Irregular gaps
    downbeats = np.array([1, 4])
    meter_insuf = 3
    min_measures_insuf = 2 # Requires 6 beats
    # Intervals: 1.0(i), 0.5, 0.5, 0.5(r), 1.0(i), 0.5, 1.0(i). Median=0.5. Tol=0.05
    # Beat Counts: 0, 1,2,3, 1,2,0, 0
    # Check sequence finder (req 6):
    # Seq [1..3]: Len 3. Invalid.
    # Seq [4..5]: Len 2. Invalid.
    with pytest.raises(BeatCalculationError, match="Could not determine a stable regular section: Longest regular sequence found .* is shorter than required"):
        # Pass args by keyword
        Beats.from_timestamps(timestamps, downbeats, meter=meter_insuf, tolerance_percent=10.0, min_consistent_measures=min_measures_insuf)

def test_regular_section_with_count_irregularities():
    """Test that beats with irregular counts break the regular sequence."""
    timestamps = np.arange(12) * 0.5 # Regular intervals
    downbeats = np.array([0, 5, 10]) # Causes count irregularity (meter 5 implied)
    meter = 4
    min_measures = 1 # Requires 4 beats
    # Beat Counts: 1,2,3,4, 0(5), 1,2,3,4, 0(5), 1,2
    # Check sequence finder (req 4):
    # Seq [0..3]: Len 4. Irreg=0%. Valid.
    # Seq [5..8]: Len 4. Irreg=0%. Valid.
    # Seq [10..11]: Len 2. Invalid.
    # Longest valid is length 4. Should pick the first one found: [0..3]
    # Pass args by keyword
    beats = Beats.from_timestamps(timestamps, downbeats, meter=meter, tolerance_percent=10.0, min_consistent_measures=min_measures)
    assert beats.start_regular_beat_idx == 0
    assert beats.end_regular_beat_idx == 4
    assert beats.regular_stats.total_beats == 4
    assert np.isclose(beats.regular_stats.median_interval, 0.5) 