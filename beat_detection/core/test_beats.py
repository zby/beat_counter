"""
Tests for the Beats data structure and its core logic.
"""

import numpy as np
import pytest
from typing import Optional

from beat_detection.core.beats import Beats, BeatStatistics, BeatCalculationError, RawBeats
from conftest import assert_raises


# Helper function to create RawBeats for testing
def create_test_raw_beats(num_beats=20, interval=0.5, beats_per_bar=4) -> RawBeats:
    """Creates predictable RawBeats for input to Beats tests."""
    timestamps = np.arange(num_beats) * interval
    # Simple downbeats every 'beats_per_bar' beats
    beat_counts = np.array([(i % beats_per_bar) + 1 for i in range(num_beats)])
    clip_length = timestamps[-1] + interval  # Add one more interval to clip_length
    return RawBeats(timestamps=timestamps, beat_counts=beat_counts, clip_length=clip_length)


# Helper function to create a standard Beats object for testing using the new init
def create_test_beats(
    beats_per_bar: Optional[int] = 4,
    num_beats=20,
    interval=0.5,
    tolerance=10.0,
    min_measures=2,
) -> Beats:
    """Creates a predictable Beats object for logic tests using __init__."""
    # Ensure enough beats for the default min_measures if bpb is specified
    required = 0
    if beats_per_bar is not None:
        required = beats_per_bar * min_measures
    else:
        # If bpb is inferred, we can't pre-calculate required beats easily.
        # Assume default 4 for pre-calc check, constructor will raise if inference fails.
        required = 4 * min_measures

    if num_beats < required:
        num_beats = required # Adjust num_beats if the defaults don't meet the minimum requirement

    # Create RawBeats using the specified or default beats_per_bar for count generation
    bpb_for_raw = beats_per_bar if beats_per_bar is not None else 4 # Use 4 if inferring
    raw_beats_input = create_test_raw_beats(
        num_beats=num_beats, interval=interval, beats_per_bar=bpb_for_raw
    )

    # Pass bpb=None if we want to test inference
    bpb_arg = beats_per_bar # Pass the original argument (might be None)

    return Beats(
        raw_beats=raw_beats_input,
        beats_per_bar=bpb_arg,
        tolerance_percent=tolerance,
        min_measures=min_measures,
    )


# Test Cases

def test_beat_creation_infer_bpb():
    """Test Beats creation inferring beats_per_bar."""
    # Create RawBeats explicitly with bpb=3 for count generation
    raw_beats = create_test_raw_beats(beats_per_bar=3, num_beats=12, interval=0.6)
    # Create Beats object, letting it infer bpb (should be 3)
    beats = Beats(raw_beats=raw_beats, min_measures=2)

    assert isinstance(beats, Beats)
    assert beats.beats_per_bar == 3 # Check inferred value
    assert beats.overall_stats.total_beats == 12
    assert np.isclose(beats.overall_stats.median_interval, 0.6)
    assert np.isclose(beats.regular_stats.median_interval, 0.6)
    assert len(beats.irregular_beat_indices) == 0


def test_beat_creation_override_bpb():
    """Test Beats creation providing an explicit beats_per_bar override."""
    # RawBeats generated with counts 1, 2, 3, 4, ...
    raw_beats = create_test_raw_beats(beats_per_bar=4, num_beats=12, interval=0.5)
    # Override bpb to 3 during Beats creation
    # We need to use min_measures=1 since we only have 12 beats and with bpb=3 we need at least 3 beats per measure
    beats = Beats(raw_beats=raw_beats, beats_per_bar=3, min_measures=1)

    assert isinstance(beats, Beats)
    assert beats.beats_per_bar == 3 # Check override value
    assert beats.overall_stats.total_beats == 12
    assert np.isclose(beats.overall_stats.median_interval, 0.5)

    # The processed counts should be [1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0]
    # because input count 4 is invalid for bpb=3
    expected_processed_counts = [1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0]
    np.testing.assert_array_equal(beats.counts, expected_processed_counts)

    # Check irregularity based on count=0
    assert len(beats.irregular_beat_indices) == 3
    np.testing.assert_array_equal(beats.irregular_beat_indices, [3, 7, 11])
    expected_irregularity = (3 / 12) * 100
    assert np.isclose(beats.overall_stats.irregularity_percent, expected_irregularity)

    # Check regular section (should be 0-2 due to irregularity at index 3)
    assert beats.start_regular_beat_idx == 0
    assert beats.end_regular_beat_idx == 3 # Longest regular sequence is first 3 beats
    assert beats.regular_stats.total_beats == 3
    # Note: Regular stats might be less meaningful with only 3 beats


def test_beat_creation_properties():
    """Test basic Beats object properties using the new helper."""
    beats = create_test_beats(beats_per_bar=4, num_beats=20, interval=0.5, min_measures=2)

    assert isinstance(beats, Beats)
    assert beats.beats_per_bar == 4
    assert beats.overall_stats.total_beats == 20
    assert isinstance(beats.overall_stats, BeatStatistics)
    assert isinstance(beats.regular_stats, BeatStatistics)
    assert np.isclose(beats.overall_stats.median_interval, 0.5)
    assert beats.start_regular_beat_idx == 0
    assert beats.end_regular_beat_idx == 20
    assert beats.regular_stats.total_beats == 20
    assert np.isclose(beats.regular_stats.median_interval, 0.5)
    assert np.isclose(beats.clip_length, 10.0)  # 20 beats * 0.5s interval + 0.5s

    # Test properties
    assert isinstance(beats.beat_data, np.ndarray)
    assert beats.beat_data.shape == (20, 2)
    assert len(beats.timestamps) == 20
    assert len(beats.counts) == 20
    assert np.all(beats.timestamps == beats.beat_data[:, 0])
    assert np.all(beats.counts == beats.beat_data[:, 1])

    expected_downbeat_indices = [0, 4, 8, 12, 16]
    actual_downbeat_indices = np.where(beats.counts == 1)[0].tolist()
    assert actual_downbeat_indices == expected_downbeat_indices
    assert len(beats.irregular_beat_indices) == 0
    assert isinstance(beats.irregular_beat_indices, np.ndarray)


def test_beat_creation_invalid_bpb_override():
    """Test error raising for invalid explicit beats_per_bar."""
    raw_beats = create_test_raw_beats(num_beats=8)
    with pytest.raises(
        BeatCalculationError, match="Invalid beats_per_bar provided: 1"
    ):
        Beats(raw_beats=raw_beats, beats_per_bar=1)
    with pytest.raises(
        BeatCalculationError, match="Invalid beats_per_bar provided: 0"
    ):
        Beats(raw_beats=raw_beats, beats_per_bar=0)
    with pytest.raises(
        BeatCalculationError, match="Invalid beats_per_bar provided: -2"
    ):
        Beats(raw_beats=raw_beats, beats_per_bar=-2)


def test_beat_creation_inference_fails_empty():
    """Test error raising when inferring bpb from empty RawBeats."""
    raw_beats = RawBeats(np.array([]), np.array([]), clip_length=3.0)
    with pytest.raises(BeatCalculationError, match="Cannot infer beats_per_bar: No beats provided"):
        Beats(raw_beats=raw_beats, beats_per_bar=None)


def test_beat_creation_inference_fails_all_zero_counts():
    """Test error raising when inferring bpb from only zero counts."""
    raw_beats = RawBeats(np.array([0.5, 1.0, 1.5]), np.array([0, 0, 0]), clip_length=2.0)
    with assert_raises(
        BeatCalculationError,
        match=r"Cannot infer beats_per_bar: No valid \(non-zero\) beat counts found",
    ):
        Beats(raw_beats=raw_beats, beats_per_bar=None)


def test_beat_creation_inference_fails_invalid_max():
    """Test error raising when inferred bpb is <= 1."""
    # Max count is 1
    raw_beats = RawBeats(np.array([0.5, 1.0, 1.5]), np.array([1, 1, 1]), clip_length=2.0)
    with assert_raises(
        BeatCalculationError,
        match=r"Inferred beats_per_bar \(1\) is invalid. Must be > 1."
    ):
        Beats(raw_beats=raw_beats, beats_per_bar=None)
    # Max count is 0 (after filtering)
    raw_beats_zero = RawBeats(np.array([0.5, 1.0, 1.5]), np.array([0, 0, -1]), clip_length=2.0)
    with assert_raises(
        BeatCalculationError,
        match=r"Cannot infer beats_per_bar: No valid \(non-zero\) beat counts found",
    ):
        Beats(raw_beats=raw_beats_zero, beats_per_bar=None)


def test_beat_counting_regular():
    """Test beat counting for a regular sequence."""
    beats = create_test_beats(
        beats_per_bar=4, num_beats=8, interval=0.5, min_measures=2
    )
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
    timestamps = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0]
    ) # Irregular interval 2.5 -> 3.5 (1.0s vs median 0.5s)
    beats_per_bar = 4
    counts_input = np.array(
        [1, 2, 3, 4, 1, 2, 1, 2]
    ) # Counts are sequentially valid initially
    raw_beats = RawBeats(timestamps=timestamps, beat_counts=counts_input, clip_length=4.5)

    # Expected regular sequence: 0-5 (len 6). min_measures=1 requires 4 beats.
    beats = Beats(raw_beats=raw_beats, beats_per_bar=beats_per_bar, tolerance_percent=10.0, min_measures=1)

    assert beats.overall_stats.total_beats == 8
    # Regular section should be 0-5 (exclusive end 6)
    assert beats.start_regular_beat_idx == 0
    assert beats.end_regular_beat_idx == 6

    # Check processed counts in beat_data
    expected_processed_counts = [1, 2, 3, 4, 1, 2, 1, 2] # Counts are valid w.r.t bpb=4
    np.testing.assert_array_equal(beats.counts, expected_processed_counts)

    # Beat 6 (ts 3.5) follows an irregular interval and breaks the sequence.
    # Beat 7 (ts 4.0) also follows an irregular interval.
    # get_info_at_time should return 0 for beats outside this regular section.

    # Check beat 6 (index 6)
    count_irr_interval, _, idx_irr_interval = beats.get_info_at_time(3.5 + 0.01)
    assert count_irr_interval == 0 # Outside regular section
    assert idx_irr_interval == 6

    # Check overall irregularity percentage (based on processed counts == 0) - should be 0
    assert beats.overall_stats.irregularity_percent == 0.0
    # Check irregular_beat_indices property (based on processed counts == 0) - should be empty
    assert len(beats.irregular_beat_indices) == 0

    # Scenario 2: Irregular input count
    timestamps_c = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    beats_per_bar_c = 4
    beat_counts_irr_c = np.array([1, 2, 3, 4, 5, 1, 2]) # Invalid count 5 at index 4
    raw_beats_c = RawBeats(timestamps=timestamps_c, beat_counts=beat_counts_irr_c, clip_length=3.5)

    beats_c = Beats(
        raw_beats=raw_beats_c,
        beats_per_bar=beats_per_bar_c,
        tolerance_percent=10.0,
        min_measures=1,
    )

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
    # Default create_test_beats uses min_measures=2, bpb=4, so needs >= 8 beats
    # Scenario 1: Not enough beats overall for min_measures
    raw_beats_short = create_test_raw_beats(num_beats=7, beats_per_bar=4)
    with assert_raises(BeatCalculationError, match=r"Longest regular sequence found.*shorter than required"):
        Beats(raw_beats=raw_beats_short, beats_per_bar=4, min_measures=2)

    # Scenario 2: Enough beats overall, but longest regular sequence is too short
    timestamps = np.arange(10) * 0.5
    counts = np.array([1, 2, 3, 4, 1, 2, 0, 1, 2, 3]) # Regular sequence 0-5 (len 6)
    raw_beats_irr = RawBeats(timestamps=timestamps, beat_counts=counts, clip_length=5.5)
    with pytest.raises(
        BeatCalculationError,
        match=r"Longest regular sequence found \(6 beats from index 0 to 5\) is shorter than required \(8 beats = 2 measures of 4/X time\)\.$"
    ):
        Beats(raw_beats=raw_beats_irr, beats_per_bar=4, min_measures=2)

    # Scenario 3: 0 or 1 beat (should still check min_measures)
    raw_beats_0 = RawBeats(np.array([]), np.array([]), clip_length=3.0)
    with pytest.raises(BeatCalculationError, match=r"Insufficient number of beats \(0\) for analysis with beats_per_bar 4"):
        Beats(raw_beats=raw_beats_0, beats_per_bar=4, min_measures=1) # requires 4

    raw_beats_1 = RawBeats(np.array([0.5]), np.array([1]), clip_length=1.0)
    with pytest.raises(BeatCalculationError, match=r"Insufficient number of beats \(1\) for analysis with beats_per_bar 4"):
        Beats(raw_beats=raw_beats_1, beats_per_bar=4, min_measures=1) # requires 4

    # Scenario 4: Valid case with exactly minimum required beats
    raw_beats_min = create_test_raw_beats(num_beats=8, beats_per_bar=4)
    beats_min = Beats(raw_beats=raw_beats_min, beats_per_bar=4, min_measures=2)
    assert beats_min.regular_stats.total_beats == 8

    # Scenario 5: Strict increasing timestamp error (now caught by RawBeats)
    with pytest.raises(ValueError, match="Timestamps must be strictly increasing"):
        RawBeats(np.array([0.0, 0.5, 0.5, 1.0]), np.array([1, 2, 3, 4]), clip_length=1.5)

    # Scenario 6: Invalid tolerance
    raw_beats_valid = create_test_raw_beats(num_beats=8)
    with pytest.raises(BeatCalculationError, match="Invalid tolerance_percent"):
        Beats(raw_beats=raw_beats_valid, tolerance_percent=-5.0)

    # Scenario 7: Invalid beats_per_bar (tested separately above)
    # Scenario 8: Median interval <= 0
    with pytest.raises(ValueError, match="Timestamps must be strictly increasing"):
        RawBeats(np.array([0.0, 0.0, 0.0, 0.0]), np.array([1, 2, 3, 4]), clip_length=1.0)

    # Scenario 9: Mismatched lengths (now caught by RawBeats)
    with pytest.raises(ValueError, match=r"Timestamp count \(4\) does not match beat count \(3\)"):
        RawBeats(np.array([0.0, 0.5, 1.0, 1.5]), np.array([1, 2, 3]), clip_length=2.0)

    # Scenario 10: Time past last beat tolerance
    # Create beats with 8 beats at 0.5s intervals (last beat at 3.5s)
    raw_beats_tolerance = create_test_raw_beats(num_beats=8, interval=0.5)
    beats_tolerance = Beats(raw_beats=raw_beats_tolerance, beats_per_bar=4, min_measures=2, tolerance_percent=10.0)
    
    # Calculate tolerance interval (0.5s * 10% = 0.05s)
    tolerance_interval = 0.5 * (10.0 / 100.0)
    last_beat_time = 3.5  # Last beat at 3.5s
    
    # Test just before tolerance (should return last beat)
    time_before_tolerance = last_beat_time + tolerance_interval - 0.01
    count_before, time_since_before, idx_before = beats_tolerance.get_info_at_time(time_before_tolerance)
    assert count_before == 4  # Last beat count
    assert np.isclose(time_since_before, tolerance_interval - 0.01)
    assert idx_before == 7  # Last beat index
    
    # Test just past tolerance (should return 0)
    time_after_tolerance = last_beat_time + tolerance_interval + 0.01
    count_after, time_since_after, idx_after = beats_tolerance.get_info_at_time(time_after_tolerance)
    assert count_after == 0  # Should return 0
    assert np.isclose(time_since_after, 0.0)  # Should return 0.0 for time since
    assert idx_after == -1  # Should return -1 for index


def test_regular_section_detection_full():
    """Regular section covers the entire input."""
    beats = create_test_beats(num_beats=16, beats_per_bar=4)
    assert beats.start_regular_beat_idx == 0
    assert beats.end_regular_beat_idx == 16
    assert beats.regular_stats.total_beats == 16


def test_regular_section_detection_intro_outro():
    """Regular section excludes irregular intro/outro beats based on intervals/counts."""
    timestamps = np.array(
        [0.1, 0.7, 1.2, 1.7, 2.2, 2.7, 3.2, 3.7, 4.2, 4.9] # Intro 0.1, Outro 4.9 (intervals 0.6, 0.7)
    ) # Median interval 0.5
    counts = np.array([0, 1, 2, 3, 4, 1, 2, 3, 4, 0])
    raw_beats = RawBeats(timestamps=timestamps, beat_counts=counts, clip_length=5.5)
    # Needs 5 measures of 4 beats = 20 beats! Helper adjusts num_beats
    # Let's use min_measures=1 (requires 4 beats)
    beats = Beats(raw_beats=raw_beats, beats_per_bar=4, min_measures=1)

    # Regular section expected: index 1 to 8 (exclusive end 9)
    assert beats.start_regular_beat_idx == 1
    assert beats.end_regular_beat_idx == 9
    assert beats.regular_stats.total_beats == 8 # Beats from index 1 to 8
    np.testing.assert_allclose(beats.regular_stats.median_interval, 0.5, atol=1e-6)


def test_regular_section_detection_insufficient():
    """Test that BeatCalculationError is raised if no sequence meets min_measures."""
    timestamps = np.arange(7) * 0.5 # 7 beats
    counts = np.array([1, 2, 3, 4, 1, 2, 3])
    raw_beats = RawBeats(timestamps=timestamps, beat_counts=counts, clip_length=4.0)

    # Requires min_measures=2 * beats_per_bar=4 = 8 beats for regular section
    with pytest.raises(
        BeatCalculationError,
        match=r"Longest regular sequence found \(7 beats from index 0 to 6\) is shorter than required \(8 beats = 2 measures of 4/X time\)\.$"
    ):
        Beats(raw_beats=raw_beats, beats_per_bar=4, min_measures=2)


def test_regular_section_with_count_irregularities():
    """Test regular section finding when counts are invalid within a potential sequence."""
    timestamps = np.arange(12) * 0.5
    # Counts have invalid values (0, 5) breaking regularity
    counts = np.array([1, 2, 3, 4, 1, 0, 3, 4, 1, 2, 5, 4])
    raw_beats = RawBeats(timestamps=timestamps, beat_counts=counts, clip_length=6.5)

    # Longest sequences: 0-4 (len 5), 8-9 (len 2). Need min 1 measure = 4 beats.
    beats = Beats(raw_beats=raw_beats, beats_per_bar=4, min_measures=1)

    # Expect the longest valid sequence: indices 0 to 4 (exclusive end 5)
    assert beats.start_regular_beat_idx == 0
    assert beats.end_regular_beat_idx == 5
    assert beats.regular_stats.total_beats == 5

    # Test with different min_measures
    # Requires min 2 measures = 8 beats. Should fail.
    with pytest.raises(
        BeatCalculationError,
        match=r"Longest regular sequence found \(5 beats from index 0 to 4\) is shorter than required \(8 beats = 2 measures of 4/X time\)\.$",
    ):
        Beats(raw_beats=raw_beats, beats_per_bar=4, min_measures=2)


def test_regular_sequence_starts_from_downbeat():
    """Test that the identified regular sequence must start with beat count 1."""
    timestamps = np.arange(10) * 0.5
    # Sequence starts with 2, then becomes regular from index 1 (count 1)
    counts = np.array([2, 1, 2, 3, 4, 1, 2, 3, 4, 1])
    raw_beats = RawBeats(timestamps=timestamps, beat_counts=counts, clip_length=5.5)
    # Requires 1 measure = 4 beats
    beats = Beats(raw_beats=raw_beats, beats_per_bar=4, min_measures=1)

    # Expected regular sequence: index 1 to 9 (exclusive end 10)
    assert beats.start_regular_beat_idx == 1
    assert beats.end_regular_beat_idx == 10
    assert beats.regular_stats.total_beats == 9

    # Scenario: No downbeat found at all within a potential segment
    counts_no_db = np.array([2, 3, 4, 2, 3, 4, 2, 3, 4]) # Regular intervals, counts > 1
    raw_beats_no_db = RawBeats(np.arange(9) * 0.5, counts_no_db, clip_length=5.0)
    with pytest.raises(BeatCalculationError, match="No regular sequence starting with beat count '1'"):
        Beats(raw_beats=raw_beats_no_db, beats_per_bar=4, min_measures=1)

    # Scenario: Downbeat occurs late, after max_start_time
    timestamps_late_db = np.arange(40) * 0.5 # Starts at 0.0, extends past 30s
    counts_late_db = np.roll(np.tile([1, 2, 3, 4], 10), 2) # First '1' will be after 30s
    first_downbeat_index = np.where(counts_late_db == 1)[0][0] # Should be index 2
    first_downbeat_time = timestamps_late_db[first_downbeat_index]
    assert first_downbeat_time < 30.0 # Should be 1.0s

    # Ensure first beat > 30s starts with 1
    late_start_idx = 61 # time = 30.5
    timestamps_late = np.arange(late_start_idx, late_start_idx + 20) * 0.5
    counts_late = np.array([(i % 4) + 1 for i in range(20)])
    counts_late[0] = 1 # Ensure it starts with 1
    assert timestamps_late[0] > 30.0
    raw_beats_late = RawBeats(timestamps_late, counts_late, clip_length=timestamps_late[-1] + 0.5)
    with pytest.raises(BeatCalculationError, match="No regular sequence starting with beat count '1' within the first 30.0 seconds"):
         Beats(raw_beats=raw_beats_late, beats_per_bar=4, min_measures=1, max_start_time=30.0)

    # Test that max_start_time allows a later start if changed
    beats_late_allowed = Beats(raw_beats=raw_beats_late, beats_per_bar=4, min_measures=1, max_start_time=31.0)
    assert beats_late_allowed.start_regular_beat_idx == 0 # Starts at index 0 of this late RawBeats
    assert beats_late_allowed.end_regular_beat_idx == 20


# --- Static Method Tests (moved _find_longest_regular_sequence_static to Beats) ---

def create_beat_data(timestamps: list[float], counts: list[int]) -> np.ndarray:
    """Helper to create beat_data array for static method tests."""
    return np.stack((np.array(timestamps), np.array(counts)), axis=1)

@pytest.fixture
def default_static_params():
    # Provides common median and tolerance for static tests
    median_interval = 0.5
    tolerance_interval = 0.05  # Fixed tolerance interval of 0.05 seconds
    beats_per_bar = 4
    max_start_time = 30.0 # Default max start time
    return {
        "median_interval": median_interval,
        "tolerance_interval": tolerance_interval,
        "beats_per_bar": beats_per_bar,
        "max_start_time": max_start_time,
    }

def test_find_longest_regular_sequence_correct_counts(default_static_params):
    beat_data = create_beat_data(
        [0.0, 0.5, 1.0, 1.5, 2.0], [1, 2, 3, 4, 1]
    )
    start, end, _ = Beats._find_longest_regular_sequence_static(
        beat_data, **default_static_params
    )
    assert start == 0
    assert end == 4

def test_find_longest_regular_sequence_incorrect_count_breaks(default_static_params):
    # Incorrect count 5 at index 2
    beat_data = create_beat_data(
        [0.0, 0.5, 1.0, 1.5, 2.0], [1, 2, 5, 4, 1]
    )
    start, end, _ = Beats._find_longest_regular_sequence_static(
        beat_data, **default_static_params
    )
    # Sequence 0-1 (len 2)
    assert start == 0
    assert end == 1

def test_find_longest_regular_sequence_incorrect_wrap_breaks(default_static_params):
    # Incorrect wrap 4 -> 2 at index 4
    beat_data = create_beat_data(
        [0.0, 0.5, 1.0, 1.5, 2.0], [1, 2, 3, 4, 2]
    )
    start, end, _ = Beats._find_longest_regular_sequence_static(
        beat_data, **default_static_params
    )
    # Sequence 0-3 (len 4)
    assert start == 0
    assert end == 3

def test_find_longest_regular_sequence_zero_count_breaks(default_static_params):
    # Zero count at index 2
    beat_data = create_beat_data(
        [0.0, 0.5, 1.0, 1.5, 2.0], [1, 2, 0, 4, 1]
    )
    start, end, _ = Beats._find_longest_regular_sequence_static(
        beat_data, **default_static_params
    )
    # Sequence 0-1 (len 2)
    assert start == 0
    assert end == 1

def test_find_longest_regular_sequence_non_downbeat_start_ignored(
    default_static_params,
):
    # Starts with 2, then regular 1,2,3,4 from index 1
    beat_data = create_beat_data(
        [0.0, 0.5, 1.0, 1.5, 2.0], [2, 1, 2, 3, 4]
    )
    start, end, _ = Beats._find_longest_regular_sequence_static(
        beat_data, **default_static_params
    )
    # Should pick sequence 1-4 (len 4)
    assert start == 1
    assert end == 4

def test_find_longest_regular_sequence_multiple_candidates(default_static_params):
    # Two sequences: 0-1 (len 2) and 3-6 (len 4). Should pick the longer one.
    beat_data = create_beat_data(
        [0.0, 0.5, 1.1, 1.6, 2.1, 2.6, 3.1], # Interval irregularity at idx 2
        [1,   2,   1,   1,   2,   3,   4]
    )
    start, end, _ = Beats._find_longest_regular_sequence_static(
        beat_data, **default_static_params
    )
    assert start == 3
    assert end == 6

def test_find_longest_regular_sequence_no_valid_sequence(default_static_params):
    # No sequence starts with 1
    beat_data = create_beat_data([0.0, 0.5, 1.0], [2, 3, 4])
    with pytest.raises(BeatCalculationError, match="No regular sequence starting with beat count '1'"):
        Beats._find_longest_regular_sequence_static(
            beat_data, **default_static_params
        )

def test_find_longest_regular_sequence_only_downbeats(default_static_params):
    # Sequence of only 1s
    beat_data = create_beat_data([0.0, 0.5, 1.0], [1, 1, 1])
    start, end, _ = Beats._find_longest_regular_sequence_static(
        beat_data, **default_static_params
    )
    # Longest sequence is just the first beat (len 1)
    assert start == 0
    assert end == 0

def test_find_longest_regular_sequence_max_start_time(default_static_params):
    params = default_static_params.copy()
    # Sequence 1: 0-4 (len 5), starts at 0.0s
    # Sequence 2: 6-10 (len 5), starts at 3.0s
    beat_data = create_beat_data(
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.6, 3.0, 3.5, 4.0, 4.5, 5.0],
        [1,   2,   3,   4,   1,   0,   1,   2,   3,   4,   1]
    )

    # Default max_start_time=30.0 allows both, should pick first one (indices 0-4)
    start, end, _ = Beats._find_longest_regular_sequence_static(beat_data, **params)
    assert start == 0
    assert end == 4

    # Limit max_start_time to 2.5s - only first sequence qualifies, still picked
    params["max_start_time"] = 2.5
    start, end, _ = Beats._find_longest_regular_sequence_static(beat_data, **params)
    assert start == 0
    assert end == 4

    # Limit max_start_time to 0.1s - first sequence (start=0.0) is valid, but second (start=3.0) isn't.
    # Should still pick the first one.
    # Actually, the loop finds the *longest*. If the first one starting at 0.0 fails later,
    # it might reset and find the one starting at 3.0. Let's re-verify.
    # The logic finds the *longest* sequence that *starts* within max_start_time.
    # Sequence 0-4 (len 5) starts at 0.0 <= 0.1 -> Candidate
    # Sequence 6-10 (len 5) starts at 3.0 > 0.1 -> Not a candidate
    # So, it should still pick 0-4.
    params["max_start_time"] = 0.1
    start, end, _ = Beats._find_longest_regular_sequence_static(beat_data, **params)
    assert start == 0
    assert end == 4

    # Modify data so first sequence is shorter
    # Seq 1: 0-2 (len 3), start 0.0s
    # Seq 2: 4-8 (len 5), start 2.0s
    beat_data_short_first = create_beat_data(
        [0.0, 0.5, 1.0, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0],
        [1,   2,   3,   0,   1,   2,   3,   4,   1]
    )

    # Default max_start_time=30.0: Allows both, picks longer (4-8)
    params["max_start_time"] = 30.0
    start, end, _ = Beats._find_longest_regular_sequence_static(beat_data_short_first, **params)
    assert start == 4
    assert end == 8

    # Limit max_start_time to 1.0s: Allows first (start=0.0), rejects second (start=2.0). Picks first (0-2).
    params["max_start_time"] = 1.0
    start, end, _ = Beats._find_longest_regular_sequence_static(beat_data_short_first, **params)
    assert start == 0
    assert end == 2

    # Limit max_start_time to 0.1s: Allows first (start=0.0), rejects second. Picks first (0-2).
    params["max_start_time"] = 0.1
    start, end, _ = Beats._find_longest_regular_sequence_static(beat_data_short_first, **params)
    assert start == 0
    assert end == 2

    # Limit max_start_time to 0.0s: Allows first (start=0.0), rejects second. Picks first (0-2).
    params["max_start_time"] = 0.0
    start, end, _ = Beats._find_longest_regular_sequence_static(beat_data_short_first, **params)
    assert start == 0
    assert end == 2

    # Limit max_start_time to negative: No sequence can start <= -0.1. Should raise error.
    params["max_start_time"] = -0.1
    with pytest.raises(BeatCalculationError, match="No regular sequence starting with beat count '1' within the first -0.1 seconds"):
        Beats._find_longest_regular_sequence_static(beat_data_short_first, **params)

# Removed tests for _find_longest_regular_sequence preferring shorter sequence < 30s,
# as that logic is now implicitly handled by the single pass finding the first longest valid one.

def test_beat_creation_invalid_clip_length():
    """Test error raising for invalid clip_length in RawBeats."""
    # Create timestamps that exceed clip_length
    timestamps = np.array([0.5, 1.0, 1.5, 2.0])
    counts = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError, match="Last timestamp.*exceeds clip_length"):
        RawBeats(timestamps=timestamps, beat_counts=counts, clip_length=1.5)

def test_beat_creation_negative_clip_length():
    """Test error raising for negative clip_length in RawBeats."""
    timestamps = np.array([0.5, 1.0])
    counts = np.array([1, 2])
    with pytest.raises(ValueError, match="clip_length must be a positive number"):
        RawBeats(timestamps=timestamps, beat_counts=counts, clip_length=-1.0)

def test_beat_creation_zero_clip_length():
    """Test error raising for zero clip_length in RawBeats."""
    timestamps = np.array([0.5, 1.0])
    counts = np.array([1, 2])
    with pytest.raises(ValueError, match="clip_length must be a positive number"):
        RawBeats(timestamps=timestamps, beat_counts=counts, clip_length=0.0)

def test_beat_creation_empty_with_clip_length():
    """Test creating Beats with empty RawBeats but valid clip_length."""
    raw_beats = RawBeats(np.array([]), np.array([]), clip_length=3.0)
    with pytest.raises(BeatCalculationError, match="Cannot infer beats_per_bar: No beats provided"):
        Beats(raw_beats=raw_beats, beats_per_bar=None)

def test_beat_creation_single_beat_with_clip_length():
    """Test creating Beats with single beat RawBeats and valid clip_length."""
    raw_beats = RawBeats(np.array([0.5]), np.array([1]), clip_length=1.0)
    with pytest.raises(BeatCalculationError, match="Insufficient number of beats.*for analysis"):
        Beats(raw_beats=raw_beats, beats_per_bar=4, min_measures=1)

def test_beat_creation_clip_length_preserved():
    """Test that clip_length is preserved from RawBeats to Beats."""
    raw_beats = create_test_raw_beats(num_beats=8, interval=0.5)
    expected_clip_length = raw_beats.clip_length
    beats = Beats(raw_beats=raw_beats, beats_per_bar=4, min_measures=1)
    assert np.isclose(beats.clip_length, expected_clip_length)

def test_beat_to_dict_includes_clip_length():
    """Test that to_dict includes clip_length in serialization."""
    beats = create_test_beats(num_beats=8, interval=0.5)
    data = beats.to_dict()
    assert "clip_length" in data
    assert np.isclose(data["clip_length"], beats.clip_length)
    assert isinstance(data["clip_length"], float)
