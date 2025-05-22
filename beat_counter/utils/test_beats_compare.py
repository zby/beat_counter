"""
Unit tests for the beat_detection.utils.beats_compare module.
"""

import pytest
from beat_counter.utils.beats_compare import compare_beats_data, format_comparison_output

# Test data for compare_beats_data
TS_EMPTY = []
BC_EMPTY = []

TS1_BASIC = [1.0, 2.0, 3.0]
BC1_BASIC = [10, 20, 30]

TS2_BASIC_MATCH = [1.0, 2.0, 3.0]
BC2_BASIC_MATCH = [10, 20, 30]

TS2_BASIC_SLIGHT_DIFF_MATCH = [1.02, 1.98, 3.01] # All should match with default threshold
BC2_BASIC_CONTENT_MISMATCH = [10, 25, 30]

TS1_PROXIMITY = [1.0, 1.005, 2.0] # 1.0 and 1.005 are too close (0.005 < 0.01 default)
BC1_PROXIMITY = [1, 1, 1]
TS2_PROXIMITY_OK = [1.0, 1.015, 2.0] # 1.0 and 1.015 are ok (0.015 > 0.01 default)
BC2_PROXIMITY_OK = [1, 1, 1]


@pytest.mark.parametrize(
    "ts1, bc1, ts2, bc2, match_thresh, expected_prox_errors_len, expected_prox_details",
    [
        # No proximity errors (1.0, 2.0, 3.0 - diffs are 1.0, well above 0.05)
        (TS1_BASIC, BC1_BASIC, TS2_BASIC_MATCH, BC2_BASIC_MATCH, 0.05, 0, []),
        # Proximity error in ts1 (1.0, 1.005 -> diff 0.005 < match_thresh 0.05)
        # AND Proximity error in ts2 (1.0, 1.015 -> diff 0.015 < match_thresh 0.05)
        (TS1_PROXIMITY, BC1_PROXIMITY, TS2_PROXIMITY_OK, BC2_PROXIMITY_OK, 0.05, 2, # Changed from 1 to 2
         [
             {"file_id": "file1", "timestamp1": 1.0, "timestamp2": 1.005, "diff": pytest.approx(0.005, abs=1e-6)},
             {"file_id": "file2", "timestamp1": 1.0, "timestamp2": 1.015, "diff": pytest.approx(0.015, abs=1e-6)} # Added this expected error
         ]),
        # Proximity error in ts2 (swap inputs for ts1_proximity)
        # AND Proximity error in ts1 (using TS2_PROXIMITY_OK which has 1.0, 1.015 -> diff 0.015 < 0.05)
        (TS2_PROXIMITY_OK, BC2_PROXIMITY_OK, TS1_PROXIMITY, BC1_PROXIMITY, 0.05, 2, # Changed from 1 to 2
         [
             {"file_id": "file2", "timestamp1": 1.0, "timestamp2": 1.005, "diff": pytest.approx(0.005, abs=1e-6)},
             {"file_id": "file1", "timestamp1": 1.0, "timestamp2": 1.015, "diff": pytest.approx(0.015, abs=1e-6)} # Added this expected error
         ]),
        # Timestamps at 0.01 diff, match_thresh 0.05 -> error (0.01 < 0.05)
        ([1.0, 1.01], [1], [2.0, 2.02], [1], 0.05, 2, # Errors in both file1 and file2
         [
             {"file_id": "file1", "timestamp1": 1.0, "timestamp2": 1.01, "diff": pytest.approx(0.01, abs=1e-6)},
             {"file_id": "file2", "timestamp1": 2.0, "timestamp2": 2.02, "diff": pytest.approx(0.02, abs=1e-6)}
         ]),
        # Timestamps at 0.01 diff, match_thresh 0.005 -> no error (0.01 > 0.005)
        ([1.0, 1.01], [1], [2.0, 2.02], [1], 0.005, 0, []),
        # Multiple proximity errors with match_thresh = 0.05
        # ts1: (1.0, 1.005, diff 0.005 < 0.05 -> err), (2.0, 2.003, diff 0.003 < 0.05 -> err)
        # ts2: (3.0, 3.008, diff 0.008 < 0.05 -> err)
        ([1.0, 1.005, 2.0, 2.003], [1,1,1,1], [3.0, 3.008, 4.0], [1,1,1], 0.05, 3,
         [
             {"file_id": "file1", "timestamp1": 1.0, "timestamp2": 1.005, "diff": pytest.approx(0.005, abs=1e-6)},
             {"file_id": "file1", "timestamp1": 2.0, "timestamp2": 2.003, "diff": pytest.approx(0.003, abs=1e-6)},
             {"file_id": "file2", "timestamp1": 3.0, "timestamp2": 3.008, "diff": pytest.approx(0.008, abs=1e-6)},
         ]
        ),
    ]
)
def test_compare_beats_data_internal_proximity(
    ts1, bc1, ts2, bc2, match_thresh, expected_prox_errors_len, expected_prox_details
):
    result = compare_beats_data(ts1, bc1, ts2, bc2, match_thresh)
    assert len(result["internal_proximity_errors"]) == expected_prox_errors_len
    if expected_prox_details:
        for expected_detail in expected_prox_details:
            found = False
            for actual_error in result["internal_proximity_errors"]:
                if (actual_error["file_id"] == expected_detail["file_id"] and
                    pytest.approx(actual_error["timestamp1"]) == expected_detail["timestamp1"] and
                    pytest.approx(actual_error["timestamp2"]) == expected_detail["timestamp2"] and
                    pytest.approx(actual_error["diff"]) == expected_detail["diff"]):
                    found = True
                    break
            assert found, f"Expected proximity error detail not found: {expected_detail}"


@pytest.mark.parametrize(
    "ts1, bc1, ts2, bc2, expected_bc_status, expected_bc_message_part, expected_bc_len1, expected_bc_len2, expected_bc_diff_idx",
    [
        (TS1_BASIC, BC1_BASIC, TS2_BASIC_MATCH, BC2_BASIC_MATCH, "match", "identical", len(BC1_BASIC), len(BC2_BASIC_MATCH), None),
        (TS1_BASIC, BC1_BASIC, TS2_BASIC_MATCH, [10, 20], "length_mismatch", "lengths differ", len(BC1_BASIC), 2, None),
        (TS1_BASIC, [10,20], TS2_BASIC_MATCH, BC2_BASIC_MATCH, "length_mismatch", "lengths differ", 2, len(BC2_BASIC_MATCH), None),
        (TS1_BASIC, BC1_BASIC, TS2_BASIC_SLIGHT_DIFF_MATCH, BC2_BASIC_CONTENT_MISMATCH, "content_mismatch", "content. First difference at index 1", len(BC1_BASIC), len(BC2_BASIC_CONTENT_MISMATCH), 1),
        (TS_EMPTY, BC_EMPTY, TS_EMPTY, BC_EMPTY, "match", "identical", 0, 0, None),
        (TS1_BASIC, BC1_BASIC, TS_EMPTY, BC_EMPTY, "length_mismatch", "lengths differ", len(BC1_BASIC), 0, None),
    ]
)
def test_compare_beats_data_beat_counts(
    ts1, bc1, ts2, bc2, expected_bc_status, expected_bc_message_part, expected_bc_len1, expected_bc_len2, expected_bc_diff_idx
):
    result = compare_beats_data(ts1, bc1, ts2, bc2)
    summary = result["beat_counts_summary"]
    assert summary["status"] == expected_bc_status
    assert expected_bc_message_part in summary["message"]
    assert summary["details"]["len1"] == expected_bc_len1
    assert summary["details"]["len2"] == expected_bc_len2
    if expected_bc_diff_idx is not None:
        assert summary["details"]["first_diff_index"] == expected_bc_diff_idx


@pytest.mark.parametrize(
    "ts1, bc1, ts2, bc2, match_thresh, expected_diff_len, expected_common, expected_unique1, expected_unique2, expected_first_diff_item",
    [
        # Identical timestamps
        (TS1_BASIC, BC1_BASIC, TS2_BASIC_MATCH, BC2_BASIC_MATCH, 0.05, 3, 3, 0, 0, {"type": "exact_match", "file1_ts": 1.0, "file2_ts": 1.0}),
        # Completely different
        ([1.0, 2.0], [1,1], [3.0, 4.0], [1,1], 0.05, 4, 0, 2, 2, {"type": "delete", "file1_ts": 1.0}),
        # Only in ts1 (deletions)
        ([1.0, 2.0], [1,1], TS_EMPTY, BC_EMPTY, 0.05, 2, 0, 2, 0, {"type": "delete", "file1_ts": 1.0}),
        # Only in ts2 (additions)
        (TS_EMPTY, BC_EMPTY, [1.0, 2.0], [1,1], 0.05, 2, 0, 0, 2, {"type": "add", "file2_ts": 1.0}),
        # Mixed: ts1 = [1.0, 2.0, 3.5], ts2 = [1.02, 2.5, 3.51]. match_thresh = 0.05
        # Diff: approx_match(1.0, 1.02), delete(2.0), add(2.5), approx_match(3.5, 3.51) -> 4 items
        ([1.0, 2.0, 3.5], [1,1,1], [1.02, 2.5, 3.51], [1,1,1], 0.05, 4, 2, 1, 1,
         {"type": "approx_match", "file1_ts": 1.0, "file2_ts": 1.02}
        ),
         # Simpler Mixed: ts1=[1,3], ts2=[1,2,3,4] -> exact_match(1), add(2), exact_match(3), add(4)
        ([1.0, 3.0], [1,1], [1.0, 2.0, 3.0, 4.0], [1,1,1,1], 0.05, 4, 2, 0, 2, {"type": "exact_match", "file1_ts": 1.0}),
        # Edge case: Approx match, diff 0.04 < threshold 0.05
        ([1.0], [1], [1.04], [1], 0.05, 1, 1, 0, 0, {"type": "approx_match", "file1_ts": 1.0, "file2_ts": 1.04}),
        # Edge case: just outside match_threshold (1.0 vs 1.051, thresh 0.05 -> delete 1.0, add 1.051)
        ([1.0], [1], [1.051], [1], 0.05, 2, 0, 1, 1, {"type": "delete", "file1_ts": 1.0}),
        # Empty lists
        (TS_EMPTY, BC_EMPTY, TS_EMPTY, BC_EMPTY, 0.05, 0, 0, 0, 0, None),
    ]
)
def test_compare_beats_data_timestamps_diff(
    ts1, bc1, ts2, bc2, match_thresh, expected_diff_len, expected_common, expected_unique1, expected_unique2, expected_first_diff_item
):
    result = compare_beats_data(ts1, bc1, ts2, bc2, match_threshold=match_thresh)
    diff = result["timestamps_diff"]
    stats = result["summary_stats"]

    assert len(diff) == expected_diff_len
    assert stats["exact_matches"] + stats["proximate_matches"] == expected_common
    assert stats["unique_to_file1"] == expected_unique1
    assert stats["unique_to_file2"] == expected_unique2

    if expected_first_diff_item:
        assert len(diff) > 0
        # Basic check of the first item's type and one key value
        assert diff[0]["type"] == expected_first_diff_item["type"]
        if "file1_ts" in expected_first_diff_item:
            assert diff[0]["file1_ts"] == pytest.approx(expected_first_diff_item["file1_ts"])
        if "file2_ts" in expected_first_diff_item:
            assert diff[0]["file2_ts"] == pytest.approx(expected_first_diff_item["file2_ts"])
    elif expected_diff_len == 0:
        assert len(diff) == 0

# Specific test for the more complex mixed case for timestamps_diff
def test_compare_beats_data_timestamps_diff_complex_mixed():
    ts1 = [1.0, 2.0, 3.5]
    bc1 = [1,1,1]
    ts2 = [1.02, 2.5, 3.51] # approx_match(1.0,1.02), delete(2.0), add(2.5), approx_match(3.5,3.51)
    bc2 = [1,1,1]
    match_thresh = 0.05

    result = compare_beats_data(ts1, bc1, ts2, bc2, match_threshold=match_thresh)
    diff = result["timestamps_diff"]
    stats = result["summary_stats"]

    assert len(diff) == 4
    assert stats["exact_matches"] == 0
    assert stats["proximate_matches"] == 2
    assert stats["unique_to_file1"] == 1 # 2.0
    assert stats["unique_to_file2"] == 1 # 2.5

    expected_diff_sequence = [
        {"type": "approx_match", "file1_ts": 1.0, "file2_ts": 1.02, "diff_ms": pytest.approx(-20.0, abs=1e-1)},
        {"type": "delete", "file1_ts": 2.0},
        {"type": "add", "file2_ts": 2.5},
        {"type": "approx_match", "file1_ts": 3.5, "file2_ts": 3.51, "diff_ms": pytest.approx(-10.0, abs=1e-1)}
    ]

    for i, expected_item in enumerate(expected_diff_sequence):
        actual_item = diff[i]
        assert actual_item["type"] == expected_item["type"]
        if "file1_ts" in expected_item:
            assert actual_item["file1_ts"] == pytest.approx(expected_item["file1_ts"])
        if "file2_ts" in expected_item:
            assert actual_item["file2_ts"] == pytest.approx(expected_item["file2_ts"])
        if "diff_ms" in expected_item:
            assert actual_item["diff_ms"] == pytest.approx(expected_item["diff_ms"])
    
    assert stats["max_match_diff_ms"] == pytest.approx(20.0) # abs value
    assert stats["avg_match_diff_ms"] == pytest.approx(-15.0) # (-20 + -10) / 2


# Tests for format_comparison_output

def test_format_comparison_output_typical_case():
    # Re-use the complex mixed case for data generation
    ts1 = [1.0, 1.005, 2.0, 3.5] # proximity error, 2.0 unique, 3.5 match 
    bc1 = [10, 20, 30, 40]
    ts2 = [1.02, 2.5, 3.51]     # 2.5 unique, 1.02 & 3.51 match
    bc2 = [10, 20, 50]          # length mismatch
    
    comparison_results = compare_beats_data(
        ts1, bc1, ts2, bc2, 
        match_threshold=0.05
    )
    # Expected proximity: (1.0, 1.005) in file1, diff=0.005. match_threshold=0.05. 0.005 < 0.05 is an error.
    # Expected bc: length mismatch (4 vs 3)
    # Expected ts_diff:
    #   approx_match (1.005, 1.02) - note: 1.0 is now part of proximity error, so 1.005 is first sorted unique for diff
    #   delete (2.0)
    #   add (2.5)
    #   approx_match (3.5, 3.51)
    # It's important to remember ts lists are sorted *within* compare_beats_data before diffing
    # So ts1 for diffing is [1.0, 1.005, 2.0, 3.5]
    # ts2 for diffing is [1.02, 2.5, 3.51]
    # Diff logic re-evaluation:
    # 1. (1.0 vs 1.02) -> diff -0.02 (abs 0.02 <= 0.05) -> APPROX_MATCH (1.0, 1.02), diff_ms -20.0
    # 2. (1.005 vs 2.5) -> 1.005 is smaller -> DELETE (1.005)
    # 3. (2.0 vs 2.5) -> 2.0 is smaller -> DELETE (2.0)
    # 4. (3.5 vs 2.5) -> 2.5 is smaller -> ADD (2.5)
    # 5. (3.5 vs 3.51) -> diff -0.01 (abs 0.01 <= 0.05) -> APPROX_MATCH (3.5, 3.51), diff_ms -10.0

    output = format_comparison_output(
        comparison_results, "fileA.beats", "fileB.beats",
        num_context_lines=0  # Explicitly 0 for this test, limit left as default (None)
    )

    assert "--- fileA.beats" in output
    assert "+++ fileB.beats" in output
    assert "Internal Timestamp Proximity Violations:" in output
    assert "WARNING: In file1, timestamps 1.000s and 1.005s are too close" in output
    assert "Beat Counts Summary:" in output
    assert "Beat counts differ" in output
    # Diff section header may or may not be present; primary concern is diff markers below
    assert "~ 1.000s | 1.020s" in output  # approximate match line
    assert "[PROXIMATE]" in output  # new label for approximate matches
    assert "- 1.005s" in output           # deletion
    assert "[ONLY IN fileA.beats]" in output  # new label for deletions
    assert "- 2.000s" in output
    assert "+ 2.500s" in output           # addition
    assert "[ONLY IN fileB.beats]" in output  # new label for additions
    assert "~ 3.500s | 3.510s" in output  # second approximate match

    assert "Summary Statistics:" in output
    assert "Exact matches: 0" in output
    assert "Proximate matches: 2" in output
    assert "Total matches: 2" in output  # new total matches line
    assert "Timestamps only in fileA.beats: 2" in output
    assert "Timestamps only in fileB.beats: 1" in output
    assert "Average match difference" in output
    assert "Maximum match difference" in output

def test_format_comparison_output_no_errors_or_diffs():
    comparison_results = compare_beats_data(TS1_BASIC, BC1_BASIC, TS2_BASIC_MATCH, BC2_BASIC_MATCH)
    output = format_comparison_output(
        comparison_results, "f1", "f2", num_context_lines=0
    )

    assert "--- f1" in output
    assert "+++ f2" in output
    assert "Internal Timestamp Proximity Violations:" not in output  # No errors
    assert "Beat counts are identical" in output
    # For identical timestamps, we show a summary rather than all timestamps
    assert "(All timestamps match exactly - no differences found)" in output
    assert "Exact matches: 3" in output
    assert "Proximate matches: 0" in output
    assert "Total matches: 3" in output  # new total matches line
    assert "Timestamps only in f1: 0" in output
    assert "Timestamps only in f2: 0" in output
    assert "Average match difference" not in output  # No difference for exact matches
    assert "Maximum match difference" not in output  # No difference for exact matches

def test_format_comparison_output_empty_inputs():
    comparison_results = compare_beats_data(TS_EMPTY, BC_EMPTY, TS_EMPTY, BC_EMPTY)
    output = format_comparison_output(
        comparison_results, "empty1", "empty2", num_context_lines=0
    )
    
    assert "--- empty1" in output
    assert "+++ empty2" in output
    assert "Internal Timestamp Proximity Violations:" not in output
    assert "Beat counts are identical" in output
    # For empty inputs, the diff section should say no differences detected
    assert "Timestamps Diff: (no differences detected)" in output
    
    # For empty input there should be no diff lines starting with +, -, or ~
    # followed by a timestamp (which would look like "+ 1.000s" etc.)
    diff_section = output.split("Timestamps Diff:")[1].split("Summary Statistics:")[0]
    assert "s [ONLY IN" not in diff_section  # No addition/deletion timestamps with labels
    assert "[PROXIMATE]" not in diff_section  # No approximate match timestamps
    
    assert "Exact matches: 0" in output
    assert "Proximate matches: 0" in output
    assert "Total matches: 0" in output  # new total matches line
    assert "Timestamps only in empty1: 0" in output
    assert "Timestamps only in empty2: 0" in output

# Test for the new verbose parameter
def test_format_comparison_output_verbose():
    ts1 = [1.0, 2.0]
    bc1 = [1, 1]
    ts2 = [1.0, 3.0]  # 1.0 exact match, 2.0 deletion, 3.0 addition
    bc2 = [1, 1]
    
    comparison_results = compare_beats_data(ts1, bc1, ts2, bc2)
    output = format_comparison_output(
        comparison_results, "verbose1", "verbose2", 
        num_context_lines=0, verbose=True
    )
    
    assert "Raw Timestamp Diff Data:" in output
    assert "Item 1:" in output
    assert "'type': 'exact_match'" in output
    assert "'file1_ts': 1.0" in output
    
    # Check non-verbose mode for comparison
    non_verbose_output = format_comparison_output(
        comparison_results, "verbose1", "verbose2", 
        num_context_lines=0, verbose=False
    )
    assert "Raw Timestamp Diff Data:" not in non_verbose_output


# ---------------------------------------------------------------------------
# Additional tests for the num_context_lines (context-hunk) feature
# ---------------------------------------------------------------------------


def _build_context_diff_test_data():
    """Utility to generate a simple data set with a single insertion so the
    presence/absence of context lines is easy to detect."""
    ts1 = [1.0, 2.0, 3.0]          # Original timestamps
    ts2 = [1.0, 2.5, 3.0]          # ts2 inserts 2.5 between 2.0 and 3.0
    bc1 = bc2 = [1, 1, 1]          # Beat-counts identical (not under test here)
    comparison_results = compare_beats_data(ts1, bc1, ts2, bc2, match_threshold=0.05)
    return comparison_results


def test_format_comparison_output_no_context_lines():
    """When num_context_lines = 0 we expect only change lines (+ / -) to show, no
    surrounding exact-match context lines."""
    comparison_results = _build_context_diff_test_data()

    output = format_comparison_output(
        comparison_results,
        "base.beats",
        "variant.beats",
        num_context_lines=0,
    )

    # Change lines should be present
    assert "- 2.000s" in output
    assert "+ 2.500s" in output

    # Exact-match context lines should *not* be present
    # (they would start with a leading space)
    assert " 1.000s" not in output
    assert " 3.000s" not in output


def test_format_comparison_output_with_context_lines():
    """With num_context_lines > 0 we expect surrounding identical lines to be
    included once per change hunk."""
    comparison_results = _build_context_diff_test_data()

    output = format_comparison_output(
        comparison_results,
        "base.beats",
        "variant.beats",
        num_context_lines=1,
    )

    # Change lines still present
    assert "- 2.000s" in output
    assert "+ 2.500s" in output

    # Now context lines should appear exactly once
    assert output.count(" 1.000s") == 1
    assert output.count(" 3.000s") == 1


if __name__ == "__main__":
    # This allows running the tests directly for easier debugging.
    # You can execute this file with python -m beat_detection.utils.test_beats_compare
    # or, if pytest is on your path, just by running this file.
    # You can also pass pytest arguments, e.g., python -m beat_detection.utils.test_beats_compare -k "test_format_comparison_output_typical_case"
    pytest.main([__file__]) 