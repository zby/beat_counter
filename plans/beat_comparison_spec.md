# Plan: .beats File Comparison Specification

## 1. Overview

This document outlines the plan for creating a utility to compare two `.beats` files. The utility will identify differences in timestamps and beat counts, provide a formatted diff output, and include specific checks for timestamp integrity.

The primary goals are:
- To provide a clear, `diff -u` like comparison for timestamps.
- To report on the consistency of beat counts.
- To implement a configurable threshold for matching timestamps between the two files.
- To implement a check for timestamps within the *same* file that are too close to each other, indicating potential issues in the beat detection process.

## 2. File Structure of `.beats` Files (Assumed)

It's assumed that `.beats` files primarily contain:
1.  A series of timestamps (float values representing seconds).
2.  A series of beat counts (integer values).

The parsing of these files into appropriate data structures (e.g., lists of floats for timestamps, lists of integers for beat counts) is a prerequisite and considered outside the scope of the functions defined here, which will operate on the parsed data.

## 3. Core Functions

Two main Python functions will be developed:

### 3.1. `compare_beats_data`

This function will perform the logical comparison of the data extracted from two `.beats` files.

**Signature:**
```python
def compare_beats_data(
    timestamps1: list[float],
    beat_counts1: list[int],
    timestamps2: list[float],
    beat_counts2: list[int],
    match_threshold: float = 0.05, # Default: 50ms
) -> dict:
    # ... implementation ...
```

**Inputs:**
- `timestamps1`: A list of float values representing beat timestamps from the first file.
- `beat_counts1`: A list of integer values representing beat counts from the first file.
- `timestamps2`: A list of float values representing beat timestamps from the second file.
- `beat_counts2`: A list of integer values representing beat counts from the second file.
- `match_threshold`: A float (in seconds) defining the maximum difference for two timestamps (one from each file) to be considered a match. It's also used as the minimum difference between consecutive timestamps *within the same file*. If any two are closer than this, it's flagged.

**Processing:**

1.  **Internal Timestamp Proximity Check:**
    - For `timestamps1`: Iterate through sorted unique timestamps. If `abs(t_i - t_{i+1}) < match_threshold` for any consecutive pair, record an error/warning object containing the file identifier (e.g., "file1"), the problematic timestamps, and their difference.
    - Repeat for `timestamps2`.

2.  **Beat Counts Comparison:**
    - Compare `beat_counts1` and `beat_counts2`.
    - Determine if they are:
        - Identical.
        - Different lengths.
        - Same length but different content (note first differing index and values).
    - Store this information in the result.

3.  **Timestamps Diff Generation (Cross-File):**
    - This is the core diffing logic. The aim is to produce a list of changes that can be formatted similarly to `diff -u`.
    - A common approach is to use a variation of the Longest Common Subsequence (LCS) algorithm or a simpler two-pointer scan if strict ordering is maintained and large shifts are not expected. Given the "Fail Fast" principle and desire for clarity, a two-pointer approach seems appropriate for an initial version.
    - Iterate through sorted `timestamps1` (pointer `i`) and `timestamps2` (pointer `j`):
        - If `i` reaches end of `timestamps1`, remaining `timestamps2[j:]` are additions.
        - If `j` reaches end of `timestamps2`, remaining `timestamps1[i:]` are deletions.
        - If `abs(timestamps1[i] - timestamps2[j]) <= match_threshold`: This is a match. Record as `(type: "match", ts1: timestamps1[i], ts2: timestamps2[j], diff: timestamps1[i] - timestamps2[j])`. Increment `i` and `j`.
        - If `timestamps1[i] < timestamps2[j] - match_threshold`: `timestamps1[i]` is unique to file 1 (a "deletion" relative to file 2). Record as `(type: "delete", ts: timestamps1[i])`. Increment `i`.
        - If `timestamps2[j] < timestamps1[i] - match_threshold`: `timestamps2[j]` is unique to file 2 (an "addition" relative to file 1). Record as `(type: "add", ts: timestamps2[j])`. Increment `j`.
        - If timestamps are close but not within `match_threshold` (e.g., `timestamps1[i] < timestamps2[j]` but `timestamps1[i] > timestamps2[j] - some_larger_context_window`), this might indicate a shifted block. For V1, we'll stick to the simpler cases above. A more advanced diff could be considered later.

**Outputs:**
A dictionary containing the comparison results, structured for easy formatting:
```json
{
    "internal_proximity_errors": [
        {"file_id": "file1", "timestamp1": 1.23, "timestamp2": 1.235, "diff": 0.005},
        // ... more errors ...
    ],
    "beat_counts_summary": {
        "status": "mismatch", // "match", "length_mismatch", "content_mismatch"
        "message": "Beat counts differ. File 1 has 10, File 2 has 12.",
        "details": { // Optional, for more specific info
            "len1": 10,
            "len2": 12,
            "first_diff_index": null // or index if same length
        }
    },
    "timestamps_diff": [
        // Example entries:
        {"type": "match", "file1_ts": 1.01, "file2_ts": 1.03, "diff_ms": 20.0},
        {"type": "delete", "file1_ts": 1.50}, // Timestamp only in file1
        {"type": "add", "file2_ts": 1.65},    // Timestamp only in file2
        // ... more diff entries ...
    ],
    "summary_stats": {
        "common_timestamps": 0,
        "unique_to_file1": 0,
        "unique_to_file2": 0,
        "max_match_diff_ms": 0.0,
        "avg_match_diff_ms": 0.0
    }
}
```

### 3.2. `format_comparison_output`

This function will take the output from `compare_beats_data` and generate a human-readable, nicely formatted string.

**Signature:**
```python
def format_comparison_output(
    comparison_results: dict,
    file1_name: str = "file1",
    file2_name: str = "file2",
    limit: int | None = None,          # maximum number of diff lines to print (None ⇒ no limit)
    num_context_lines: int = 3,        # unchanged lines to show before/after each diff hunk
) -> str:
    # ... implementation ...
```

**Inputs:**
- `comparison_results`: The dictionary returned by `compare_beats_data`.
- `file1_name`: A string name/identifier for the first file (e.g., its path).
- `file2_name`: A string name/identifier for the second file.
- `limit`: Optional integer limiting the total number of *diff* lines (after context expansion). If `None`, everything is printed.
- `num_context_lines`: How many surrounding identical-timestamp lines to show around each diff hunk, similar to the `-U`/`--unified` option of `diff`.

**Processing:**

1.  **Header:**
    - Print a header comparing the two file names.
    - `--- {file1_name}`
    - `+++ {file2_name}`

2.  **Internal Proximity Errors:**
    - If any errors exist in `comparison_results["internal_proximity_errors"]`:
        - Print a section header (e.g., "Internal Timestamp Proximity Violations:").
        - For each error: `WARNING: In {error['file_id']}, timestamps {error['timestamp1']:.3f}s and {error['timestamp2']:.3f}s are too close (diff: {error['diff'] * 1000:.1f}ms). Threshold for this check is {match_threshold * 1000:.1f}ms.`

3.  **Beat Counts Summary:**
    - Print the `comparison_results["beat_counts_summary"]["message"]`.

4.  **Timestamps Diff:**
    - Iterate through `comparison_results["timestamps_diff"]` to build unified diff *hunks*.
    - Each output line is prefixed with a single-character marker:
        - `+`  → timestamp exists **only** in *file2* (addition).
        - `-`  → timestamp exists **only** in *file1* (deletion).
        - `~`  → timestamps are within `match_threshold` but not identical ("approximate match").
        - ` `  → timestamps are *exactly* identical and are emitted as **context** lines.
    - For each group of adjacent non-context changes, include up to `num_context_lines` preceding and following identical lines.
    - If `limit` is not `None`, truncate the diff output once that many lines (including context) have been produced. Append a final line like `... (output truncated)` to signal the omission.

5.  **Summary Statistics:**
    - Print a summary section:
        - `Matched timestamps: {summary_stats['common_timestamps']}`
        - `Timestamps only in {file1_name}: {summary_stats['unique_to_file1']}`
        - `Timestamps only in {file2_name}: {summary_stats['unique_to_file2']}`
        - If matched timestamps > 0:
            - `Average match difference: {summary_stats['avg_match_diff_ms']:.1f}ms`
            - `Maximum match difference: {summary_stats['max_match_diff_ms']:.1f}ms`

**Outputs:**
- A single multi-line string containing the formatted comparison report.

**Example Diff (with `num_context_lines = 2` and `limit = None`):**
```diff
--- song_A.beats
+++ song_B.beats
Internal Timestamp Proximity Violations:
WARNING: In file2, timestamps 12.000s and 12.030s are too close (diff: 30.0 ms). Threshold is 50.0 ms.

Beat counts differ. File 1 has 128 beats, File 2 has 130 beats.

@@ 11.00 – 13.50 @@
  11.000s
~ 11.500s | 11.520s (diff: 20.0 ms)
- 12.750s
+ 12.900s
  13.000s
  13.250s

@@ 45.00 – 47.00 @@
  45.000s
- 45.250s
~ 45.500s | 45.540s (diff: 40.0 ms)
+ 45.800s
  46.000s

Matched timestamps: 120
Timestamps only in song_A.beats: 5
Timestamps only in song_B.beats: 7
Average match difference: 18.4 ms
Maximum match difference: 45.0 ms
```

## 4. Fail-Fast Considerations

- **Input Validation:** The `compare_beats_data` function should assume inputs (`timestamps1`, `beat_counts1`, etc.) are already validated (e.g., lists of correct types). If pre-parsed data is not guaranteed, initial validation steps would be needed, raising `TypeError` or `ValueError` on incorrect input types or formats.
- **No Implicit Defaults (for data):** The functions operate on provided data. Missing data should be handled by the calling code.
- **Clear Error Messages:** The internal proximity check should produce clear messages. The diff output itself is the primary "error" report for discrepancies.

## 5. Testing Strategy

Only unit tests - no filesystem interaction.

- **Unit Tests for `compare_beats_data`:**
    - Test cases for internal proximity: no violations, one violation, multiple violations.
    - Test cases for beat counts: identical, different lengths, same length but different values.
    - Test cases for timestamps diff:
        - Identical timestamp lists.
        - Completely different timestamp lists.
        - Timestamps only in file1.
        - Timestamps only in file2.
        - Mixed additions, deletions, and matches.
        - Edge cases for `match_threshold` (timestamps exactly on threshold, just outside).
        - Empty timestamp lists.
- **Unit Tests for `format_comparison_output`:**
    - Test that each section (proximity errors, beat counts, timestamps diff, summary) is formatted correctly based on various inputs from `compare_beats_data`.
    - Test with different file names.
- Tests should be co-located with the code if they are simple unit tests. Heavier integration tests (e.g., involving actual file parsing) would go into the `/tests/` directory.

## 6. Future Considerations (Out of Scope for V1)

- More sophisticated diff algorithms (e.g., Myers diff) for timestamps if the simple two-pointer approach proves insufficient for complex reorderings or shifts.
- Visualizations of the diff.
- Handling different `.beats` file format versions if they arise.
- Option for "smart matching" that might try to align sequences of beats even if there are some insertions/deletions in between, beyond the simple `match_threshold` for individual beats. The current `compare_experiments.py` script has a `--smart-match` option; this spec focuses on a more direct, `diff -u` style comparison first. 