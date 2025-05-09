"""
Beat Comparison Utilities

This module provides utilities for comparing beat detection results,
specifically parsed timestamp data and beat counts from two sources.
It focuses on data comparison rather than file operations.
"""

import json
import difflib
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set, Optional


def find_beats_files(experiment_dir: Path) -> List[Path]:
    """
    Find all .beats files in an experiment directory.
    
    Parameters
    ----------
    experiment_dir : Path
        Path to the experiment directory.
        
    Returns
    -------
    List[Path]
        List of paths to .beats files.
        
    Raises
    ------
    FileNotFoundError
        If the experiment directory doesn't exist.
    """
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    return list(experiment_dir.glob("**/*.beats"))


def normalize_path(experiment_dir: Path, beats_file: Path) -> str:
    """
    Normalize the path of a beats file to be relative to the experiment directory.
    
    Parameters
    ----------
    experiment_dir : Path
        Path to the experiment directory.
    beats_file : Path
        Path to the beats file.
        
    Returns
    -------
    str
        Normalized path.
    """
    return str(beats_file.relative_to(experiment_dir))


def find_matching_timestamps(array1: List[float], array2: List[float], 
                            max_diff_sec: float = 0.1) -> Dict[str, Any]:
    """
    Find matching timestamps between two arrays within a tolerance.
    
    Parameters
    ----------
    array1 : List[float]
        First array of timestamps.
    array2 : List[float]
        Second array of timestamps.
    max_diff_sec : float
        Maximum difference in seconds to consider two timestamps as matching.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing matching information.
    """
    matches = []
    unmatched1 = set(range(len(array1)))
    unmatched2 = set(range(len(array2)))
    
    # For each timestamp in array1, find the closest match in array2
    for i, ts1 in enumerate(array1):
        best_match = None
        best_diff = float('inf')
        
        for j, ts2 in enumerate(array2):
            diff = abs(ts1 - ts2)
            if diff < best_diff:
                best_diff = diff
                best_match = j
        
        # If the best match is within tolerance, consider it a match
        if best_match is not None and best_diff <= max_diff_sec:
            matches.append({
                "index1": i,
                "index2": best_match,
                "time1": ts1,
                "time2": array2[best_match],
                "diff_ms": best_diff * 1000
            })
            unmatched1.discard(i)
            unmatched2.discard(best_match)
    
    # Find best unique matches (multiple timestamps in array1 might match the same timestamp in array2)
    # Sort matches by difference (smallest first)
    matches.sort(key=lambda x: x["diff_ms"])
    
    # Keep track of which indices we've matched
    matched_indices1 = set()
    matched_indices2 = set()
    unique_matches = []
    
    for match in matches:
        idx1, idx2 = match["index1"], match["index2"]
        # Only keep matches where neither index has been matched yet
        if idx1 not in matched_indices1 and idx2 not in matched_indices2:
            unique_matches.append(match)
            matched_indices1.add(idx1)
            matched_indices2.add(idx2)
    
    # Convert unmatched indices to actual timestamps
    unmatched_ts1 = [array1[i] for i in unmatched1]
    unmatched_ts2 = [array2[i] for i in unmatched2]
    
    # Calculate statistics
    if unique_matches:
        diffs_ms = [m["diff_ms"] for m in unique_matches]
        avg_diff_ms = sum(diffs_ms) / len(diffs_ms)
        max_diff_ms = max(diffs_ms)
    else:
        avg_diff_ms = 0
        max_diff_ms = 0
    
    return {
        "matched_count": len(unique_matches),
        "unmatched_count1": len(unmatched_ts1),
        "unmatched_count2": len(unmatched_ts2),
        "matches": unique_matches,
        "unmatched_timestamps1": unmatched_ts1,
        "unmatched_timestamps2": unmatched_ts2,
        "match_stats": {
            "avg_diff_ms": avg_diff_ms,
            "max_diff_ms": max_diff_ms
        }
    }


def compare_arrays(array1: List, array2: List, is_timestamps: bool = False, 
                  max_diff_sec: float = 0.1) -> Dict[str, Any]:
    """
    Compare two arrays (timestamps or beat_counts) and provide detailed analysis.
    
    Parameters
    ----------
    array1 : List
        First array.
    array2 : List
        Second array.
    is_timestamps : bool
        Flag indicating if we're comparing timestamp arrays (for time-based comparisons).
    max_diff_sec : float
        Maximum difference in seconds to consider two timestamps as matching.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing the comparison analysis.
    """
    result = {
        "count_diff": len(array1) - len(array2),
        "first_count": len(array1),
        "second_count": len(array2)
    }
    
    # Early output if one array is empty
    if not array1 or not array2:
        return result
    
    # For timestamps, do detailed timing analysis
    if is_timestamps:
        # Try to find matching timestamps first (smart matching)
        match_result = find_matching_timestamps(array1, array2, max_diff_sec)
        result["matching"] = match_result
        
        # Also do the index-by-index comparison for completeness
        timing_diffs = []
        max_common_index = min(len(array1), len(array2))
        
        for i in range(max_common_index):
            diff = array1[i] - array2[i]
            if abs(diff) > 0.001:  # Small threshold to account for floating point imprecision
                timing_diffs.append({
                    "index": i,
                    "first_time": array1[i],
                    "second_time": array2[i],
                    "diff_ms": diff * 1000  # Convert to milliseconds
                })
        
        # Include summary statistics if there are differences
        if timing_diffs:
            result["differences"] = timing_diffs
            diffs_ms = [d["diff_ms"] for d in timing_diffs]
            result["timing_stats"] = {
                "max_diff_ms": max(diffs_ms, key=abs),
                "avg_diff_ms": sum(diffs_ms) / len(diffs_ms),
                "diff_count": len(timing_diffs)
            }
        
        # Include extra timestamps if counts differ
        if len(array1) > len(array2):
            result["extra_in_first"] = array1[max_common_index:]
        elif len(array2) > len(array1):
            result["extra_in_second"] = array2[max_common_index:]
    
    # For beat_counts, find value differences
    else:
        value_diffs = []
        max_common_index = min(len(array1), len(array2))
        
        for i in range(max_common_index):
            if array1[i] != array2[i]:
                value_diffs.append({
                    "index": i,
                    "first_value": array1[i],
                    "second_value": array2[i]
                })
        
        if value_diffs:
            result["differences"] = value_diffs
            result["diff_count"] = len(value_diffs)
        
        # Include extra values if counts differ
        if len(array1) > len(array2):
            result["extra_in_first"] = array1[max_common_index:]
        elif len(array2) > len(array1):
            result["extra_in_second"] = array2[max_common_index:]
    
    return result


def compare_beats_files(exp1_dir: Path, exp2_dir: Path, 
                       max_diff_sec: float = 0.1) -> Tuple[List[str], List[str], List[Tuple[str, Dict]]]:
    """
    Compare .beats files between two experiment directories.
    Files are expected to be JSON containing 'timestamps' and 'beat_counts'.
    
    Parameters
    ----------
    exp1_dir : Path
        Path to the first experiment directory.
    exp2_dir : Path
        Path to the second experiment directory.
    max_diff_sec : float
        Maximum difference in seconds to consider two timestamps as matching.
        
    Returns
    -------
    Tuple[List[str], List[str], List[Tuple[str, Dict]]]
        Tuple containing:
        - List of beats files only in exp1
        - List of beats files only in exp2
        - List of tuples of (file_path, differences_or_error) for files that differ
          or do not conform to the expected structure. The dictionary will
          contain an 'error' key if structure validation fails.
          
    Raises
    ------
    FileNotFoundError
        If either experiment directory doesn't exist.
    """
    # Find all .beats files
    exp1_beats_files = find_beats_files(exp1_dir)
    exp2_beats_files = find_beats_files(exp2_dir)
    
    # Normalize paths
    exp1_normalized = {normalize_path(exp1_dir, f): f for f in exp1_beats_files}
    exp2_normalized = {normalize_path(exp2_dir, f): f for f in exp2_beats_files}
    
    # Find files only in one experiment
    exp1_only = [p for p in exp1_normalized.keys() if p not in exp2_normalized]
    exp2_only = [p for p in exp2_normalized.keys() if p not in exp1_normalized]
    
    # Compare common files
    common_paths = set(exp1_normalized.keys()) & set(exp2_normalized.keys())
    different_files = []
    
    for path in common_paths:
        exp1_file_path = exp1_normalized[path]
        exp2_file_path = exp2_normalized[path]
        
        exp1_content = None
        exp2_content = None
        error_messages = []

        # Validate and load first file
        try:
            with open(exp1_file_path, 'r') as f1:
                exp1_content = json.load(f1)
            if not isinstance(exp1_content, dict):
                error_messages.append(f"File {exp1_file_path.name} (from {exp1_dir.name}) is not a JSON dictionary.")
            else:
                if 'timestamps' not in exp1_content:
                    error_messages.append(f"File {exp1_file_path.name} (from {exp1_dir.name}) missing 'timestamps' field.")
                if 'beat_counts' not in exp1_content:
                    error_messages.append(f"File {exp1_file_path.name} (from {exp1_dir.name}) missing 'beat_counts' field.")
        except json.JSONDecodeError:
            error_messages.append(f"File {exp1_file_path.name} (from {exp1_dir.name}) is not valid JSON.")
        except Exception as e:
            error_messages.append(f"Error reading file {exp1_file_path.name} (from {exp1_dir.name}): {e}")

        # Validate and load second file
        try:
            with open(exp2_file_path, 'r') as f2:
                exp2_content = json.load(f2)
            if not isinstance(exp2_content, dict):
                error_messages.append(f"File {exp2_file_path.name} (from {exp2_dir.name}) is not a JSON dictionary.")
            else:
                if 'timestamps' not in exp2_content:
                    error_messages.append(f"File {exp2_file_path.name} (from {exp2_dir.name}) missing 'timestamps' field.")
                if 'beat_counts' not in exp2_content:
                    error_messages.append(f"File {exp2_file_path.name} (from {exp2_dir.name}) missing 'beat_counts' field.")
        except json.JSONDecodeError:
            error_messages.append(f"File {exp2_file_path.name} (from {exp2_dir.name}) is not valid JSON.")
        except Exception as e:
            error_messages.append(f"Error reading file {exp2_file_path.name} (from {exp2_dir.name}): {e}")
            
        if error_messages:
            different_files.append((path, {"error": "; ".join(error_messages)}))
            continue

        # If we reach here, both files are valid JSON dicts with the required keys.
        file_specific_differences = {}
        
        # Compare timestamps
        # Ensure fields exist before accessing, though validated above, this is defensive.
        # Type validation of list elements could be added here or in compare_arrays if strictness is needed.
        if 'timestamps' in exp1_content and 'timestamps' in exp2_content:
            if exp1_content['timestamps'] != exp2_content['timestamps']:
                file_specific_differences['timestamps'] = compare_arrays(
                    exp1_content['timestamps'], 
                    exp2_content['timestamps'],
                    is_timestamps=True,
                    max_diff_sec=max_diff_sec
                )
        
        # Compare beat_counts
        if 'beat_counts' in exp1_content and 'beat_counts' in exp2_content:
            if exp1_content['beat_counts'] != exp2_content['beat_counts']:
                file_specific_differences['beat_counts'] = compare_arrays(
                    exp1_content['beat_counts'], 
                    exp2_content['beat_counts']
                )
        
        if file_specific_differences:
            different_files.append((path, file_specific_differences))
            
    return exp1_only, exp2_only, different_files 


def compare_beats_data(
    timestamps1: List[float],
    beat_counts1: List[int],
    timestamps2: List[float],
    beat_counts2: List[int],
    match_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Compares beat data (timestamps and counts) from two sources.

    Args:
        timestamps1: List of float timestamps from the first source.
        beat_counts1: List of integer beat counts from the first source.
        timestamps2: List of float timestamps from the second source.
        beat_counts2: List of integer beat counts from the second source.
        match_threshold: Max difference (seconds) for two timestamps
                         (one from each file) to be considered a match.
                         Also used as the min difference for internal
                         proximity checks.

    Returns:
        A dictionary containing detailed comparison results.
    """
    results: Dict[str, Any] = {
        "internal_proximity_errors": [],
        "beat_counts_summary": {},
        "timestamps_diff": [],
        "summary_stats": {
            "common_timestamps": 0,
            "unique_to_file1": 0,
            "unique_to_file2": 0,
            "max_match_diff_ms": 0.0,
            "avg_match_diff_ms": 0.0
        }
    }

    # 1. Internal Timestamp Proximity Check
    for file_id, ts_list in [("file1", timestamps1), ("file2", timestamps2)]:
        sorted_unique_ts = sorted(list(set(ts_list)))
        for i in range(len(sorted_unique_ts) - 1):
            t1 = sorted_unique_ts[i]
            t2 = sorted_unique_ts[i+1]
            diff = t2 - t1
            if diff < match_threshold:
                results["internal_proximity_errors"].append({
                    "file_id": file_id,
                    "timestamp1": t1,
                    "timestamp2": t2,
                    "diff": diff
                })

    # 2. Beat Counts Comparison
    if beat_counts1 == beat_counts2:
        results["beat_counts_summary"] = {
            "status": "match",
            "message": f"Beat counts are identical ({len(beat_counts1)} counts).",
            "details": {"len1": len(beat_counts1), "len2": len(beat_counts2)}
        }
    elif len(beat_counts1) != len(beat_counts2):
        results["beat_counts_summary"] = {
            "status": "length_mismatch",
            "message": f"Beat count lengths differ. File 1 has {len(beat_counts1)}, File 2 has {len(beat_counts2)}.",
            "details": {"len1": len(beat_counts1), "len2": len(beat_counts2)}
        }
    else: # Same length, different content
        first_diff_index = -1
        for i in range(len(beat_counts1)):
            if beat_counts1[i] != beat_counts2[i]:
                first_diff_index = i
                break
        results["beat_counts_summary"] = {
            "status": "content_mismatch",
            "message": f"Beat counts have the same length ({len(beat_counts1)}) but differ in content. First difference at index {first_diff_index}.",
            "details": {
                "len1": len(beat_counts1),
                "len2": len(beat_counts2),
                "first_diff_index": first_diff_index
            }
        }

    # 3. Timestamps Diff Generation (Cross-File)
    # Ensure timestamps are sorted for the diff algorithm
    sorted_ts1 = sorted(timestamps1)
    sorted_ts2 = sorted(timestamps2)

    ptr1, ptr2 = 0, 0
    diff_list = []
    match_diffs_ms = []

    while ptr1 < len(sorted_ts1) or ptr2 < len(sorted_ts2):
        if ptr1 < len(sorted_ts1) and ptr2 < len(sorted_ts2):
            ts1_val = sorted_ts1[ptr1]
            ts2_val = sorted_ts2[ptr2]
            diff = ts1_val - ts2_val

            if abs(diff) <= match_threshold:
                diff_ms = diff * 1000
                diff_list.append({
                    "type": "match",
                    "file1_ts": ts1_val,
                    "file2_ts": ts2_val,
                    "diff_ms": diff_ms
                })
                match_diffs_ms.append(diff_ms)
                results["summary_stats"]["common_timestamps"] += 1
                ptr1 += 1
                ptr2 += 1
            elif ts1_val < ts2_val: # (and diff is > match_threshold, or ts1_val < ts2_val - match_threshold)
                diff_list.append({"type": "delete", "file1_ts": ts1_val})
                results["summary_stats"]["unique_to_file1"] += 1
                ptr1 += 1
            else: # ts2_val < ts1_val (and diff is < -match_threshold, or ts2_val < ts1_val - match_threshold)
                diff_list.append({"type": "add", "file2_ts": ts2_val})
                results["summary_stats"]["unique_to_file2"] += 1
                ptr2 += 1
        elif ptr1 < len(sorted_ts1):
            ts1_val = sorted_ts1[ptr1]
            diff_list.append({"type": "delete", "file1_ts": ts1_val})
            results["summary_stats"]["unique_to_file1"] += 1
            ptr1 += 1
        elif ptr2 < len(sorted_ts2):
            ts2_val = sorted_ts2[ptr2]
            diff_list.append({"type": "add", "file2_ts": ts2_val})
            results["summary_stats"]["unique_to_file2"] += 1
            ptr2 += 1
        else: # Should not happen if loop condition is correct
            break 
            
    results["timestamps_diff"] = diff_list

    # Calculate summary statistics for matches
    if results["summary_stats"]["common_timestamps"] > 0 and match_diffs_ms:
        results["summary_stats"]["avg_match_diff_ms"] = sum(match_diffs_ms) / len(match_diffs_ms)
        # Use abs for max_match_diff_ms to report largest deviation magnitude
        abs_match_diffs_ms = [abs(d) for d in match_diffs_ms]
        results["summary_stats"]["max_match_diff_ms"] = max(abs_match_diffs_ms)
    
    return results


def format_comparison_output(
    comparison_results: Dict[str, Any],
    file1_name: str = "file1",
    file2_name: str = "file2",
    limit: Optional[int] = None,
    num_context_lines: int = 2
) -> str:
    """
    Formats the comparison results from compare_beats_data into a
    human-readable string.

    Args:
        comparison_results: The dictionary from compare_beats_data.
        file1_name: Name/identifier for the first file.
        file2_name: Name/identifier for the second file.
        limit: Optional maximum number of diff items to display.
        num_context_lines: Number of surrounding identical-timestamp lines to
            emit before and after each diff hunk (similar to `-u` in `diff`).

    Returns:
        A multi-line string containing the formatted report.
    """
    output_lines = []

    # ------------------------------------------------------------------
    # 1. Header
    # ------------------------------------------------------------------
    output_lines.append(f"--- {file1_name}")
    output_lines.append(f"+++ {file2_name}")
    output_lines.append("")

    # ------------------------------------------------------------------
    # 2. Internal proximity violations
    # ------------------------------------------------------------------
    proximity_errors = comparison_results.get("internal_proximity_errors", [])
    if proximity_errors:
        output_lines.append("Internal Timestamp Proximity Violations:")
        for err in proximity_errors:
            diff_ms = err["diff"] * 1000.0
            output_lines.append(
                f"WARNING: In {err['file_id']}, timestamps {err['timestamp1']:.3f}s and "
                f"{err['timestamp2']:.3f}s are too close (diff: {diff_ms:.1f}ms)."
            )
        output_lines.append("")

    # ------------------------------------------------------------------
    # 3. Beat-count summary
    # ------------------------------------------------------------------
    beat_counts_summary = comparison_results.get("beat_counts_summary", {})
    if beat_counts_summary.get("message"):
        output_lines.append("Beat Counts Summary:")
        # Ensure phrase "Beat counts differ" appears for any non-match status
        if beat_counts_summary.get("status") != "match":
            output_lines.append("  Beat counts differ.")
        output_lines.append(f"  {beat_counts_summary['message']}")
        output_lines.append("")

    # ------------------------------------------------------------------
    # 4. Timestamp diff
    # ------------------------------------------------------------------
    timestamps_diff = comparison_results.get("timestamps_diff", [])

    if not timestamps_diff:
        output_lines.append("Timestamps Diff: (no differences detected)")
    else:
        output_lines.append("Timestamps Diff:")

    # ------------------------------------------------------------------
    # Convert diff items to textual lines (+/-/~ and context) and mark changes
    # ------------------------------------------------------------------
    diff_text: List[str] = []
    change_flags: List[bool] = []  # True if the line is an addition/deletion/approx match

    for item in timestamps_diff:
        t_type = item["type"]
        if t_type == "match":
            diff_ms = abs(item.get("diff_ms", 0.0))
            # todo: check if this is robust for floats comparison
            if diff_ms == 0:
                diff_text.append(f"  {item['file1_ts']:.3f}s")  # exact match – context candidate
                change_flags.append(False)
            else:
                diff_text.append(
                    f"~ {item['file1_ts']:.3f}s | {item['file2_ts']:.3f}s (diff: {diff_ms:.1f} ms)"
                )
                change_flags.append(True)
        elif t_type == "delete":
            ts_val = item["file1_ts"]
            diff_text.append(f"- {ts_val:.3f}s")
            change_flags.append(True)
        elif t_type == "add":
            ts_val = item["file2_ts"]
            diff_text.append(f"+ {ts_val:.3f}s")
            change_flags.append(True)

    # ------------------------------------------------------------------
    # Apply context window based on num_context_lines
    # ------------------------------------------------------------------
    if num_context_lines is None or num_context_lines < 0:
        emitted_lines = diff_text
    else:
        indices_to_include: Set[int] = set()
        for idx, is_change in enumerate(change_flags):
            if is_change:
                start = max(0, idx - num_context_lines)
                end = min(len(diff_text) - 1, idx + num_context_lines)
                for i in range(start, end + 1):
                    indices_to_include.add(i)

        if not indices_to_include:
            # No actual changes ⇒ include entire context (all lines)
            emitted_lines = diff_text
        else:
            # Ensure deterministic order
            emitted_lines = [diff_text[i] for i in range(len(diff_text)) if i in indices_to_include]

    # ------------------------------------------------------------------
    # Apply limit if requested
    # ------------------------------------------------------------------
    if limit is not None and len(emitted_lines) > limit:
        truncated = emitted_lines[:limit]
        truncated.append("... (output truncated)")
        emitted_lines = truncated

    output_lines.extend(emitted_lines)
    output_lines.append("")

    # ------------------------------------------------------------------
    # 5. Summary statistics
    # ------------------------------------------------------------------
    stats = comparison_results.get("summary_stats", {})
    output_lines.append("Summary Statistics:")
    output_lines.append(f"  Matched timestamps: {stats.get('common_timestamps', 0)}")
    output_lines.append(f"  Timestamps only in {file1_name}: {stats.get('unique_to_file1', 0)}")
    output_lines.append(f"  Timestamps only in {file2_name}: {stats.get('unique_to_file2', 0)}")
    if stats.get("common_timestamps", 0):
        output_lines.append(
            f"  Average match difference: {stats.get('avg_match_diff_ms', 0.0):.1f} ms"
        )
        output_lines.append(
            f"  Maximum match difference: {stats.get('max_match_diff_ms', 0.0):.1f} ms"
        )

    return "\n".join(output_lines)


# Example Usage (can be removed or kept for testing):
if __name__ == "__main__":
    # Test Case 1: Basic differences
    ts1_test = [1.0, 2.0, 3.0, 4.0, 4.1, 4.2, 5.0]
    bc1_test = [1, 1, 1, 1, 1, 1, 1]
    ts2_test = [1.01, 2.0, 3.03, 4.15, 5.0, 6.0, 6.1]
    bc2_test = [1, 1, 0, 1, 1, 1, 1]

    print(f"--- Test Case 1 (Context Lines Test) ---")
    s_ts1 = sorted(ts1_test)
    s_ts2 = sorted(ts2_test)
    results_test1 = compare_beats_data(s_ts1, bc1_test, s_ts2, bc2_test, match_threshold=0.05)
    # Using num_context_lines=2 for a more thorough test of context
    print(format_comparison_output(results_test1, "fileA.beats", "fileB.beats", num_context_lines=2, limit=10))
    print("\n\n")

    # Test Case 2: Internal proximity
    ts1_prox = [1.0, 1.005, 2.0]
    bc1_prox = [1,1,1]
    ts2_prox = [1.0, 1.06, 2.0]
    bc2_prox = [1,1,1]

    print(f"--- Test Case 2: Internal Proximity ---")
    s_ts1_prox = sorted(ts1_prox)
    s_ts2_prox = sorted(ts2_prox)
    results_test2 = compare_beats_data(
        s_ts1_prox, bc1_prox, s_ts2_prox, bc2_prox, 
        match_threshold=0.05
    )
    print(format_comparison_output(
        results_test2, "prox_file1.beats", "prox_file2.beats",
        num_context_lines=1
    ))
    print("\n\n")
    
    # Test Case 3: All unique
    ts1_unique = [1.0, 2.0]
    bc1_unique = [1,1]
    ts2_unique = [3.0, 4.0]
    bc2_unique = [1,1]
    
    print(f"--- Test Case 3: All Unique Timestamps ---")
    s_ts1_unique = sorted(ts1_unique)
    s_ts2_unique = sorted(ts2_unique)
    results_test3 = compare_beats_data(s_ts1_unique, bc1_unique, s_ts2_unique, bc2_unique)
    print(format_comparison_output(results_test3, "uniqueA.beats", "uniqueB.beats", num_context_lines=1))
    print("\n\n")

    # Test Case 4: Empty lists
    print(f"--- Test Case 4: Empty Lists ---")
    results_test4 = compare_beats_data([], [], [], [])
    print(format_comparison_output(results_test4, "empty1.beats", "empty2.beats", num_context_lines=1))
    print("\n\n")

    # Test Case 5: One empty, one not
    ts1_empty_test = [1.0, 2.0]
    bc1_empty_test = [1,1]
    print(f"--- Test Case 5: One Empty List ---")
    s_ts1_empty = sorted(ts1_empty_test)
    results_test5 = compare_beats_data(s_ts1_empty, bc1_empty_test, [], [])
    print(format_comparison_output(results_test5, "data.beats", "empty.beats", num_context_lines=1))
    print("\n\n")
    
    # Test Case 6 from Spec
    ts1_spec = [1.01, 1.50]
    bc1_spec = [1,1]
    ts2_spec = [1.03, 1.65]
    bc2_spec = [1,1]
    print(f"--- Test Case 6: Spec Example ---")
    s_ts1_spec = sorted(ts1_spec)
    s_ts2_spec = sorted(ts2_spec)
    results_test6 = compare_beats_data(s_ts1_spec, bc1_spec, s_ts2_spec, bc2_spec, match_threshold=0.05)
    print(format_comparison_output(results_test6, "spec1.beats", "spec2.beats", num_context_lines=1))
    print("\nRaw results_test6:")
    import json
    print(json.dumps(results_test6, indent=4)) 