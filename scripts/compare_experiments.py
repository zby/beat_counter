#!/usr/bin/env python3
"""
Experiment and File Comparison Script

This script compares the results of two beat detection experiments (directories)
or two individual .beats files.
"""

import argparse
import logging
import sys
import json
from pathlib import Path

from beat_counter.utils.beats_compare import compare_beats_files, compare_beats_data, format_comparison_output


def _determine_if_details_indicate_difference(comparison_details: dict) -> bool:
    """
    Determines if the comparison_details from compare_beats_data indicate any difference.
    This is a helper to set the exit code.
    """
    # Check beats per bar if it exists
    if comparison_details.get("beats_per_bar_diff", False):
        return True
        
    # Check beat counts
    beat_summary = comparison_details.get("beat_counts_summary", {})
    if beat_summary.get("status") != "match":
        return True

    # Check timestamps differences based on summary_stats
    summary_stats = comparison_details.get("summary_stats", {})
    if summary_stats.get("unique_to_file1", 0) > 0 or \
       summary_stats.get("unique_to_file2", 0) > 0:
        return True
    
    # The unique_to_file1/2 checks should cover mismatches where counts are the same
    # but timestamps don't align. If all timestamps match, unique counts will be 0.
    # If counts are the same but timestamps differ such that some are unique to one file
    # and some to another (offsetting each other in total count), unique_to_file1/2
    # will capture this.

    # If timestamps perfectly align but values differ slightly (within match_threshold),
    # they are still 'match' type in timestamps_diff. The 'max_match_diff_ms'
    # in summary_stats would show this, but typically we consider these "matching"
    # for the purpose of "are the files different in terms of beat presence/absence".
    # If a stricter definition of "different" is needed (e.g., any numeric diff in matched pairs),
    # one might check summary_stats["max_match_diff_ms"] != 0.
    # For now, focus on structural differences (presence/absence of beats).

    # Check for internal proximity errors
    # These are warnings about data quality within a single file, but for the purpose
    # of this script, they don't make the *two files* different from each other.
    # If desired, uncomment the following to consider proximity errors as a difference:
    # if comparison_details.get("internal_proximity_errors"): # Non-empty list is true
    #     return True
        
    return False


def _compare_two_files(file1_path: Path, file2_path: Path, tolerance: float, limit: int = None, output_path: Path = None) -> int:
    """
    Compare two individual beat files and return the exit code.
    
    Args:
        file1_path: Path to the first file to compare
        file2_path: Path to the second file to compare
        tolerance: Maximum time difference (in seconds) to consider two timestamps as matching
        limit: Optional limit for the number of array entries shown in detailed reports
        output_path: Optional path to save comparison results as JSON
        
    Returns:
        int: Exit code (0 if no differences, 1 if differences found or errors occurred)
    """
    try:
        with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}", file=sys.stderr)
        return 1

    timestamps1 = data1.get('timestamps', [])
    beat_counts1 = data1.get('beat_counts', [])
    timestamps2 = data2.get('timestamps', [])
    beat_counts2 = data2.get('beat_counts', [])
    
    # Get beats per bar information if it exists
    beats_per_bar1 = data1.get('beats_per_bar')
    beats_per_bar2 = data2.get('beats_per_bar')
    
    comparison_details = compare_beats_data(
        timestamps1, beat_counts1,
        timestamps2, beat_counts2,
        match_threshold=tolerance
    )
    
    # Add beats per bar comparison
    if beats_per_bar1 is not None or beats_per_bar2 is not None:
        if beats_per_bar1 != beats_per_bar2:
            comparison_details["beats_per_bar_diff"] = {
                "file1_value": beats_per_bar1,
                "file2_value": beats_per_bar2
            }
    
    report_options = {
        "file1_name": str(file1_path),
        "file2_name": str(file2_path),
    }
    if limit is not None:
        report_options["limit"] = limit

    formatted_report = format_comparison_output(
        comparison_details,
        num_context_lines=2,
        **report_options
    )
    
    # Display beats per bar difference prominently if it exists
    if comparison_details.get("beats_per_bar_diff"):
        beats_diff = comparison_details["beats_per_bar_diff"]
        print("\n⚠️ IMPORTANT DIFFERENCE: Beats per bar mismatch ⚠️")
        print(f"  File 1 ({file1_path.name}): {beats_diff['file1_value']}")
        print(f"  File 2 ({file2_path.name}): {beats_diff['file2_value']}")
        print("")
    
    print("\nComparison Report:")
    for report_line in formatted_report.splitlines():
        print(f"  {report_line}")
    print("")

    if output_path:
        results = {
            "file1": str(file1_path),
            "file2": str(file2_path),
            "tolerance_seconds": tolerance,
            "comparison_details": comparison_details 
        }
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Comparison details saved to {output_path}")

    # Determine exit code based on whether differences were found
    are_different = _determine_if_details_indicate_difference(comparison_details)
    return 1 if are_different else 0


def _compare_directories(path1: Path, path2: Path, tolerance: float, summarize: bool, verbose: bool, limit: int, use_percentages: bool, output_path: Path):
    """
    Lightweight directory comparison helper.
    """
    exp1_dir = path1
    exp2_dir = path2

    print(f"Comparing experiment directories: {exp1_dir.name} vs {exp2_dir.name}")

    # Collect file sets using the existing utility
    exp1_only, exp2_only, different_files = compare_beats_files(
        exp1_dir, exp2_dir, max_diff_sec=tolerance
    )

    # Calculate identical files by finding common files that aren't in different_files
    all_files1 = set(f.relative_to(exp1_dir) for f in exp1_dir.rglob("*.beats"))
    all_files2 = set(f.relative_to(exp2_dir) for f in exp2_dir.rglob("*.beats"))
    common_files = all_files1.intersection(all_files2)
    different_file_paths = {Path(p) for p, _ in different_files}
    identical_files = common_files - different_file_paths

    # --- Summary Section ----------------------------------------------------
    print(f"Files only in {exp1_dir.name}: {len(exp1_only)}")
    print(f"Files only in {exp2_dir.name}: {len(exp2_only)}")
    print(f"Files with differences: {len(different_files)}")
    print(f"Identical files: {len(identical_files)}")

    # Early-exit behaviour for --summarize -----------------------------------
    if summarize and not verbose:
        return 1 if (exp1_only or exp2_only or different_files) else 0

    # --- Non-summarised listings -------------------------------------------
    if exp1_only:
        print(f"\nFiles only in {exp1_dir.name}:")
        for rel_path in exp1_only:
            print(f"  {rel_path}")

    if exp2_only:
        print(f"\nFiles only in {exp2_dir.name}:")
        for rel_path in exp2_only:
            print(f"  {rel_path}")

    if different_files:
        print(f"\nFiles with differences:")
        for rel_path, diff in different_files:
            summary = _summarize_diff(diff, exp1_dir.name, exp2_dir.name, use_percentages=use_percentages)
            print(f"  {rel_path}: {summary}")

    # --- Verbose section: full diff for each differing file -----------------
    if verbose and different_files:
        print("\nDetailed differences (verbose):")
        for rel_path, _ in different_files:
            print("\n" + "=" * 80)
            print(f"Diff for {rel_path}")
            _compare_two_files(
                exp1_dir / rel_path,
                exp2_dir / rel_path,
                tolerance,
                limit,
                None,
            )

    # --- Optional JSON output ----------------------------------------------
    if output_path:
        # We only care about paths; strip differences content for brevity
        results = {
            "experiment1_path": str(exp1_dir),
            "experiment2_path": str(exp2_dir),
            "tolerance_seconds": tolerance,
            "only_in_path1": exp1_only,
            "only_in_path2": exp2_only,
            "different_files": [p for p, _ in different_files],
            "identical_files_count": len(identical_files),
        }
        with open(output_path, "w") as fp:
            json.dump(results, fp, indent=2)
        print(f"\nResults summary saved to {output_path}")

    # Non-zero exit if anything differs -------------------------------------
    return 1 if (exp1_only or exp2_only or different_files) else 0


def _summarize_diff(differences: dict, exp1_label: str, exp2_label: str, *, use_percentages: bool = False) -> str:
    """Return a concise human-readable summary string for a single file diff.

    When use_percentages=True, numeric counts are converted to percentages based on the
    larger of the two array lengths involved.
    """
    def _fmt(value: int, denom: int) -> str:
        if not use_percentages or denom == 0:
            return str(value)
        return f"{(value/denom*100):.1f}%"

    parts = []
    
    # --- beats per bar (highest priority) ---------------------------------
    beats_per_bar_diff = differences.get("beats_per_bar_diff")
    if beats_per_bar_diff:
        file1_value = beats_per_bar_diff.get("file1_value")
        file2_value = beats_per_bar_diff.get("file2_value")
        parts.append(f"BEATS PER BAR MISMATCH: {file1_value} vs {file2_value}")

    # --- timestamps -------------------------------------------------------
    ts_diff = differences.get("timestamps")
    has_ts_diff = ts_diff is not None
    if has_ts_diff:
        count_diff = ts_diff.get("count_diff", 0)

        # Count approximate matches (diff_ms != 0)
        match_info = ts_diff.get("matching", {})
        matches_list = match_info.get("matches", [])
        approx_matches = sum(1 for m in matches_list if abs(m.get("diff_ms", 0)) > 0.001)

        denom_ts = max(ts_diff.get("first_count", 0), ts_diff.get("second_count", 0))

        if count_diff:
            direction = "more" if count_diff > 0 else "fewer"
            parts.append(f"{_fmt(abs(count_diff), denom_ts)} {direction} timestamps in {exp1_label}")

        if approx_matches:
            parts.append(f"{_fmt(approx_matches, denom_ts)} approx matches")

    # --- beat counts ------------------------------------------------------
    bc_diff = differences.get("beat_counts")
    # Only include beat-count info when there is no timestamp difference
    if bc_diff is not None and not has_ts_diff:
        count_diff = bc_diff.get("count_diff", 0)
        denom_bc = max(bc_diff.get("first_count", 0), bc_diff.get("second_count", 0))
        if count_diff:
            direction = "more" if count_diff > 0 else "fewer"
            parts.append(f"{_fmt(abs(count_diff), denom_bc)} {direction} beat counts in {exp1_label}")
        else:
            parts.append("beat count values differ")

    # Fallback
    if not parts:
        parts.append("differences found")

    return ", ".join(parts)


def main():
    """Main entry point for the script."""
    # Configure logging
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compare results from two beat detection experiments (directories) or two .beats files.")
    parser.add_argument("path1", help="Path to the first experiment directory or .beats file")
    parser.add_argument("path2", help="Path to the second experiment directory or .beats file")
    parser.add_argument("selected_file", nargs="?", help="When comparing directories, only compare this specific file")
    parser.add_argument("--output", help="Path to save comparison results as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed differences (primarily for directory mode)")
    parser.add_argument("--summarize", "-s", action="store_true", help="Show summary statistics only (directory mode)")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit the number of array entries shown in detailed reports")
    parser.add_argument("--tolerance", "-t", type=float, default=0.1, 
                      help="Maximum time difference (in seconds) to consider two timestamps as matching")
    # By default we display percentages; use --absolute to revert to absolute counts
    parser.add_argument("--absolute", "-a", action="store_true", help="Show absolute numbers instead of percentages in directory summaries (default is percentage)")
    
    args = parser.parse_args()
    
    try:
        path1 = Path(args.path1)
        path2 = Path(args.path2)

        if not path1.exists():
            print(f"Error: Path does not exist: {path1}", file=sys.stderr)
            return 1
        if not path2.exists():
            print(f"Error: Path does not exist: {path2}", file=sys.stderr)
            return 1

        # Mode 1: Comparing two individual files
        if path1.is_file() and path2.is_file():
            print(f"Comparing files: {path1.name} vs {path2.name}")
            if args.summarize:
                print("Warning: --summarize is ignored when comparing individual files.", file=sys.stderr)
            
            output_path = Path(args.output) if args.output else None
            return _compare_two_files(path1, path2, args.tolerance, args.limit, output_path)
            
        # Mode 3: Comparing a specific file from two directories
        elif path1.is_dir() and path2.is_dir() and args.selected_file:
            file1_path = path1 / args.selected_file
            file2_path = path2 / args.selected_file
            
            if not file1_path.exists():
                print(f"Error: File does not exist: {file1_path}", file=sys.stderr)
                return 1
            if not file2_path.exists():
                print(f"Error: File does not exist: {file2_path}", file=sys.stderr)
                return 1
                
            print(f"Comparing file '{args.selected_file}' from directories: {path1.name} vs {path2.name}")
            
            output_path = Path(args.output) if args.output else None
            return _compare_two_files(file1_path, file2_path, args.tolerance, args.limit, output_path)

        # Mode 2: Comparing two directories
        elif path1.is_dir() and path2.is_dir():
            # Delegate directory logic to a helper for compactness
            return _compare_directories(
                path1,
                path2,
                tolerance=args.tolerance,
                summarize=args.summarize,
                verbose=args.verbose,
                limit=args.limit,
                use_percentages=not args.absolute,
                output_path=Path(args.output) if args.output else None,
            )

        else:
            # Paths are mixed (one file, one directory) or invalid type
            print("Error: Both paths must be files or both paths must be directories.", file=sys.stderr)
            if not path1.is_file() and not path1.is_dir():
                 print(f"Error: Path1 '{path1}' is neither a file nor a directory.", file=sys.stderr)
            if not path2.is_file() and not path2.is_dir():
                 print(f"Error: Path2 '{path2}' is neither a file nor a directory.", file=sys.stderr)
            return 1
        
    except Exception as e:
        # Use logging for unexpected errors if it's configured, otherwise print
        # logging.error(f"Comparison failed with an unexpected error: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        # Consider if more detailed stack trace is useful for users or only for dev
        # import traceback
        # traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Basic logging configuration (can be enhanced)
    # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    sys.exit(main()) 