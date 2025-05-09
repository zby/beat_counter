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

from beat_detection.utils.beats_compare import compare_beats_files, compare_beats_data, format_comparison_output


def _determine_if_details_indicate_difference(comparison_details: dict) -> bool:
    """
    Determines if the comparison_details from compare_beats_data indicate any difference.
    This is a helper to set the exit code.
    """
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

    comparison_details = compare_beats_data(
        timestamps1, beat_counts1,
        timestamps2, beat_counts2,
        match_threshold=tolerance
    )
    
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
            exp1_dir = path1
            exp2_dir = path2
            print(f"Comparing experiment directories: {exp1_dir.name} vs {exp2_dir.name}")
            
            exp1_only, exp2_only, different_files = compare_beats_files(
                exp1_dir, exp2_dir, 
                max_diff_sec=args.tolerance
            )
            
            # Print summary
            print(f"Files only in {exp1_dir.name}: {len(exp1_only)}")
            print(f"Files only in {exp2_dir.name}: {len(exp2_only)}")
            print(f"Files with differences: {len(different_files)}")
            
            if args.summarize and not args.verbose:
                 if exp1_only or exp2_only or different_files:
                    return 1 # Differences exist
                 else:
                    return 0 # No differences

            # Print details
            if exp1_only:
                print(f"\nFiles only in {exp1_dir.name}:")
                for file in exp1_only:
                    print(f"  {file}")
                    
            if exp2_only:
                print(f"\nFiles only in {exp2_dir.name}:")
                for file in exp2_only:
                    print(f"  {file}")
                    
            if different_files:
                print(f"\nFiles with differences:")
                for file, differences in different_files:
                    file_summary = []
                    
                    if 'timestamps' in differences:
                        ts_diff = differences['timestamps']
                        # Standard diff interpretation
                        count_diff = ts_diff.get('count_diff', 0)
                        if count_diff == 0 and ts_diff.get('values_differ', False): # Assuming a 'values_differ' flag
                            file_summary.append("same number of timestamps but values differ")
                        elif count_diff != 0 and 'matching' not in ts_diff:
                            # Only add this if we don't have more detailed matching info
                            direction = "more" if count_diff > 0 else "fewer" # count_diff > 0 means file1 has more
                            file_summary.append(f"{abs(count_diff)} {direction} timestamps in {exp1_dir.name}")
                        elif ts_diff.get('values_differ', False): # If count_diff is 0 but values differ
                            file_summary.append("timestamps values differ")

                        # Extract matching information if available from the smart matching
                        if 'matching' in ts_diff:
                            match_info = ts_diff['matching']
                            match_count = match_info.get('matched_count', 0)
                            # Clarify that these are timestamps that match within the tolerance threshold
                            file_summary.append(f"{match_count} timestamps matched within tolerance")
                            
                            # Add info about unmatched timestamps, but avoid redundancy
                            unmatched1 = match_info.get('unmatched_count1', 0)
                            unmatched2 = match_info.get('unmatched_count2', 0)
                            if unmatched1 > 0:
                                file_summary.append(f"{unmatched1} timestamps only in {exp1_dir.name}")
                            if unmatched2 > 0:
                                file_summary.append(f"{unmatched2} timestamps only in {exp2_dir.name}")
                            # Remove the count_diff information since it's redundant with unmatched counts
                            # (unmatched2 - unmatched1 = count_diff)

                    if 'beat_counts' in differences:
                        bc_diff = differences['beat_counts']
                        count_diff = bc_diff.get('count_diff', 0)
                        if count_diff == 0 and bc_diff.get('values_differ', False):
                             file_summary.append("same number of beat_counts but values differ")
                        elif count_diff != 0:
                            direction = "more" if count_diff > 0 else "fewer"
                            file_summary.append(f"{abs(count_diff)} {direction} beat_counts in {exp1_dir.name}")
                        elif bc_diff.get('values_differ', False):
                            file_summary.append("beat_counts values differ")
                    
                    if 'text_diff' in differences:
                        file_summary.append("text content differs")
                    
                    if 'error' in differences:
                        file_summary.append(f"error: {differences['error']}")
                    
                    # If we don't have a summary yet, but we're in verbose mode, we'll get detailed info below
                    # For non-verbose mode, generate a more specific summary by re-running the comparison
                    if not file_summary and not args.verbose:
                        try:
                            exp1_full_path = exp1_dir / file
                            exp2_full_path = exp2_dir / file

                            # Quickly load and compare to get exact/proximate match counts
                            with open(exp1_full_path, 'r') as f1_content_file:
                                content1 = json.load(f1_content_file)
                            with open(exp2_full_path, 'r') as f2_content_file:
                                content2 = json.load(f2_content_file)

                            timestamps1_v = content1.get('timestamps', []) 
                            beat_counts1_v = content1.get('beat_counts', [])
                            timestamps2_v = content2.get('timestamps', [])
                            beat_counts2_v = content2.get('beat_counts', [])

                            detailed_comp = compare_beats_data(
                                timestamps1_v, beat_counts1_v,
                                timestamps2_v, beat_counts2_v,
                                match_threshold=args.tolerance
                            )
                            
                            # Extract exact and proximate match counts
                            stats = detailed_comp.get("summary_stats", {})
                            exact_matches = stats.get("exact_matches", 0)
                            proximate_matches = stats.get("proximate_matches", 0)
                            unique_to_file1 = stats.get("unique_to_file1", 0)
                            unique_to_file2 = stats.get("unique_to_file2", 0)
                            
                            # Add match information, avoiding redundancy
                            total_matches = exact_matches + proximate_matches
                            if total_matches > 0:
                                if exact_matches > 0 and proximate_matches > 0:
                                    file_summary.append(f"{exact_matches} exact + {proximate_matches} proximate matches")
                                elif exact_matches > 0:
                                    file_summary.append(f"{exact_matches} exact matches")
                                elif proximate_matches > 0:
                                    file_summary.append(f"{proximate_matches} proximate matches")
                            
                            # Add unique timestamps information
                            if unique_to_file1 > 0:
                                file_summary.append(f"{unique_to_file1} timestamps only in {exp1_dir.name}")
                            if unique_to_file2 > 0:
                                file_summary.append(f"{unique_to_file2} timestamps only in {exp2_dir.name}")
                            
                            # Also check the beat counts status
                            beat_summary = detailed_comp.get("beat_counts_summary", {})
                            if beat_summary.get("status") == "length_mismatch":
                                bc_len1 = beat_summary.get("details", {}).get("len1", 0)
                                bc_len2 = beat_summary.get("details", {}).get("len2", 0)
                                if bc_len1 != bc_len2:
                                    diff = abs(bc_len1 - bc_len2)
                                    which_greater = f"{exp1_dir.name}" if bc_len1 > bc_len2 else f"{exp2_dir.name}"
                                    file_summary.append(f"{diff} more beat counts in {which_greater}")
                            elif beat_summary.get("status") == "content_mismatch":
                                file_summary.append("beat counts have same length but different content")
                                
                        except Exception as e:
                            # If there's any issue, fall back to generic message
                            file_summary.append("differences found")
                    
                    print(f"  {file}: {', '.join(file_summary) if file_summary else 'differences found (reasons unknown)'}")
                    
                    if args.verbose:
                        if 'error' in differences:
                            print(f"    Error processing file {file}: {differences['error']}\n")
                        else: 
                            try:
                                exp1_full_path = exp1_dir / file
                                exp2_full_path = exp2_dir / file

                                # This detailed comparison should ideally come from compare_beats_files
                                # or be regenerated consistently.
                                # For now, re-calculating as per original script logic for verbose mode.
                                with open(exp1_full_path, 'r') as f1_content_file:
                                    content1 = json.load(f1_content_file)
                                with open(exp2_full_path, 'r') as f2_content_file:
                                    content2 = json.load(f2_content_file)

                                timestamps1_v = content1.get('timestamps', []) 
                                beat_counts1_v = content1.get('beat_counts', [])
                                timestamps2_v = content2.get('timestamps', [])
                                beat_counts2_v = content2.get('beat_counts', [])

                                detailed_comp = compare_beats_data(
                                    timestamps1_v, beat_counts1_v,
                                    timestamps2_v, beat_counts2_v,
                                    match_threshold=args.tolerance
                                )

                                report_file1_name = f"{exp1_dir.name}/{file}"
                                report_file2_name = f"{exp2_dir.name}/{file}"
                                
                                report_options_verbose = {
                                    "file1_name": report_file1_name,
                                    "file2_name": report_file2_name,
                                }
                                if args.limit is not None:
                                    report_options_verbose["limit"] = args.limit

                                formatted_report_verbose = format_comparison_output(
                                    detailed_comp,
                                    num_context_lines=2,
                                    **report_options_verbose
                                )
                                
                                for report_line in formatted_report_verbose.splitlines():
                                    print(f"    {report_line}")
                                print("")

                            except Exception as e:
                                print(f"    Failed to generate detailed comparison for {file}: {e}")
                                print(f"    Raw differences from initial scan for {file}:")
                                raw_diff_output = json.dumps(differences, indent=2)
                                for line in raw_diff_output.splitlines():
                                    print(f"      {line}")
                                print("")
            
            if args.output:
                output_path = Path(args.output)
                # Ensure 'differences' in different_files is serializable
                serializable_diff_files = []
                for f, d in different_files:
                    # Basic check, can be more sophisticated if d contains non-serializable types
                    try:
                        json.dumps(d) # Test serializability
                        serializable_diff_files.append({"file": f, "differences": d})
                    except TypeError:
                        serializable_diff_files.append({"file": f, "differences": str(d)}) # Fallback to string

                results = {
                    "experiment1_path": str(exp1_dir),
                    "experiment2_path": str(exp2_dir),
                    "tolerance_seconds": args.tolerance,
                    "only_in_path1": exp1_only,
                    "only_in_path2": exp2_only,
                    "different_files_summary": serializable_diff_files
                }
                
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"\nResults summary saved to {output_path}")
            
            return 0 if not (exp1_only or exp2_only or different_files) else 1
            
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