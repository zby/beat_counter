#!/usr/bin/env python3
"""
Experiment Comparison Script

This script compares the results of two beat detection experiments,
focusing on checking differences in the .beats files.
"""

import argparse
import logging
import sys
import json
from pathlib import Path

from beat_detection.utils.beats_compare import compare_beats_files, compare_beats_data, format_comparison_output


def main():
    """Main entry point for the script."""
    # Configure logging
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compare results from two beat detection experiments")
    parser.add_argument("exp1_dir", help="Path to the first experiment directory")
    parser.add_argument("exp2_dir", help="Path to the second experiment directory")
    parser.add_argument("--output", help="Path to save comparison results as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed differences")
    parser.add_argument("--summarize", "-s", action="store_true", help="Show summary statistics only")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit the number of array entries shown")
    parser.add_argument("--tolerance", "-t", type=float, default=0.1, 
                      help="Maximum time difference (in seconds) to consider two timestamps as matching")
    parser.add_argument("--smart-match", "-m", action="store_true", 
                      help="Use smart matching algorithm to find matching beats regardless of their order")
    
    args = parser.parse_args()
    
    try:
        exp1_dir = Path(args.exp1_dir)
        exp2_dir = Path(args.exp2_dir)
        
        print(f"Comparing experiments: {exp1_dir.name} vs {exp2_dir.name}")
        
        exp1_only, exp2_only, different_files = compare_beats_files(exp1_dir, exp2_dir, max_diff_sec=args.tolerance)
        
        # Print summary
        print(f"Files only in {exp1_dir.name}: {len(exp1_only)}")
        print(f"Files only in {exp2_dir.name}: {len(exp2_only)}")
        print(f"Files with differences: {len(different_files)}")
        
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
                
                # Handle timestamps differences
                if 'timestamps' in differences:
                    ts_diff = differences['timestamps']
                    if args.smart_match and 'matching' in ts_diff:
                        match_stats = ts_diff['matching']
                        file_summary.append(f"{match_stats['matched_count']} matched timestamps, "
                                           f"{match_stats['unmatched_count1']} only in {exp1_dir.name}, "
                                           f"{match_stats['unmatched_count2']} only in {exp2_dir.name}")
                    else:
                        count_diff = ts_diff.get('count_diff', 0)
                        
                        if count_diff == 0:
                            file_summary.append("same number of timestamps but values differ")
                        else:
                            direction = "more" if count_diff > 0 else "fewer"
                            file_summary.append(f"{abs(count_diff)} {direction} timestamps in {exp1_dir.name}")
                
                # Handle beat_counts differences
                if 'beat_counts' in differences:
                    bc_diff = differences['beat_counts']
                    count_diff = bc_diff.get('count_diff', 0)
                    
                    if count_diff == 0:
                        file_summary.append("same number of beat_counts but values differ")
                    else:
                        direction = "more" if count_diff > 0 else "fewer"
                        file_summary.append(f"{abs(count_diff)} {direction} beat_counts in {exp1_dir.name}")
                
                # Handle text diff
                if 'text_diff' in differences:
                    file_summary.append("text content differs")
                
                # Handle error
                if 'error' in differences:
                    file_summary.append(f"error: {differences['error']}")
                
                # Print the file summary
                print(f"  {file}: {', '.join(file_summary)}")
                
                # Print detailed differences if verbose
                if args.verbose:
                    if 'error' in differences:
                        print(f"    Error processing file {file}: {differences['error']}")
                        print("")
                    else: 
                        # This is the case for JSON files with data differences (not structural errors).
                        # Attempt to use the new detailed formatter.
                        try:
                            exp1_full_path = exp1_dir / file
                            exp2_full_path = exp2_dir / file

                            with open(exp1_full_path, 'r') as f1_content_file:
                                content1 = json.load(f1_content_file)
                            with open(exp2_full_path, 'r') as f2_content_file:
                                content2 = json.load(f2_content_file)

                            # Assuming keys exist if no 'error' was reported by compare_beats_files
                            timestamps1 = content1.get('timestamps', []) 
                            beat_counts1 = content1.get('beat_counts', [])
                            timestamps2 = content2.get('timestamps', [])
                            beat_counts2 = content2.get('beat_counts', [])

                            comparison_details = compare_beats_data(
                                timestamps1, beat_counts1,
                                timestamps2, beat_counts2,
                                match_threshold=args.tolerance
                            )

                            report_file1_name = f"{exp1_dir.name}/{file}"
                            report_file2_name = f"{exp2_dir.name}/{file}"
                            
                            formatted_report = format_comparison_output(
                                comparison_details,
                                file1_name=report_file1_name,
                                file2_name=report_file2_name
                            )
                            
                            # Print the formatted report, indented.
                            for report_line in formatted_report.splitlines():
                                print(f"    {report_line}")
                            print("")

                        except Exception as e:
                            print(f"    Failed to generate detailed comparison for {file} using format_comparison_output: {e}")
                            print(f"    Raw differences from initial scan for {file}:")
                            # Pretty print the differences dictionary
                            raw_diff_output = json.dumps(differences, indent=2)
                            for line in raw_diff_output.splitlines():
                                print(f"      {line}")
                            print("")
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            results = {
                "experiment1": str(exp1_dir),
                "experiment2": str(exp2_dir),
                "only_in_exp1": exp1_only,
                "only_in_exp2": exp2_only,
                "different_files": [{"file": f, "differences": d} for f, d in different_files]
            }
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to {output_path}")
            
        return 0 if not (exp1_only or exp2_only or different_files) else 1
        
    except Exception as e:
        logging.error(f"Comparison failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 