#!/usr/bin/env python3
"""
Beat detection tool.

This script detects beats in audio files and generates beat timestamp files.
These files can be used to create visualization videos.

Usage:
    detect-beats [options] [input_file_or_directory]

If input is not specified, processes all audio files in data/original/.
"""

import os
import sys
import pathlib
import argparse
import numpy as np

from beat_detection.core.detector import BeatDetector
from beat_detection.utils import file_utils, reporting
from beat_detection.utils.constants import AUDIO_EXTENSIONS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect beats in music files and generate beat timestamp files."
    )
    
    # Input arguments
    parser.add_argument(
        "input", nargs="?", 
        help="Input audio file or directory. If not provided, all files in data/original will be processed."
    )
    
    # Output directory
    parser.add_argument(
        "-o", "--output-dir", 
        default="data/beats",
        help="Output directory for beat files (default: data/beats)"
    )
    
    # Beat detection options
    parser.add_argument(
        "--min-bpm", type=int, default=60,
        help="Minimum BPM to detect (default: 60)"
    )
    parser.add_argument(
        "--max-bpm", type=int, default=240,
        help="Maximum BPM to detect (default: 240)"
    )
    parser.add_argument(
        "--tolerance", type=float, default=10.0,
        help="Percentage tolerance for beat intervals (default: 10.0)"
    )
    parser.add_argument(
        "--no-skip-intro", action="store_true",
        help="Don't attempt to detect and skip intro sections"
    )
    parser.add_argument(
        "--beats-per-bar", type=int, default=4,
        help="Number of beats per bar for downbeat detection (default: 4)"
    )
    
    # Output options
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )
    
    return parser.parse_args()


def process_audio_file(input_file, output_dir, detector, skip_intro=True, verbose=True):
    """
    Process a single audio file to detect beats.
    
    Parameters:
    -----------
    input_file : str or pathlib.Path
        Path to the input audio file
    output_dir : str or pathlib.Path
        Directory to save output files
    detector : BeatDetector
        Beat detector to use
    skip_intro : bool
        Whether to detect and skip intro sections
    verbose : bool
        Whether to print progress and statistics
        
    Returns:
    --------
    tuple
        (beat_statistics, irregular_beats)
    """
    # Ensure paths are Path objects
    input_path = pathlib.Path(input_file)
    base_output_dir = pathlib.Path(output_dir)
    
    # Preserve subdirectory structure if input is in a subdirectory
    if "data/original" in str(input_path):
        # Get relative path from data/original
        rel_path = input_path.relative_to(pathlib.Path("data/original"))
        # If there's a parent directory, use it in the output path
        if rel_path.parent != pathlib.Path("."):
            output_directory = file_utils.ensure_directory(base_output_dir / rel_path.parent)
        else:
            output_directory = file_utils.ensure_directory(base_output_dir)
    else:
        output_directory = file_utils.ensure_directory(base_output_dir)
    
    if verbose:
        print(f"\nProcessing: {input_path}")
        print("=" * 80)
    
    # Generate output file paths
    beats_file = file_utils.get_output_path(input_path, output_directory, suffix='_beats', ext='.txt')
    stats_file = file_utils.get_output_path(input_path, output_directory, suffix='_beat_stats', ext='.txt')
    
    # Detect beats and downbeats
    beat_timestamps, stats, irregular_beats, downbeats = detector.detect_beats(
        str(input_path), skip_intro=skip_intro
    )
    
    if verbose:
        reporting.print_beat_timestamps(beat_timestamps, irregular_beats, downbeats)
        reporting.print_statistics(stats, irregular_beats)
        print(f"\nDetected {len(downbeats)} downbeats")
    
    # Save beat timestamps and statistics
    reporting.save_beat_timestamps(beat_timestamps, beats_file, downbeats)
    reporting.save_beat_statistics(stats, irregular_beats, stats_file, 
                                  filename=input_path.name)
    
    # Save intervals for debugging
    if len(beat_timestamps) > 1:
        intervals = np.diff(beat_timestamps)
        intervals_file = file_utils.get_output_path(input_path, output_directory, suffix='_intervals', ext='.txt')
        with open(intervals_file, 'w') as f:
            f.write("# Beat intervals (seconds)\n")
            f.write("# Format: beat_number interval\n")
            for i, interval in enumerate(intervals):
                f.write(f"{i+1} {interval:.4f}\n")
        
        if verbose:
            print(f"Beat intervals saved as: {intervals_file}")
    
    if verbose:
        print(f"\nTotal number of beats detected: {len(beat_timestamps)}")
        print(f"Beat timestamps saved as: {beats_file}")
        print(f"Beat statistics saved as: {stats_file}")
        print("\nNext steps:")
        print(f"  Create visualization: generate-videos {beats_file}")
    
    return stats, irregular_beats


def process_directory(directory, output_dir, detector, extensions=None, 
                     skip_intro=True, verbose=True):
    """
    Process all audio files in a directory.
    
    Parameters:
    -----------
    directory : str or pathlib.Path
        Directory containing input audio files
    output_dir : str or pathlib.Path
        Directory to save output files
    detector : BeatDetector
        Beat detector to use
    extensions : list of str
        List of file extensions to process
    skip_intro : bool
        Whether to detect and skip intro sections
    verbose : bool
        Whether to print progress and statistics
        
    Returns:
    --------
    list
        List of tuples with filename and statistics for each processed file
    """
    # Find audio files
    audio_files = file_utils.find_audio_files(directory, extensions)
    
    # Process each file
    results = file_utils.batch_process(
        audio_files,
        process_func=process_audio_file,
        verbose=verbose,
        output_dir=output_dir,
        detector=detector,
        skip_intro=skip_intro
    )
    
    # Generate summary if multiple files were processed
    if len(results) > 1:
        summary_file = pathlib.Path(output_dir) / "batch_summary.txt"
        reporting.save_batch_summary(results, summary_file)
        
        if verbose:
            reporting.print_batch_summary(results)
            print(f"Summary statistics saved to: {summary_file}")
    
    return results


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Configure detector based on args
    detector = BeatDetector(
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        tolerance_percent=args.tolerance,
        beats_per_bar=args.beats_per_bar
    )
    
    # Determine output path
    output_dir = pathlib.Path(args.output_dir)
    
    # Process input
    if args.input:
        file_utils.process_input_path(
            args.input,
            default_dir="data/original",
            process_file_func=process_audio_file,
            process_dir_func=process_directory,
            output_dir=output_dir,
            detector=detector,
            skip_intro=not args.no_skip_intro,
            verbose=not args.quiet
        )
    else:
        # Default: process all files in data/original
        process_directory(
            "data/original",
            output_dir,
            detector,
            skip_intro=not args.no_skip_intro,
            verbose=not args.quiet
        )


if __name__ == "__main__":
    main()