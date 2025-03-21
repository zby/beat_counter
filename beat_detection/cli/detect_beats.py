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
from tqdm import tqdm

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
        help="Input audio file or directory. If not provided, all files in data/input will be processed."
    )
    
    # Output directory
    parser.add_argument(
        "-o", "--output-dir", 
        default="data/output",
        help="Output directory for beat files (default: data/output)"
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
        "--no-skip-ending", action="store_true",
        help="Don't attempt to detect and skip ending sections"
    )
    parser.add_argument(
        "--beats-per-bar", type=int, default=None,
        help="Number of beats per bar for downbeat detection (default: None, will try all supported meters)"
    )
    
    # Output options
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Don't show progress bar"
    )
    
    return parser.parse_args()


def process_audio_file(input_file, output_dir=None, detector=None, skip_intro=True, skip_ending=True, verbose=True,
                    input_base_dir="data/input", output_base_dir="data/output", show_progress=True):
    """
    Process a single audio file to detect beats.
    
    Parameters:
    -----------
    input_file : str or pathlib.Path
        Path to the input audio file
    output_dir : str or pathlib.Path, optional
        Directory to save output files
    detector : BeatDetector
        Beat detector to use
    skip_intro : bool
        Whether to detect and skip intro sections
    skip_ending : bool
        Whether to detect and skip ending sections
    verbose : bool
        Whether to print progress and statistics
    input_base_dir : str
        Base input directory (default: "data/input")
    output_base_dir : str
        Base output directory (default: "data/output")
        
    Returns:
    --------
    tuple
        (beat_statistics, irregular_beats)
    """
    # Ensure paths are Path objects
    input_path = pathlib.Path(input_file)
    
    # Determine output directory
    if output_dir is not None:
        output_directory = file_utils.get_output_directory(input_path, input_base_dir=input_base_dir, output_base_dir=output_dir)
    else:
        output_directory = file_utils.get_output_directory(input_path, input_base_dir, output_base_dir)
    
    # Generate output file paths
    beats_file = file_utils.get_output_path(input_path, suffix='_beats', ext='.txt')
    stats_file = file_utils.get_output_path(input_path, suffix='_beat_stats', ext='.txt')
    
    # Setup progress bar
    progress_bar = None
    progress_callback = None
    
    if show_progress:
        progress_bar = tqdm(total=100, desc="Detecting beats", bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        def update_progress(status, progress):
            progress_bar.set_description(status)
            # Convert progress from 0-1 to 0-100 and calculate the difference to update
            new_progress = int(progress * 100)
            progress_bar.update(new_progress - progress_bar.n)
        
        progress_callback = update_progress
    
    # Detect beats and downbeats
    beat_timestamps, stats, irregular_beats, downbeats, intro_end_idx, ending_start_idx, detected_meter = detector.detect_beats(
        str(input_path), skip_intro=skip_intro, skip_ending=skip_ending, progress_callback=progress_callback
    )
    
    # Close progress bar
    if progress_bar:
        progress_bar.close()
    
    if verbose:
        reporting.print_beat_timestamps(beat_timestamps, irregular_beats, downbeats)
        reporting.print_statistics(stats, irregular_beats)
        print(f"\nDetected {len(downbeats)} downbeats")
        print(f"Detected meter: {detected_meter}/4 time signature")
        if intro_end_idx > 0:
            print(f"Detected intro section: skipped first {intro_end_idx} beats")
        if ending_start_idx < len(beat_timestamps):
            print(f"Detected ending section: skipped last {len(beat_timestamps) - ending_start_idx} beats")
    
    # Save beat timestamps and statistics
    reporting.save_beat_timestamps(beat_timestamps, beats_file, downbeats, 
                                intro_end_idx=intro_end_idx, ending_start_idx=ending_start_idx,
                                detected_meter=detected_meter)
    reporting.save_beat_statistics(stats, irregular_beats, stats_file, 
                                  filename=input_path.name)
    
    # Save intervals for debugging
    if len(beat_timestamps) > 1:
        intervals = np.diff(beat_timestamps)
        intervals_file = file_utils.get_output_path(input_path, suffix='_intervals', ext='.txt')
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
                     skip_intro=True, skip_ending=True, verbose=True, show_progress=True):
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
    
    if not audio_files:
        if verbose:
            print(f"No audio files found in {directory}")
        return []
    
    if verbose:
        print(f"Found {len(audio_files)} audio files in {directory}")
    
    # Process each file
    results = []
    
    for audio_file in audio_files:
        try:
            if verbose:
                print(f"\nProcessing: {audio_file}")
                print("=" * 80)
            
            stats, irregular_beats = process_audio_file(
                audio_file,
                output_dir=output_dir,
                detector=detector,
                skip_intro=skip_intro,
                skip_ending=skip_ending,
                verbose=verbose,
                show_progress=show_progress
            )
            results.append((audio_file.name, (stats, irregular_beats)))
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            results.append((audio_file.name, None))
    
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
    
    # Determine whether to skip intro and ending
    skip_intro = not args.no_skip_intro
    skip_ending = not args.no_skip_ending
    
    # Determine base input and output directories
    input_base_dir = "data/input"
    output_base_dir = pathlib.Path(args.output_dir)
    
    # Get input paths to process
    input_path = args.input if args.input else input_base_dir
    
    # Find audio files to process
    audio_files = file_utils.find_audio_files(input_path)
    
    if not audio_files:
        print(f"No audio files found in {input_path}")
        return
    
    if not args.quiet:
        print(f"Found {len(audio_files)} audio files to process")
    
    # Process all audio files
    results = []
    
    for audio_file in audio_files:
        try:
            if not args.quiet:
                print(f"\nProcessing: {audio_file}")
                print("=" * 80)
            
            stats, irregular_beats = process_audio_file(
                audio_file,
                output_dir=output_base_dir,
                detector=detector,
                skip_intro=skip_intro,
                skip_ending=skip_ending,
                verbose=not args.quiet,
                input_base_dir=input_base_dir,
                output_base_dir=output_base_dir,
                show_progress=not args.no_progress
            )
            results.append((audio_file.name, (stats, irregular_beats)))
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            results.append((audio_file.name, None))
    
    # Generate summary if multiple files were processed
    if len(results) > 1 and not args.quiet:
        summary_file = output_base_dir / "batch_summary.txt"
        reporting.save_batch_summary(results, summary_file)
        reporting.print_batch_summary(results)
        print(f"Summary statistics saved to: {summary_file}")


if __name__ == "__main__":
    main()