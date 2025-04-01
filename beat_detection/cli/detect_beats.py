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
import logging
from pathlib import Path
from typing import Optional, List, Tuple

from beat_detection.core.detector import BeatDetector, MadmomBeatProcessor
from beat_detection.core.video import BeatVideoGenerator
from beat_detection.utils import file_utils, reporting
from beat_detection.utils.constants import AUDIO_EXTENSIONS
from beat_detection.utils.file_utils import get_audio_files, ensure_output_dir
from beat_detection.utils.beat_file import save_beats


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


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


def process_file(
    audio_file: str,
    output_dir: str,
    detector: BeatDetector,
    video_generator: Optional[BeatVideoGenerator] = None,
    verbose: bool = False
) -> bool:
    """
    Process a single audio file.
    
    Parameters:
    -----------
    audio_file : str
        Path to input audio file
    output_dir : str
        Path to output directory
    detector : BeatDetector
        Beat detector instance
    video_generator : Optional[BeatVideoGenerator]
        Optional video generator instance
    verbose : bool
        Whether to print verbose output
        
    Returns:
    --------
    bool
        True if processing was successful
    """
    try:
        # Create output directory if it doesn't exist
        ensure_output_dir(output_dir)
        
        # Get base filename without extension
        base_name = Path(audio_file).stem
        
        # Detect beats
        beats = detector.detect_beats(audio_file)
        
        # Save beat data
        beat_file = os.path.join(output_dir, f"{base_name}.beats")
        save_beats(beat_file, beats)
        
        if verbose:
            logging.info(f"Saved beat data to {beat_file}")
            
        # Generate video if requested
        if video_generator:
            video_file = os.path.join(output_dir, f"{base_name}.mp4")
            video_generator.generate_video(beats, video_file)
            
            if verbose:
                logging.info(f"Generated video at {video_file}")
                
        return True
        
    except Exception as e:
        logging.error(f"Error processing {audio_file}: {str(e)}")
        return False


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Configure processor and detector based on args
    processor = MadmomBeatProcessor(
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        fps=args.fps
    )
    
    detector = BeatDetector(
        beat_processor=processor.process_beats,
        downbeat_processor=processor.process_downbeats,
        beat_tracker=processor.track_beats,
        downbeat_tracker=processor.track_downbeats,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        fps=args.fps,
        tolerance_percent=args.tolerance,
        beats_per_bar=args.beats_per_bar,
        skip_intro=not args.no_skip_intro,
        skip_ending=not args.no_skip_ending
    )
    
    # Configure video generator if requested
    video_generator = None
    if args.generate_video:
        video_generator = BeatVideoGenerator(fps=args.fps)
    
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
            
            success = process_file(
                str(audio_file),
                str(output_base_dir),
                detector,
                video_generator,
                not args.quiet
            )
            if success:
                results.append((audio_file.name, success))
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            results.append((audio_file.name, False))
    
    # Generate summary if multiple files were processed
    if len(results) > 1 and not args.quiet:
        summary_file = output_base_dir / "batch_summary.txt"
        reporting.save_batch_summary(results, summary_file)
        reporting.print_batch_summary(results)
        print(f"Summary statistics saved to: {summary_file}")


if __name__ == "__main__":
    setup_logging()
    exit(main())