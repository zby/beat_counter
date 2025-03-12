#!/usr/bin/env python3
"""
Generate beat visualization videos from beat timestamp files.

This script takes beat timestamp files (created by detect-beats) and creates
visual representations of the beats as videos.

Usage:
    generate-videos [options] [beats_file_or_directory]

If beats_file is provided, processes that specific file.
If directory is provided, processes all beat files in that directory.
If nothing is provided, processes all beat files in data/beats/.
"""

import os
import sys
import pathlib
import argparse
import numpy as np
from typing import List, Tuple, Optional, Union

from beat_detection.core.video import BeatVideoGenerator
from beat_detection.utils import file_utils, reporting
from beat_detection.utils.constants import AUDIO_EXTENSIONS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate beat visualization videos from beat timestamp files."
    )
    
    # Input arguments
    parser.add_argument(
        "input", nargs="?", 
        help="Input beats.txt file or directory containing beat files. If not provided, all files in data/beats will be processed."
    )
    
    # Output directory
    parser.add_argument(
        "-o", "--output-dir", 
        help="Output directory for videos (default: same as input)"
    )
    
    # Audio file
    parser.add_argument(
        "-a", "--audio-file",
        help="Path to audio file (default: auto-detected based on beat filename)"
    )
    
    # Audio directory
    parser.add_argument(
        "--audio-dir", 
        default="data/original",
        help="Directory containing original audio files (default: data/original)"
    )
    
    # Video options
    parser.add_argument(
        "--meter", type=int, default=4,
        help="Number of beats per measure (time signature numerator, default: 4)"
    )
    parser.add_argument(
        "--resolution", type=str, default="1280x720",
        help="Video resolution in format WIDTHxHEIGHT (default: 1280x720)"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second for output videos (default: 30)"
    )
    parser.add_argument(
        "--flash-duration", type=float, default=0.1,
        help="Duration of flash effect in seconds (default: 0.1)"
    )
    
    # Video type options
    parser.add_argument(
        "--flash", action="store_true",
        help="Generate flash video (disabled by default)"
    )
    parser.add_argument(
        "--no-counter", action="store_true",
        help="Skip generating counter video"
    )
    
    # Output options
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )
    
    return parser.parse_args()


def load_beat_data(beat_file: pathlib.Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load beat timestamps and downbeat information from a file.
    
    Parameters:
    -----------
    beat_file : pathlib.Path
        Path to the beat timestamp file
    
    Returns:
    --------
    tuple
        (beat_timestamps, downbeats) where:
        - beat_timestamps: Array of beat timestamps in seconds
        - downbeats: Array of indices that correspond to downbeats, or None if not available
    """
    try:
        # First load the data from file
        data = np.loadtxt(beat_file, comments='#')
        
        # Check if we have downbeat information (2 columns)
        if data.ndim == 2 and data.shape[1] == 2:
            # First column: timestamps
            beat_timestamps = data[:, 0]
            
            # Second column: downbeat flags (1=downbeat, 0=regular beat)
            downbeat_flags = data[:, 1].astype(int)
            
            # Get indices of downbeats
            downbeats = np.where(downbeat_flags == 1)[0]
            
            print(f"Loaded {len(beat_timestamps)} beats with {len(downbeats)} downbeats")
            return beat_timestamps, downbeats
        else:
            # No downbeat information, just timestamps
            beat_timestamps = data if data.ndim == 1 else data[:, 0]
            print(f"Loaded {len(beat_timestamps)} beats (no downbeat information)")
            return beat_timestamps, None
            
    except Exception as e:
        print(f"Error loading beats from {beat_file}: {e}")
        return np.array([]), None


def process_beat_file(beat_file, audio_dir, output_dir, resolution, fps, 
                      meter, flash_duration, generate_flash, no_counter, 
                      audio_file=None, verbose=True):
    """
    Process a single beat file to generate videos.
    
    Parameters:
    -----------
    beat_file : str or pathlib.Path
        Path to the beat timestamp file
    audio_dir : str or pathlib.Path
        Directory containing audio files
    output_dir : str or pathlib.Path
        Directory to save output files
    resolution : tuple
        Video resolution as (width, height)
    fps : int
        Frames per second for the video
    meter : int
        Number of beats per measure (time signature numerator)
    flash_duration : float
        Duration of the flash effect in seconds
    generate_flash : bool
        Whether to generate the flash video (default is False)
    no_counter : bool
        Whether to skip generating the counter video
    audio_file : str or pathlib.Path, optional
        Path to a specific audio file to use
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    bool
        True if at least one video was successfully generated
    """
    # Ensure paths are Path objects
    beats_path = pathlib.Path(beat_file)
    audio_directory = pathlib.Path(audio_dir)
    base_output_dir = pathlib.Path(output_dir) if output_dir else beats_path.parent
    
    # Preserve subdirectory structure
    if "data/beats" in str(beats_path):
        # Get relative path from data/beats
        rel_path = beats_path.relative_to(pathlib.Path("data/beats"))
        # If there's a parent directory, use it in the output path
        if rel_path.parent != pathlib.Path("."):
            output_dir_path = base_output_dir / rel_path.parent
            output_directory = file_utils.ensure_directory(output_dir_path)
        else:
            output_directory = file_utils.ensure_directory(base_output_dir)
    else:
        output_directory = file_utils.ensure_directory(base_output_dir)
    
    if verbose:
        print(f"\nProcessing: {beats_path}")
        print("=" * 80)
    
    # Load beat timestamps and downbeat information
    beat_timestamps, downbeats = load_beat_data(beats_path)
    
    if len(beat_timestamps) == 0:
        if verbose:
            print(f"No beats found in {beats_path}")
        return False
    
    # Find audio file
    audio_file_path = None
    if audio_file:
        audio_file_path = pathlib.Path(audio_file)
        if not audio_file_path.exists():
            if verbose:
                print(f"Audio file not found: {audio_file_path}")
            return False
    else:
        # Extract audio filename from the beats filename
        audio_name = beats_path.stem.replace('_beats', '')
        
        # Check if we have a subdirectory structure to preserve
        if beats_path.parent.name != "beats":
            # If beat file is in a subdirectory of data/beats, look for audio in corresponding subdirectory
            subdir_path = beats_path.parent.relative_to(pathlib.Path("data/beats")) if "data/beats" in str(beats_path) else beats_path.parent.name
            audio_subdir = audio_directory / subdir_path
            audio_file_path = file_utils.find_related_file(
                audio_name, 
                audio_subdir,
                extensions=AUDIO_EXTENSIONS
            )
        else:
            # No subdirectory, search in the main audio directory
            audio_file_path = file_utils.find_related_file(
                audio_name, 
                audio_directory,
                extensions=AUDIO_EXTENSIONS
            )
        
        # If not found in expected location, try a recursive search
        if not audio_file_path:
            # Try searching recursively through all subdirectories
            for ext in AUDIO_EXTENSIONS:
                matches = list(audio_directory.rglob(f"{audio_name}{ext}"))
                if matches:
                    audio_file_path = matches[0]
                    break
        
        if not audio_file_path:
            if verbose:
                print(f"Could not find matching audio file for {beats_path}")
                print("Please specify with --audio-file")
            return False
    
    if verbose:
        print(f"Using audio file: {audio_file_path}")
    
    # Create video generator with meter and color settings
    video_generator = BeatVideoGenerator(
        resolution=resolution,
        fps=fps,
        meter=meter,
        downbeat_color=(255, 0, 0),  # Red for downbeats
        downbeat_flash_color=(255, 100, 100)  # Light red for downbeat flashes
    )
    
    # Generate output file base name
    output_base = output_directory / beats_path.stem.replace('_beats', '')
    
    # Create the output directory (it might be a subdirectory)
    os.makedirs(output_directory, exist_ok=True)
    
    # Generate videos
    videos_generated = 0
    
    # Flash video (only if explicitly requested)
    if generate_flash:
        flash_video = file_utils.get_output_path(output_base, output_directory, suffix='_flash', ext='.mp4')
        if verbose:
            print(f"Generating flash video with {'downbeat detection' if downbeats is not None else 'regular meter'}...")
        try:
            video_generator.create_flash_video(
                str(audio_file_path), 
                beat_timestamps, 
                str(flash_video),
                downbeats=downbeats,
                flash_duration=flash_duration
            )
            if verbose:
                print(f"Flash video saved: {flash_video}")
            videos_generated += 1
        except Exception as e:
            if verbose:
                print(f"Error generating flash video: {e}")
    
    # Counter video (generated by default unless explicitly disabled)
    if not no_counter:
        counter_video = file_utils.get_output_path(output_base, output_directory, suffix='_counter', ext='.mp4')
        if verbose:
            print(f"Generating counter video with {meter}/4 time and {'downbeat detection' if downbeats is not None else 'regular meter'}...")
        try:
            video_generator.create_counter_video(
                str(audio_file_path), 
                beat_timestamps, 
                str(counter_video),
                downbeats=downbeats,
                meter=meter
            )
            if verbose:
                print(f"Counter video saved: {counter_video}")
            videos_generated += 1
        except Exception as e:
            if verbose:
                print(f"Error generating counter video: {e}")
    
    if verbose:
        print(f"Generated {videos_generated} videos")
    return videos_generated > 0


def process_directory(directory, audio_dir, output_dir, resolution, fps, 
                      meter, flash_duration, generate_flash, no_counter, verbose=True):
    """
    Process all beat files in a directory.
    
    Parameters:
    -----------
    directory : str or pathlib.Path
        Directory containing beat files
    audio_dir : str or pathlib.Path
        Directory containing audio files
    output_dir : str or pathlib.Path
        Directory to save output files
    resolution : tuple
        Video resolution as (width, height)
    fps : int
        Frames per second for the video
    meter : int
        Number of beats to count before resetting
    flash_duration : float
        Duration of the flash effect in seconds
    generate_flash : bool
        Whether to generate the flash videos
    no_counter : bool
        Whether to skip generating the counter video
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    list
        List of processed files and their results
    """
    # Find beat files
    beat_files = file_utils.find_files_by_pattern(directory, '*_beats.txt')
    
    # Process each file
    return file_utils.batch_process(
        beat_files,
        process_func=process_beat_file,
        verbose=verbose,
        audio_dir=audio_dir,
        output_dir=output_dir,
        resolution=resolution,
        fps=fps,
        meter=meter,
        flash_duration=flash_duration,
        generate_flash=generate_flash,
        no_counter=no_counter
    )


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Parse video resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        print(f"Invalid resolution format: {args.resolution}. Using default 1280x720.")
        resolution = (1280, 720)
    
    # Determine output dir (if specified)
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else None
    
    # Process input
    if args.input:
        input_path = pathlib.Path(args.input)
        
        # Check if input is an audio file instead of a beats file
        if input_path.suffix.lower() in AUDIO_EXTENSIONS:
            print(f"Audio file detected: {input_path}")
            
            # Use the utility function to find the beats file
            print(f"Looking for corresponding beats file...")
            beats_file = file_utils.find_beats_file_for_audio(input_path)
            
            if beats_file:
                print(f"Found corresponding beats file: {beats_file}")
                # Process with the found beats file and the provided audio file
                process_beat_file(
                    beats_file,
                    audio_dir=args.audio_dir,
                    output_dir=output_dir,
                    resolution=resolution,
                    fps=args.fps,
                    meter=args.meter,
                    flash_duration=args.flash_duration,
                    generate_flash=args.flash,
                    no_counter=args.no_counter,
                    audio_file=str(input_path),  # Use the provided audio file
                    verbose=not args.quiet
                )
            else:
                print(f"Error: Could not find a beats file for {input_path.name}")
                print(f"Run 'detect-beats {input_path}' first to generate beat timestamps.")
                print(f"Then run 'generate-videos data/beats/{input_path.stem}_beats.txt'")
        else:
            # Standard processing for beat files or directories
            file_utils.process_input_path(
                args.input,
                default_dir="data/beats",
                process_file_func=process_beat_file,
                process_dir_func=process_directory,
                audio_dir=args.audio_dir,
                output_dir=output_dir,
                resolution=resolution,
                fps=args.fps,
                meter=args.meter,
                flash_duration=args.flash_duration,
                generate_flash=args.flash,
                no_counter=args.no_counter,
                audio_file=args.audio_file,
                verbose=not args.quiet
            )
    else:
        # Default: process all files in data/beats
        process_directory(
            "data/beats",
            args.audio_dir,
            output_dir,
            resolution=resolution,
            fps=args.fps,
            meter=args.meter,
            flash_duration=args.flash_duration,
            generate_flash=args.flash,
            no_counter=args.no_counter,
            verbose=not args.quiet
        )


if __name__ == "__main__":
    main()