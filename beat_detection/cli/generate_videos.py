#!/usr/bin/env python3
"""
Generate beat visualization videos from audio files.

This script takes audio files and creates visual representations of the beats as videos.
It can either detect beats automatically from the audio, or use pre-computed beat timestamps
if available.

Usage:
    generate-videos [options] [audio_file_or_directory]

If audio_file is provided, processes that specific file.
If directory is provided, processes all audio files in that directory.
If nothing is provided, processes all audio files in data/input/.
"""

import pathlib
import argparse
import numpy as np
from typing import Optional, Union, Tuple

from beat_detection.core.video import BeatVideoGenerator, DEFAULT_VIDEO_RESOLUTION, DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT
from beat_detection.utils import file_utils
from beat_detection.utils.constants import AUDIO_EXTENSIONS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate beat visualization videos from audio files."
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
        help="Output directory for videos (default: data/output)"
    )
    
    # Video options
    parser.add_argument(
        "--meter", type=int, default=4,
        help="Number of beats per measure (time signature numerator, default: 4)"
    )
    parser.add_argument(
        "--resolution", type=str, default=f"{DEFAULT_VIDEO_WIDTH}x{DEFAULT_VIDEO_HEIGHT}",
        help=f"Video resolution in format WIDTHxHEIGHT (default: {DEFAULT_VIDEO_WIDTH}x{DEFAULT_VIDEO_HEIGHT})"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second for output videos (default: 30)"
    )
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Generate a sample video with only the first N beats (default: None, process all beats)"
    )
    
    # Output options
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )
    
    return parser.parse_args()


def load_beat_data(beat_file: pathlib.Path) -> Tuple[np.ndarray, Optional[np.ndarray], int, int]:
    """
    Load beat timestamps, downbeat information, and intro/ending indices from a file.
    
    Parameters:
    -----------
    beat_file : pathlib.Path
        Path to the beat timestamp file
    
    Returns:
    --------
    tuple
        (beat_timestamps, downbeats, intro_end_idx, ending_start_idx) where:
        - beat_timestamps: Array of beat timestamps in seconds
        - downbeats: Array of indices that correspond to downbeats, or None if not available
        - intro_end_idx: Index where the intro ends (0 if no intro detected)
        - ending_start_idx: Index where the ending begins (len(beat_timestamps) if no ending detected)
    """
    try:
        # Initialize default values for intro and ending indices
        intro_end_idx = 0
        ending_start_idx = -1  # Will be set to len(beat_timestamps) if not found
        
        # Read the file content to extract header information
        with open(beat_file, 'r') as f:
            lines = f.readlines()
            
            # Look for intro and ending information in the header comments
            for line in lines:
                if line.startswith('#'):
                    if 'INTRO_END_IDX=' in line:
                        intro_end_idx = int(line.split('=')[1].strip())
                    elif 'ENDING_START_IDX=' in line:
                        ending_start_idx = int(line.split('=')[1].strip())
        
        # Now load the actual data
        data = np.loadtxt(beat_file, comments='#')
        
        # Check if we have downbeat information (2 columns)
        if data.ndim == 2 and data.shape[1] == 2:
            # First column: timestamps
            beat_timestamps = data[:, 0]
            
            # Second column: downbeat flags (1=downbeat, 0=regular beat)
            downbeat_flags = data[:, 1].astype(int)
            
            # Get indices of downbeats
            downbeats = np.where(downbeat_flags == 1)[0]
            
            # If ending_start_idx wasn't set in the header, set it to the length of beat_timestamps
            if ending_start_idx == -1:
                ending_start_idx = len(beat_timestamps)
                
            print(f"Loaded {len(beat_timestamps)} beats with {len(downbeats)} downbeats")
            if intro_end_idx > 0:
                print(f"Intro section ends at beat {intro_end_idx}")
            if ending_start_idx < len(beat_timestamps):
                print(f"Ending section starts at beat {ending_start_idx}")
                
            return beat_timestamps, downbeats, intro_end_idx, ending_start_idx
        else:
            # No downbeat information, just timestamps
            beat_timestamps = data if data.ndim == 1 else data[:, 0]
            
            # If ending_start_idx wasn't set in the header, set it to the length of beat_timestamps
            if ending_start_idx == -1:
                ending_start_idx = len(beat_timestamps)
                
            print(f"Loaded {len(beat_timestamps)} beats (no downbeat information)")
            if intro_end_idx > 0:
                print(f"Intro section ends at beat {intro_end_idx}")
            if ending_start_idx < len(beat_timestamps):
                print(f"Ending section starts at beat {ending_start_idx}")
                
            return beat_timestamps, None, intro_end_idx, ending_start_idx
            
    except Exception as e:
        print(f"Error loading beats from {beat_file}: {e}")
        return np.array([]), None, 0, 0


def generate_counter_video(audio_path: pathlib.Path, output_file: pathlib.Path,
                        beat_timestamps: np.ndarray, downbeats: np.ndarray,
                        intro_end_idx: int = 0, ending_start_idx: Optional[int] = None,
                        resolution=DEFAULT_VIDEO_RESOLUTION, fps=30, meter=4, 
                        sample_beats=None, verbose=True) -> bool:
    """Generate a counter video for a given audio file with beat timestamps.
    
    Parameters:
    -----------
    audio_path : pathlib.Path
        Path to the audio file
    output_file : pathlib.Path
        Path where the output video will be saved
    beat_timestamps : numpy.ndarray
        Array of beat timestamps in seconds
    downbeats : numpy.ndarray
        Array of indices that correspond to downbeats
    resolution : tuple
        Video resolution as (width, height)
    fps : int
        Frames per second for the video
    meter : int
        Number of beats per measure
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    bool
        True if video was successfully generated, False otherwise
    """
    # Create video generator with meter settings
    video_generator = BeatVideoGenerator(
        resolution=resolution,
        fps=fps,
        meter=meter
    )
    
    if verbose:
        print(f"Generating counter video with {meter}/4 time and downbeat detection...")
    try:
        # Apply intro and ending filtering if specified
        filtered_beat_timestamps = beat_timestamps
        filtered_downbeats = downbeats
        
        # Filter out intro beats if intro_end_idx is provided
        if intro_end_idx > 0:
            if verbose:
                print(f"Skipping intro section (first {intro_end_idx} beats)")
            filtered_beat_timestamps = beat_timestamps[intro_end_idx:]
            if downbeats is not None:
                # Adjust downbeat indices to account for removed intro beats
                filtered_downbeats = downbeats[downbeats >= intro_end_idx] - intro_end_idx
        
        # Filter out ending beats if ending_start_idx is provided
        if ending_start_idx is not None and ending_start_idx < len(beat_timestamps):
            if verbose:
                print(f"Skipping ending section (last {len(beat_timestamps) - ending_start_idx} beats)")
            # Apply ending filter after intro filter
            ending_idx_adjusted = ending_start_idx - intro_end_idx if intro_end_idx > 0 else ending_start_idx
            if ending_idx_adjusted > 0:  # Make sure we have beats left after filtering
                filtered_beat_timestamps = filtered_beat_timestamps[:ending_idx_adjusted]
                if filtered_downbeats is not None:
                    filtered_downbeats = filtered_downbeats[filtered_downbeats < ending_idx_adjusted]
        
        video_generator.create_counter_video(
            audio_file=str(audio_path), 
            output_file=str(output_file),
            beat_timestamps=filtered_beat_timestamps,
            downbeats=filtered_downbeats,
            meter=meter,
            sample_beats=sample_beats
        )
        if verbose:
            print(f"Counter video saved: {output_file}")
        return True
    except Exception as e:
        if verbose:
            print(f"Error generating counter video: {e}")
        return False

def process_audio_file(audio_file, output_dir=None, resolution=DEFAULT_VIDEO_RESOLUTION, fps=30, 
                      meter=4, sample_beats=None, verbose=True, 
                      input_base_dir="data/input", output_base_dir="data/output"):
    """
    Process an audio file to generate a beat visualization video.
    
    This function first tries to find a corresponding beats file to use for beat timestamps.
    If no beats file is found, it uses BeatVideoGenerator's automatic beat detection.
    
    The function generates a counter video that displays a beat counter incrementing on each beat.
    
    Parameters:
    -----------
    audio_file : str or pathlib.Path
        Path to the audio file to process
    output_dir : str or pathlib.Path, optional
        Directory to save output files
    resolution : tuple
        Video resolution as (width, height)
    fps : int
        Frames per second for the video
    meter : int
        Number of beats per measure (time signature numerator)
    verbose : bool
        Whether to print progress
    input_base_dir : str
        Base input directory (default: "data/input")
    output_base_dir : str
        Base output directory (default: "data/output")
    
    Returns:
    --------
    bool
        True if at least one video was successfully generated
    """
    audio_path = pathlib.Path(audio_file)
    
    # Try to find the corresponding beats file
    beats_file = file_utils.find_beats_file_for_audio(
        audio_path, 
        input_base_dir=input_base_dir, 
        output_base_dir=output_base_dir
    )
    
    if not audio_path.exists():
        if verbose:
            print(f"Audio file not found: {audio_path}")
        return False
    
    if verbose:
        print(f"\nProcessing: {audio_path}")
        print("=" * 80)
    
    # Load beat timestamps if available
    if beats_file is not None and beats_file.exists():
        if verbose:
            print(f"Found corresponding beats file: {beats_file}")
        beat_timestamps, downbeats, intro_end_idx, ending_start_idx = load_beat_data(beats_file)
        if len(beat_timestamps) == 0:
            raise ValueError(f"No beats found in {beats_file}. Please generate beat data first using the beat detection tool.")
    else:
        raise FileNotFoundError(f"No beats file found for {audio_path.name}. Please generate beat data first using the beat detection tool.")
    
    # Ensure we have valid beat timestamps and downbeats
    if beat_timestamps is None or len(beat_timestamps) == 0:
        raise ValueError(f"No valid beats found for {audio_path.name}. Please generate beat data first using the beat detection tool.")
    
    # Generate counter video
    counter_video = file_utils.get_output_path(
        audio_path, 
        suffix='_counter', 
        ext='.mp4',
        input_base_dir=input_base_dir,
        output_base_dir=output_base_dir
    )
    
    success = generate_counter_video(
        audio_path=audio_path,
        output_file=counter_video,
        beat_timestamps=beat_timestamps,
        downbeats=downbeats,
        intro_end_idx=intro_end_idx,
        ending_start_idx=ending_start_idx,
        resolution=resolution,
        fps=fps,
        meter=meter,
        sample_beats=sample_beats,
        verbose=verbose
    )
    
    return success


def process_directory(directory, output_dir, resolution=DEFAULT_VIDEO_RESOLUTION, fps=30, 
                      meter=4, sample_beats=None, verbose=True):
    """
    Process all audio files in a directory to generate beat visualization videos.
    
    For each audio file, this function will:
    1. Try to find a corresponding beats file to use for beat timestamps
    2. If no beats file is found, use automatic beat detection
    3. Generate counter videos that display beat counts
    
    Parameters:
    -----------
    directory : str or pathlib.Path
        Directory containing audio files
    output_dir : str or pathlib.Path
        Directory to save output files
    resolution : tuple
        Video resolution as (width, height)
    fps : int
        Frames per second for the video
    meter : int
        Number of beats to count before resetting
    no_counter : bool
        Whether to skip generating the counter video
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    list
        List of processed files and their results
    """
    # Get list of audio files in directory
    audio_files = file_utils.find_audio_files(directory)
    
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
            success = process_audio_file(
                audio_file=audio_file,
                output_dir=output_dir,
                resolution=resolution,
                fps=fps,
                meter=meter,
                sample_beats=sample_beats,
                verbose=verbose
            )
            results.append((audio_file.name, success))
        except FileNotFoundError as e:
            if verbose:
                print(f"Missing beats file for {audio_file.name}: {e}")
                print(f"Please run the beat detection tool first to generate beat data.")
            results.append((audio_file.name, False))
        except ValueError as e:
            if verbose:
                print(f"Invalid beat data for {audio_file.name}: {e}")
                print(f"Please run the beat detection tool first to generate valid beat data.")
            results.append((audio_file.name, False))
        except Exception as e:
            if verbose:
                print(f"Error processing {audio_file}: {e}")
            results.append((audio_file.name, False))
    
    return results


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Parse video resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        print(f"Invalid resolution format: {args.resolution}. Using default {DEFAULT_VIDEO_WIDTH}x{DEFAULT_VIDEO_HEIGHT}.")
        resolution = DEFAULT_VIDEO_RESOLUTION
    
    # Base directories
    input_base_dir = "data/input"
    output_base_dir = pathlib.Path(args.output_dir)
    
    # Get input path to process
    input_path = args.input if args.input else input_base_dir
    
    # Find audio files to process
    audio_files = file_utils.find_audio_files(input_path)
    
    if not audio_files:
        print(f"No audio files found in {input_path}")
        return
    
    if not args.quiet:
        print(f"Found {len(audio_files)} audio files to process")
    
    # Process each audio file
    for audio_file in audio_files:
        try:
            process_audio_file(
                audio_file=audio_file,
                output_dir=output_base_dir,
                resolution=resolution,
                fps=args.fps,
                meter=args.meter,
                sample_beats=args.sample,
                verbose=not args.quiet,
                input_base_dir=input_base_dir,
                output_base_dir=output_base_dir
            )
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")


if __name__ == "__main__":
    main()