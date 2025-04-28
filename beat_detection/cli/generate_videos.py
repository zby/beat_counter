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
from typing import Optional, Union, Tuple, Callable, List

from beat_detection.core.video import (
    BeatVideoGenerator,
    DEFAULT_VIDEO_RESOLUTION,
    DEFAULT_VIDEO_WIDTH,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_FPS,
)
from beat_detection.utils import file_utils
from beat_detection.utils.constants import AUDIO_EXTENSIONS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate beat visualization videos from audio files."
    )

    # Input arguments
    parser.add_argument(
        "input",
        nargs="?",
        help="Input audio file or directory. If not provided, all files in data/input will be processed.",
    )

    # Output directory
    parser.add_argument(
        "-o",
        "--output-dir",
        default="data/output",
        help="Output directory for videos (default: data/output)",
    )

    # Video options
    parser.add_argument(
        "--resolution",
        type=str,
        default=f"{DEFAULT_VIDEO_WIDTH}x{DEFAULT_VIDEO_HEIGHT}",
        help=f"Video resolution in format WIDTHxHEIGHT (default: {DEFAULT_VIDEO_WIDTH}x{DEFAULT_VIDEO_HEIGHT})",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Frames per second for output videos (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Generate a sample video with only the first N beats (default: None, process all beats)",
    )

    # Output options
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    return parser.parse_args()


from ..utils.beat_file import load_beat_data as base_load_beat_data


def load_beat_data(
    beat_file: Union[str, pathlib.Path],
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    """
    Load beat timestamps, downbeat information, intro/ending indices, and detected beats_per_bar from a file.

    Parameters:
    -----------
    beat_file : Union[str, Path]
        Path to the beat data file

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, int, int, int]
        (beat_timestamps, downbeats, intro_end_idx, ending_start_idx, detected_beats_per_bar) where:
        - beat_timestamps: Array of beat timestamps
        - downbeats: Array of downbeat indices
        - intro_end_idx: Index of the last intro beat
        - ending_start_idx: Index of the first ending beat
        - detected_beats_per_bar: Detected beats per bar (time signature numerator, typically 3 or 4)
    """
    try:
        # Use the base load_beat_data function from the beat_file module
        (
            beat_timestamps,
            downbeats,
            intro_end_idx,
            ending_start_idx,
            detected_beats_per_bar,
        ) = base_load_beat_data(str(beat_file))

        # Check if we have any data at all
        if len(beat_timestamps) == 0:
            raise ValueError(f"No beat timestamps found in {beat_file}")

        # Check if detected_beats_per_bar was found in the header
        if detected_beats_per_bar is None:
            raise ValueError(
                f"No detected beats_per_bar information found in {beat_file}. Please ensure the file has a '# DETECTED_BEATS_PER_BAR=X' header."
            )

        # Check if we have any downbeats
        if len(downbeats) == 0:
            raise ValueError(
                f"No downbeat information found in {beat_file}. Please ensure the file has at least one downbeat marked."
            )

        # If ending_start_idx wasn't set, set it to the length of beat_timestamps
        if ending_start_idx is None:
            ending_start_idx = len(beat_timestamps)

        # Print information about the loaded data
        print(f"Loaded {len(beat_timestamps)} beats with {len(downbeats)} downbeats")
        if intro_end_idx > 0:
            print(f"Intro section ends at beat {intro_end_idx}")
        if ending_start_idx < len(beat_timestamps):
            print(f"Ending section starts at beat {ending_start_idx}")
        print(f"Detected beats per bar: {detected_beats_per_bar}/4 time signature")

        return (
            beat_timestamps,
            downbeats,
            intro_end_idx,
            ending_start_idx,
            detected_beats_per_bar,
        )
    except Exception as e:
        print(f"Error loading beats from {beat_file}: {e}")
        raise ValueError(f"Failed to load beat data from {beat_file}: {e}")


def generate_counter_video(
    audio_file: Union[str, pathlib.Path],
    output_file: Union[str, pathlib.Path],
    beats_file: Optional[Union[str, pathlib.Path]] = None,
    resolution: Tuple[int, int] = DEFAULT_VIDEO_RESOLUTION,
    fps: int = 30,
    beats_per_bar: int = 4,
    sample_beats: Optional[int] = None,
    verbose: bool = True,
) -> str:
    """
    Generate a beat counter video.

    Parameters:
    -----------
    audio_file : Union[str, Path]
        Path to the input audio file
    output_file : Union[str, Path]
        Path to save the output video
    beats_file : Optional[Union[str, Path]]
        Path to a pre-computed beats file (optional)
    resolution : Tuple[int, int]
        Video resolution (width, height)
    fps : int
        Frames per second
    beats_per_bar : int
        Number of beats per bar (time signature numerator)
    sample_beats : Optional[int]
        Number of beats to process (for testing)
    verbose : bool
        Whether to print progress information

    Returns:
    --------
    str
        Path to the generated video file
    """
    # Create video generator with beats_per_bar settings
    video_gen = BeatVideoGenerator(
        resolution=resolution, fps=fps, beats_per_bar=beats_per_bar
    )

    if verbose:
        print(
            f"Generating counter video with {beats_per_bar}/4 time and downbeat detection..."
        )

    # Generate the video
    return video_gen.generate_video(
        audio_file, beats_file, output_file, sample_beats=sample_beats, verbose=verbose
    )


def process_audio_file(
    audio_file,
    output_dir=None,
    resolution=DEFAULT_VIDEO_RESOLUTION,
    fps=30,
    beats_per_bar=4,
    sample_beats=None,
    verbose=True,
    input_base_dir="data/input",
    output_base_dir="data/output",
):
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
    beats_per_bar : int
        Number of beats per bar (time signature numerator)
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
        audio_path, input_base_dir=input_base_dir, output_base_dir=output_base_dir
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
        (
            beat_timestamps,
            downbeats,
            intro_end_idx,
            ending_start_idx,
            detected_beats_per_bar,
        ) = load_beat_data(beats_file)
        if len(beat_timestamps) == 0:
            raise ValueError(
                f"No beats found in {beats_file}. Please generate beat data first using the beat detection tool."
            )

        # Use the detected beats_per_bar from the beat file
        beats_per_bar = detected_beats_per_bar
        if verbose:
            print(f"Using detected beats_per_bar {beats_per_bar}/4 from beats file")
    else:
        raise FileNotFoundError(
            f"No beats file found for {audio_path.name}. Please generate beat data first using the beat detection tool."
        )

    # Ensure we have valid beat timestamps and downbeats
    if beat_timestamps is None or len(beat_timestamps) == 0:
        raise ValueError(
            f"No valid beats found for {audio_path.name}. Please generate beat data first using the beat detection tool."
        )

    # Generate counter video
    counter_video = file_utils.get_output_path(
        audio_path,
        suffix="_counter",
        ext=".mp4",
        input_base_dir=input_base_dir,
        output_base_dir=output_base_dir,
    )

    success = generate_counter_video(
        audio_file=audio_path,
        output_file=counter_video,
        beats_file=beats_file,
        resolution=resolution,
        fps=fps,
        beats_per_bar=beats_per_bar,
        sample_beats=sample_beats,
        verbose=verbose,
    )

    return success


def process_directory(
    directory,
    output_dir,
    resolution=DEFAULT_VIDEO_RESOLUTION,
    fps=30,
    sample_beats=None,
    verbose=True,
):
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

    sample_beats : int or None
        Number of beats to sample from each audio file
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
                beats_per_bar=4,
                sample_beats=sample_beats,
                verbose=verbose,
            )
            results.append((audio_file.name, success))
        except FileNotFoundError as e:
            if verbose:
                print(f"Missing beats file for {audio_file.name}: {e}")
                print(
                    f"Please run the beat detection tool first to generate beat data."
                )
            results.append((audio_file.name, False))
        except ValueError as e:
            if verbose:
                print(f"Invalid beat data for {audio_file.name}: {e}")
                print(
                    f"Please run the beat detection tool first to generate valid beat data."
                )
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
        width, height = map(int, args.resolution.split("x"))
        resolution = (width, height)
    except ValueError:
        print(
            f"Invalid resolution format: {args.resolution}. Using default {DEFAULT_VIDEO_WIDTH}x{DEFAULT_VIDEO_HEIGHT}."
        )
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
                beats_per_bar=4,
                sample_beats=args.sample,
                verbose=not args.quiet,
                input_base_dir=input_base_dir,
                output_base_dir=output_base_dir,
            )
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")


if __name__ == "__main__":
    main()
