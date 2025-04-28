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
from typing import Optional, Union, Tuple, List
import sys

from beat_detection.core.video import (
    BeatVideoGenerator,
    DEFAULT_VIDEO_RESOLUTION,
    DEFAULT_VIDEO_WIDTH,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_FPS,
)
from beat_detection.utils import file_utils


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_resolution(res_str: str) -> Tuple[int, int]:
    try:
        w, h = map(int, res_str.lower().split("x"))
        return (w, h)
    except ValueError as exc:  # pragma: no cover – simple validation
        raise argparse.ArgumentTypeError(
            f"Invalid resolution format '{res_str}', expected WIDTHxHEIGHT"
        ) from exc


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate beat visualization videos from audio files."
    )

    # Input arguments
    parser.add_argument(
        "audio_file",
        help="Path to the audio file to process.",
    )

    # Output directory -> Output file
    parser.add_argument(
        "-o",
        "--output-file",
        default=None,
        help="Path to save the generated video file. Defaults to the audio file's directory with a '_counter.mp4' suffix.",
    )

    # Video options
    parser.add_argument(
        "--resolution",
        type=_parse_resolution,
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
    audio_file: pathlib.Path,
    output_file: pathlib.Path | None = None,
    resolution=DEFAULT_VIDEO_RESOLUTION,
    fps: int = 30,
    sample_beats: int | None = None,
    verbose: bool = True,
):
    """
    Process an audio file to generate a beat visualization video.

    This function first tries to find a corresponding beats file to use for beat timestamps.
    If no beats file is found, it uses BeatVideoGenerator's automatic beat detection.

    The function generates a counter video that displays a beat counter incrementing on each beat.

    Parameters:
    -----------
    audio_file : pathlib.Path
        Path to the audio file to process
    output_file : pathlib.Path | None
        Path to save the output video
    resolution : tuple
        Video resolution as (width, height)
    fps : int
        Frames per second for the video
    sample_beats : int | None
        Number of beats to process (for testing)
    verbose : bool
        Whether to print progress

    Returns:
    --------
    bool
        True if at least one video was successfully generated
    """
    audio_path = audio_file

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    beats_path = audio_path.with_suffix(".beats")
    if not beats_path.is_file():
        raise FileNotFoundError(
            f"Beats file not found: {beats_path}. Run beat detection first."
        )

    # Load Beats object
    from beat_detection.utils.beat_file import load_beats

    beats = load_beats(str(beats_path))

    # Determine output video path
    if output_file is None:
        # Default behavior: save next to audio file
        output_path = audio_path.with_name(f"{audio_path.stem}_counter.mp4")
    else:
        # Use provided output file path
        output_path = output_file
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Generating video for {audio_path} using {beats_path}")

    # Create generator
    video_gen = BeatVideoGenerator(resolution=resolution, fps=fps)

    video_gen.generate_video(audio_path, beats, output_path, sample_beats=sample_beats)

    if verbose:
        print(f"Saved video to {output_path}")

    return True


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    # Parse video resolution
    # The type argument in parse_args already handles conversion
    resolution = args.resolution

    # ------------------------------------------------------------------
    # Process the single audio file
    # ------------------------------------------------------------------
    try:
        process_audio_file(
            audio_file=pathlib.Path(args.audio_file),
            output_file=pathlib.Path(args.output_file) if args.output_file else None,
            resolution=resolution,
            fps=args.fps,
            sample_beats=args.sample,
            verbose=not args.quiet,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1) # Exit with error code 1 for file not found
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1) # Exit with error code 1 for other errors


if __name__ == "__main__":
    main()
