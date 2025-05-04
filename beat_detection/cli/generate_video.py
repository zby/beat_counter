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
from beat_detection.core.beats import Beats, RawBeats


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_resolution(res_str: str) -> Tuple[int, int]:
    """Parse resolution string WIDTHxHEIGHT."""
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
        type=parse_resolution,
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
        "--tolerance-percent",
        type=float,
        default=10.0,
        help="Tolerance percentage used for reconstructing Beats stats/sections (default: 10.0)."
    )
    parser.add_argument(
        "--min-measures",
        type=int,
        default=5,
        help="Minimum measures used for reconstructing Beats stats/sections (default: 5)."
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
    audio_file: pathlib.Path,
    beats_file: pathlib.Path,
    tolerance_percent: float,
    min_measures: int,
    output_file: pathlib.Path | None = None,
    resolution=DEFAULT_VIDEO_RESOLUTION,
    fps: int = 30,
    sample_beats: int | None = None,
    verbose: bool = True,
):
    """
    Process an audio file and its corresponding raw beats file to generate a beat visualization video.

    This function loads raw beat data, reconstructs the full Beats object using
    the provided parameters, and then generates the video.
    """
    audio_path = audio_file
    beats_path = beats_file

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not beats_path.is_file():
        raise FileNotFoundError(f"Beats file not found: {beats_path}.")

    # Load Raw Beats Data
    try:
        raw_beats = RawBeats.load_from_file(beats_path)
        if verbose:
            print(f"Loaded raw beats data from {beats_path} with {len(raw_beats.timestamps)} beats")
    except Exception as e:
        raise RuntimeError(f"Failed to load raw beats from {beats_path}: {e}") from e

    # Reconstruct Beats object using parameters from raw_beats and function args
    try:
        # Create Beats object from RawBeats, inferring beats_per_bar if not provided
        beats = Beats(
            raw_beats=raw_beats,
            beats_per_bar=None,  # Let Beats infer beats_per_bar from the pattern
            tolerance_percent=tolerance_percent,
            min_measures=min_measures
        )
        if verbose:
            print(f"Reconstructed Beats object using bpb={beats.beats_per_bar}, tol={tolerance_percent}, min_meas={min_measures}")
    except Exception as e:
        raise RuntimeError(f"Failed to reconstruct Beats object: {e}") from e

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
        print(f"Generating video for {audio_path} using reconstructed beats")

    # Create generator
    video_gen = BeatVideoGenerator(resolution=resolution, fps=fps)

    # Generate video using the reconstructed Beats object
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

    audio_path = pathlib.Path(args.audio_file)
    beats_path = audio_path.with_suffix(".beats")

    # ------------------------------------------------------------------
    # Process the single audio file
    # ------------------------------------------------------------------
    try:
        generate_counter_video(
            audio_file=audio_path,
            beats_file=beats_path,
            tolerance_percent=args.tolerance_percent,
            min_measures=args.min_measures,
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
