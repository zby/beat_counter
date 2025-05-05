#!/usr/bin/env python3
"""
Batch generate beat visualization videos.

Scans a directory for audio files, finds the corresponding .beats JSON file for
each, and generates a beat-counter video, saving it either alongside the audio
(default) or into a specified output directory.

Usage:
    generate-videos-batch [options] [DIRECTORY]

DIRECTORY defaults to `data/input`.
"""
from __future__ import annotations

import argparse
import logging
import sys
import pathlib
from typing import Tuple, List

from beat_detection.core.video import (
    generate_batch_videos,  # New centralized batch function
    DEFAULT_VIDEO_WIDTH,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_FPS,
)

# Re-use the parsing function from single-file script for resolution
from beat_detection.cli.generate_video import parse_resolution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate-videos-batch",
        description="Batch generate beat counter videos for audio files in a directory.",
    )

    parser.add_argument(
        "directory",
        nargs="?",
        default="data/input",
        help="Directory to scan recursively for audio files (default: data/input).",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        metavar="DIR",
        default=None,
        help="Directory to save output videos. Defaults to same directory as audio file.",
    )

    parser.add_argument(
        "--resolution",
        default=f"{DEFAULT_VIDEO_WIDTH}x{DEFAULT_VIDEO_HEIGHT}",
        type=parse_resolution,
        help="Video resolution WIDTHxHEIGHT (default: %(default)s)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Frames per second (default: %(default)s)",
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
        help="Generate sample videos using only the first N beats.",
    )

    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def main() -> None:
    args = parse_args()
    setup_logging(verbose=not args.quiet)

    input_dir = pathlib.Path(args.directory)
    if not input_dir.is_dir():
        logging.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    output_dir = pathlib.Path(args.output_dir) if args.output_dir else None

    try:
        # Call the centralized batch function instead of looping here
        results = generate_batch_videos(
            input_dir=input_dir,
            output_dir=output_dir,
            resolution=args.resolution,
            fps=args.fps,
            sample_beats=args.sample,
            tolerance_percent=args.tolerance_percent,
            min_measures=args.min_measures,
            verbose=not args.quiet,
            no_progress=False,  # Enable progress bar by default
        )

        # --- Summary --- (Simplified from original)
        successful_count = sum(1 for _, success, _ in results if success)
        failed_count = len(results) - successful_count

        if not args.quiet:
            print("\n" + "=" * 80)
            print("BATCH VIDEO GENERATION SUMMARY")
            print("=" * 80)
            print(f"Total files processed: {len(results)}")
            print(f"Successful: {successful_count}")
            print(f"Failed/Skipped: {failed_count}")
            if failed_count > 0:
                print("\nFiles that failed or were skipped:")
                for filename, success, _ in results:
                    if not success:
                        print(f"- {filename}")
            print("=" * 80 + "\n")

    except Exception as e:
        print(f"Error during batch processing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 