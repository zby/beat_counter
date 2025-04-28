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
    BeatVideoGenerator,
    DEFAULT_VIDEO_WIDTH,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_FPS,
)
from beat_detection.utils.file_utils import find_audio_files
from beat_detection.utils.beat_file import load_beats

# Re-use the parsing function from single-file script for resolution
from beat_detection.cli.generate_video import _parse_resolution as parse_resolution_helper

# Reuse the single-file processing logic
from beat_detection.cli.generate_video import process_audio_file as generate_single_video


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
        type=parse_resolution_helper,
        help="Video resolution WIDTHxHEIGHT (default: %(default)s)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Frames per second (default: %(default)s)",
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

    audio_files = find_audio_files(input_dir)
    if not audio_files:
        logging.warning("No audio files found in %s", input_dir)
        sys.exit(0)

    if not args.quiet:
        print(f"Found {len(audio_files)} audio files in {input_dir}")

    results: List[Tuple[str, bool]] = []

    for audio_file in audio_files:
        if not args.quiet:
            print(f"\nProcessing: {audio_file}")

        try:
            # Determine the specific output file path for this audio file
            if output_dir:
                # Ensure output dir exists (process_audio_file also does this, but belt-and-suspenders)
                output_dir.mkdir(parents=True, exist_ok=True)
                single_output_file = output_dir / f"{audio_file.stem}_counter.mp4"
            else:
                # If no output dir specified, let generate_single_video use its default logic
                single_output_file = None

            # Call the single-file processing function with the specific file path
            success = generate_single_video(
                audio_file=audio_file,
                output_file=single_output_file, # Pass the constructed path or None
                resolution=args.resolution,
                fps=args.fps,
                sample_beats=args.sample,
                verbose=not args.quiet,
            )
            results.append((audio_file.name, success))
        except FileNotFoundError as e:
            if not args.quiet:
                logging.warning("Skipping %s: %s", audio_file.name, e)
            results.append((audio_file.name, False))
        except Exception as e:
            logging.exception("Error processing %s: %s", audio_file.name, e)
            results.append((audio_file.name, False))

    # --- Summary --- (Optional)
    successful_count = sum(1 for _, success in results if success)
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
            for filename, success in results:
                if not success:
                    print(f"- {filename}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main() 