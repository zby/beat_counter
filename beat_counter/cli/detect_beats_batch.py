#!/usr/bin/env python3
"""
Batch beat-detection CLI.
It recursively scans a directory for audio files, detects beats for each, and
writes a ``.beats`` JSON file right next to every audio file.  By default the
root directory is ``data/input`` but you can override it with a positional
argument.

Example usage
-------------
    $ detect-beats-batch                       # process data/input
    $ detect-beats-batch /my/folder --quiet    # minimal output
    $ detect-beats-batch --use-genre-defaults  # use genre-specific defaults if available

The script intentionally *does not* expose an "output directory" option – the
output always lives alongside the source audio.  This matches the project's
fail-fast philosophy by making storage layout explicit and predictable.

Genre-based defaults:
--------------------
If the --use-genre-defaults flag is provided, the script will look for files in
paths matching "/by_genre/<genre>/" and use genre-specific defaults for those files.
This allows different genres to have appropriate BPM ranges and beats_per_bar values.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from typing import List, Tuple, Optional, Dict, Any

from beat_counter.core import extract_beats
from beat_counter.core.beats import Beats
from beat_counter.core.pipeline import process_batch
from beat_counter.genre_db import GenreDB, parse_genre_from_path
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the beat detection batch processor.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Process multiple audio files for beat detection. "
            "Automatically detects audio files in the specified directory (including subdirectories). "
            "Creates .beats files alongside each audio file."
        )
    )

    # Required arguments
    parser.add_argument(
        "directory",
        help="Directory containing audio files to process."
    )

    # Beat detection algorithm
    parser.add_argument(
        "--detector-name",
        choices=["madmom", "beat_this"],
        default="madmom",
        help="Which beat detection algorithm to use. Default: madmom"
    )

    # BPM range arguments
    parser.add_argument(
        "--min-bpm",
        type=float,
        help="Minimum BPM to consider (default depends on algorithm)."
    )
    parser.add_argument(
        "--max-bpm",
        type=float,
        help="Maximum BPM to consider (default depends on algorithm)."
    )

    # Beat grouping/analysis arguments
    parser.add_argument(
        "--beats-per-bar",
        type=int,
        help="Number of beats per bar, or time signature numerator. "
            "If not provided, will be inferred from the detected beats. "
            "Common values: 3 (for 3/4 waltz), 4 (for 4/4), etc."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=10.0,
        help="Tolerance percent for beat interval variation in regular section detection. Default: 10.0"
    )
    parser.add_argument(
        "--min-measures",
        type=int,
        default=2,
        help="Minimum number of consistent measures required for regular section detection.",
    )

    # Output control
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar.")
    
    # New: Genre-based defaults option
    parser.add_argument(
        "--use-genre-defaults",
        action="store_true",
        help="Use genre-specific defaults for beats_per_bar and BPM range if files are in /by_genre/<genre>/ paths.",
    )

    return parser.parse_args()


def setup_logging(quiet: bool) -> None:
    """Configure logging level according to *quiet* flag."""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> None:  # noqa: D401
    """CLI entry point."""
    args = parse_args()
    setup_logging(args.quiet)

    # Prepare directory path
    input_dir = pathlib.Path(args.directory)
    if not input_dir.is_dir():
        logging.error(f"Directory not found: {input_dir}")
        sys.exit(1)

    # Prepare detector kwargs (start with an empty dict and add valid args)
    detector_kwargs: Dict[str, Any] = {}
    if args.min_bpm is not None:
        detector_kwargs["min_bpm"] = args.min_bpm
    if args.max_bpm is not None:
        detector_kwargs["max_bpm"] = args.max_bpm

    # Prepare Beats constructor args
    beats_args: Dict[str, Any] = {
        "tolerance_percent": args.tolerance,
        "min_measures": args.min_measures,
    }
    if args.beats_per_bar is not None:
        beats_args["beats_per_bar"] = args.beats_per_bar

    # Conditionally instantiate GenreDB if genre defaults are enabled
    genre_db = None
    if args.use_genre_defaults:
        logging.info("Using genre-specific defaults for files in genre-specific directories")
        genre_db = GenreDB()

    # Call process_batch with either the GenreDB instance or None
    results = process_batch(
        directory_path=input_dir,
        detector_name=args.detector_name,
        beats_args=beats_args,
        detector_kwargs=detector_kwargs,
        no_progress=args.no_progress,
        genre_db=genre_db
    )

    # Summarize results
    success_count = sum(1 for _, beats in results if beats is not None)
    total_count = len(results)
    failure_count = total_count - success_count

    if total_count > 0:
        success_percent = (success_count / total_count) * 100
        logging.info(
            f"Processed {total_count} files: "
            f"{success_count} successful ({success_percent:.1f}%), "
            f"{failure_count} failed."
        )
    else:
        logging.warning("No audio files were processed.")


if __name__ == "__main__":
    main() 