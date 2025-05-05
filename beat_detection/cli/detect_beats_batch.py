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

The script intentionally *does not* expose an "output directory" option – the
output always lives alongside the source audio.  This matches the project's
fail-fast philosophy by making storage layout explicit and predictable.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from typing import List, Tuple, Optional, Dict, Any

from beat_detection.core.factory import process_batch
from beat_detection.core.beats import Beats


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="detect-beats-batch",
        description="Detect beats for all audio files in a directory tree.",
    )

    parser.add_argument(
        "directory",
        nargs="?",
        default="data/input",
        help="Directory to scan recursively for audio files (default: data/input).",
    )

    # Algorithm selection
    parser.add_argument(
        "--algorithm",
        type=str,
        default="madmom",
        choices=["madmom", "beat_this"],
        help="Beat detection algorithm to use.",
    )

    # Beat-detection parameters
    parser.add_argument("--min-bpm", type=int, default=60, help="Minimum BPM.")
    parser.add_argument("--max-bpm", type=int, default=240, help="Maximum BPM.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=10.0,
        help="Interval tolerance percentage.",
    )
    parser.add_argument(
        "--beats-per-bar",
        type=int,
        default=None,
        help="Fix the time-signature numerator. When omitted, it will be inferred from the detected beats.",
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

    return parser.parse_args()


def setup_logging(quiet: bool) -> None:
    """Configure logging level according to *quiet* flag."""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> None:  # noqa: D401
    args = parse_args()
    setup_logging(quiet=args.quiet)

    directory_path = pathlib.Path(args.directory)
    if not directory_path.exists():
        logging.error("Directory not found: %s", directory_path)
        sys.exit(1)

    # Detector/Beats arguments (prepared once)
    detector_kwargs: Dict[str, Any] = {
        "min_bpm": args.min_bpm,
        "max_bpm": args.max_bpm,
    }
    beats_constructor_args: Dict[str, Any] = {
        "beats_per_bar": args.beats_per_bar,
        "tolerance_percent": args.tolerance,
        "min_measures": args.min_measures,
    }

    # Call the centralized batch processing function
    # It handles finding files, iteration, progress bar, calling extract_beats, and error handling per file
    results: List[Tuple[str, Optional[Beats]]] = process_batch(
        directory_path=directory_path,
        algorithm=args.algorithm,
        beats_args=beats_constructor_args,
        detector_kwargs=detector_kwargs,
        no_progress=args.no_progress,
    )

    # Summary
    successful = [r for r in results if r[1] is not None]
    failed = [r for r in results if r[1] is None]

    if not args.quiet:
        print("\n--- Batch Processing Summary ---")
        print(f"Total files attempted: {len(results)}")
        print(f"Successful detections: {len(successful)}")
        print(f"Failed detections: {len(failed)}")
        if failed:
            print("\nFailed files:")
            for fname, _ in failed:
                print(f"  - {fname}")


if __name__ == "__main__":
    main() 