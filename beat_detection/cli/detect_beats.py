#!/usr/bin/env python3
"""
Detect beats in a single audio file.

This CLI takes one required positional argument – the path to an audio file –
and runs the beat-detection pipeline.  By default it prints the resulting beat
structure as JSON on stdout.  An optional ``-o/--output`` flag allows writing
those results into a ``.beats`` file instead.

Why a dedicated script?
-----------------------
Splitting the single-file and batch use-cases keeps each interface minimal and
eliminates several flags that only make sense in batch mode (e.g. progress bar
or output-directory layouts).  This aligns with the project's *fail-fast* rule:
less configurability means fewer silent mis-configurations.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from beat_detection.core.factory import extract_beats

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="detect-beats-file",
        description=(
            "Detect beats in a single audio file and save them to a .beats JSON file. "
            "By default, the output is saved next to the audio file with a .beats extension."
        ),
    )

    # Required audio file path
    parser.add_argument("audio_file", help="Path to the audio file to analyse.")

    # Optional output path
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Path to save the output .beats file. Defaults to <audio_file>.beats",
    )

    # Algorithm selection
    parser.add_argument(
        "--algorithm",
        type=str,
        default="madmom",
        choices=["madmom", "beat_this"],
        help="Beat detection algorithm to use.",
    )
    
    # Detection parameters
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

    return parser.parse_args()


def setup_logging() -> None:
    """Configure logging to STDERR at INFO level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> None:  # noqa: D401 – simple imperative main
    args = parse_args()
    setup_logging()

    # Prepare arguments for extract_beats
    detector_kwargs = {
        "min_bpm": args.min_bpm,
        "max_bpm": args.max_bpm,
    }
    beats_constructor_args = {
        "beats_per_bar": args.beats_per_bar,
        "tolerance_percent": args.tolerance,
        "min_measures": args.min_measures,
    }
    
    # Call extract_beats which now handles logging, file checks, and dir creation
    extract_beats(
        audio_file_path=args.audio_file,
        output_path=args.output,
        algorithm=args.algorithm,
        beats_args=beats_constructor_args,
        **detector_kwargs
    )

if __name__ == "__main__":
    main() 