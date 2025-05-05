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
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from beat_detection.core.factory import extract_beats
from beat_detection.core.beats import Beats

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

    audio_path = Path(args.audio_file)
    if not audio_path.is_file():
        logging.error("Audio file not found: %s", audio_path)
        sys.exit(1)

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
    
    # Determine output path (needed before calling extract_beats)
    output_path_str: Optional[str] = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path_str = str(output_path)

    try:
        logging.info(f"Starting beat detection for {audio_path} using {args.algorithm}...")
        # Call extract_beats which handles detector creation, detection, Beats creation, and saving
        beats_obj = extract_beats(
            audio_file_path=str(audio_path),
            output_path=output_path_str,
            algorithm=args.algorithm,
            beats_args=beats_constructor_args,
            **detector_kwargs
        )

        # Get the actual path used for saving (could be default)
        final_output_path = Path(output_path_str if output_path_str else str(audio_path.with_suffix(".beats")))

        logging.info(
            f"Successfully processed {audio_path}. Effective beats_per_bar: {beats_obj.beats_per_bar}. Beats saved to {final_output_path}."
        )

    except Exception as exc:  # fail fast but print cause
        logging.exception("Beat detection failed for %s: %s", audio_path, exc)
        sys.exit(1)


if __name__ == "__main__":
    main() 