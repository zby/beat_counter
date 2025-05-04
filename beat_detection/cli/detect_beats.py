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

from beat_detection.core.factory import get_beat_detector
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

    # Get detector from factory – uses project defaults unless user overrides
    detector = get_beat_detector(
        algorithm=args.algorithm,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
    )

    try:
        # Get simplified RawBeats object (timestamps and counts only)
        raw_beats = detector.detect(str(audio_path))
        
        # Create Beats object with optional beats_per_bar override
        beats = Beats(
            raw_beats=raw_beats,
            beats_per_bar=args.beats_per_bar,
            tolerance_percent=args.tolerance,
            min_measures=args.min_measures,
        )
        
        logging.info(
            f"Detected beats for {audio_path} with effective beats_per_bar={beats.beats_per_bar}"
        )
    except Exception as exc:  # fail fast but print cause
        logging.exception("Beat detection failed for %s: %s", audio_path, exc)
        sys.exit(1)

    # Output handling
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = audio_path.with_suffix(".beats")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save the raw_beats object
    raw_beats.save_to_file(out_path)
    logging.info("Saved raw beats to %s", out_path)


if __name__ == "__main__":
    main() 