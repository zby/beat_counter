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

Genre-based defaults:
--------------------
If the audio file path contains "/by_genre/<genre>/", the script can automatically
use genre-specific defaults for beats_per_bar and BPM range. This behavior is
enabled with the --use-genre-defaults flag.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from beat_detection.core import extract_beats
from beat_detection.genre_db import GenreDB, parse_genre_from_path

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
    
    # beat_this specific options
    parser.add_argument(
        "--use-dbn",
        action="store_true",
        help="Enable Dynamic Bayesian Network for the beat_this detector.",
    )
    
    # New: Genre-based defaults option
    parser.add_argument(
        "--use-genre-defaults",
        action="store_true",
        help="Use genre-specific defaults for beats_per_bar and BPM range if the audio file is in a /by_genre/<genre>/ path.",
    )
    
    # New: Explicit genre option
    parser.add_argument(
        "--genre",
        type=str,
        default=None,
        help="Explicitly specify a genre to use for defaults instead of inferring from path.",
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

    # Prepare base arguments for extract_beats
    detector_kwargs: Dict[str, Any] = {
        "min_bpm": args.min_bpm,
        "max_bpm": args.max_bpm,
    }
    
    # Add DBN argument for beat_this detector
    if args.algorithm == "beat_this":
        detector_kwargs["use_dbn"] = args.use_dbn
    
    beats_constructor_args: Dict[str, Any] = {
        "beats_per_bar": args.beats_per_bar,
        "tolerance_percent": args.tolerance,
        "min_measures": args.min_measures,
    }
    
    # Check for genre-based defaults if enabled
    if args.use_genre_defaults or args.genre:
        genre = None
        
        # Try to get genre from explicit argument first
        if args.genre:
            genre = args.genre
            logging.info(f"Using explicitly provided genre: {genre}")
        else:
            # Try to extract genre from path
            try:
                genre = parse_genre_from_path(args.audio_file)
                logging.info(f"Detected genre from path: {genre}")
            except ValueError as e:
                logging.warning(f"Could not infer genre from path: {e}")
        
        # If we have a genre, apply genre-specific defaults
        if genre:
            genre_db = GenreDB()
            
            # Apply genre defaults to detector kwargs (min_bpm, max_bpm, beats_per_bar)
            detector_kwargs = genre_db.detector_kwargs_for_genre(genre, existing=detector_kwargs)
            
            # Apply genre defaults to Beats constructor args (beats_per_bar)
            beats_constructor_args = genre_db.beats_kwargs_for_genre(genre, existing=beats_constructor_args)
            
            logging.info(f"Applied genre-based defaults for '{genre}':")
            logging.info(f"  Detector settings: min_bpm={detector_kwargs.get('min_bpm')}, max_bpm={detector_kwargs.get('max_bpm')}")
            logging.info(f"  Beats settings: beats_per_bar={beats_constructor_args.get('beats_per_bar')}")
    
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