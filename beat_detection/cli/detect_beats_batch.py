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

from beat_detection.core.factory import process_batch
from beat_detection.core.beats import Beats
from beat_detection.genre_db import GenreDB, parse_genre_from_path


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


def process_batch_with_genre_defaults(
    directory_path: pathlib.Path,
    algorithm: str,
    beats_args: Dict[str, Any],
    detector_kwargs: Dict[str, Any],
    no_progress: bool = False,
) -> List[Tuple[str, Optional[Beats]]]:
    """
    Enhanced version of process_batch that applies genre-specific defaults.
    
    For each audio file, it checks if it's in a path with /by_genre/<genre>/
    and applies the appropriate genre defaults before calling extract_beats.
    
    Parameters
    ----------
    directory_path : pathlib.Path
        Root directory to scan for audio files
    algorithm : str
        Beat detection algorithm to use
    beats_args : Dict[str, Any]
        Base arguments for Beats constructor
    detector_kwargs : Dict[str, Any]
        Base arguments for beat detector
    no_progress : bool, optional
        Whether to disable progress bar, by default False
        
    Returns
    -------
    List[Tuple[str, Optional[Beats]]]
        List of tuples with file path and resulting Beats object (or None if failed)
    """
    # Pre-initialize the GenreDB to avoid re-loading CSV for each file
    genre_db = GenreDB()
    
    from beat_detection.core.factory import find_audio_files, extract_beats
    from tqdm import tqdm
    
    if not directory_path.is_dir():
        logging.error("Directory not found: %s", directory_path)
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    audio_files = find_audio_files(directory_path)
    if not audio_files:
        logging.warning("No audio files found in %s", directory_path)
        return [] # Return empty list if no files found

    logging.info(f"Found {len(audio_files)} audio files to process in {directory_path}")
    
    pbar: Optional[tqdm] = None
    if not no_progress:
        pbar = tqdm(audio_files, desc="Processing files", unit="file", ncols=100)
        file_iterator = pbar
    else:
        file_iterator = audio_files
    
    results: List[Tuple[str, Optional[Beats]]] = []
    
    for audio_file in file_iterator:
        # Use relative path for reporting, but full path for processing
        relative_path_str = str(audio_file.relative_to(directory_path))
        full_path_str = str(audio_file)
        
        if pbar:
            pbar.set_description(f"Processing {audio_file.name}")
        else:
            logging.info(f"Processing {relative_path_str}...")
        
        # For each file, start with base arguments
        file_beats_args = beats_args.copy()
        file_detector_kwargs = detector_kwargs.copy()
        
        # Try to get genre from path
        try:
            genre = parse_genre_from_path(full_path_str)
            logging.info(f"Detected genre from path for {audio_file.name}: {genre}")
            
            # Apply genre defaults to detector kwargs and beats args
            file_detector_kwargs = genre_db.detector_kwargs_for_genre(genre, existing=file_detector_kwargs)
            file_beats_args = genre_db.beats_kwargs_for_genre(genre, existing=file_beats_args)
            
            logging.info(f"Applied genre defaults for '{genre}' to {audio_file.name}")
        except ValueError:
            # No genre in path, use base arguments
            logging.debug(f"No genre detected in path for {audio_file.name}, using base arguments")
        
        # Process the file
        try:
            beats_obj = extract_beats(
                audio_file_path=full_path_str,
                output_path=None,  # Let extract_beats handle default output
                algorithm=algorithm,
                beats_args=file_beats_args,
                **file_detector_kwargs,
            )
            results.append((relative_path_str, beats_obj))
        except Exception as e:
            logging.error(f"Failed to process {relative_path_str}: {e}")
            results.append((relative_path_str, None))
    
    if pbar:
        pbar.close()
    
    return results


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

    # Choose processing method based on whether genre defaults are enabled
    if args.use_genre_defaults:
        logging.info("Genre-based defaults enabled. Will check paths for genre information.")
        results = process_batch_with_genre_defaults(
            directory_path=directory_path,
            algorithm=args.algorithm,
            beats_args=beats_constructor_args,
            detector_kwargs=detector_kwargs,
            no_progress=args.no_progress,
        )
    else:
        # Standard processing without genre detection
        results = process_batch(
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