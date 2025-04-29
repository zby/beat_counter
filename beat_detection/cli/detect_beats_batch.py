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
from typing import List, Tuple, Optional

from tqdm import tqdm

from beat_detection.core.detector import BeatDetector
from beat_detection.utils.file_utils import find_audio_files
from beat_detection.utils.beat_file import save_beats
from beat_detection.utils import reporting
from beat_detection.core.beats import Beats, RawBeats


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

    # Beat-detection parameters
    parser.add_argument("--min-bpm", type=int, default=60, help="Minimum BPM.")
    parser.add_argument("--max-bpm", type=int, default=240, help="Maximum BPM.")
    parser.add_argument(
        "--tolerance", type=float, default=10.0, help="Interval tolerance percentage."
    )
    parser.add_argument(
        "--beats-per-bar",
        type=int,
        default=None,
        help="Fix the time-signature numerator.  If omitted the detector tries to infer it.",
    )
    parser.add_argument(
        "--min-measures",
        type=int,
        default=1,
        help="Minimum number of consistent measures for stable section analysis.",
    )

    # Output control
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar.")

    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    """Configure logging level according to *verbose* flag."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def process_file(audio_file: pathlib.Path, detector: BeatDetector, verbose: bool) -> Tuple[str, Optional[RawBeats]]:
    """Detect beats for *audio_file* and write them alongside the file.

    Returns a tuple ``(filename, RawBeats|None)`` where the second element is the
    RawBeats object on success or ``None`` on failure.
    """
    try:
        # Get RawBeats object (contains bpb)
        raw_beats = detector.detect_beats(str(audio_file))
        logging.debug(f"Detected raw beats for {audio_file} with bpb={raw_beats.beats_per_bar}")

        beats_file = audio_file.with_suffix(".beats")
        # Pass RawBeats object to save_beats
        save_beats(str(beats_file), raw_beats)
        if verbose:
            logging.info("Saved raw beat data to %s", beats_file)
        # Return RawBeats object on success
        return (audio_file.name, raw_beats)
    except Exception:
        logging.exception("Error processing %s", audio_file)
        # Return None on failure
        return (audio_file.name, None)


def main() -> None:  # noqa: D401
    args = parse_args()
    setup_logging(verbose=not args.quiet)

    # Progress bar setup
    pbar: Optional[tqdm] = None
    progress_callback = None

    if not args.quiet and not args.no_progress:
        pbar = tqdm(total=100, desc="Processing", unit="%", ncols=80)

        def _update(progress_value: float) -> None:
            if pbar is not None:
                pbar.n = int(progress_value * 100)
                pbar.refresh()

        progress_callback = _update

    # Configure detector
    detector = BeatDetector(
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        beats_per_bar=args.beats_per_bar,
        progress_callback=progress_callback,
    )

    directory_path = pathlib.Path(args.directory)
    if not directory_path.exists():
        logging.error("Directory not found: %s", directory_path)
        sys.exit(1)

    audio_files = find_audio_files(directory_path)
    if not audio_files:
        logging.error("No audio files found in %s", directory_path)
        sys.exit(1)

    if not args.quiet:
        print(f"Found {len(audio_files)} audio files to process")

    # Results list now holds Optional[RawBeats]
    results: List[Tuple[str, Optional[RawBeats]]] = []

    for audio_file in audio_files:
        if pbar:
            pbar.set_description(f"Processing {audio_file.name}")
            pbar.reset()
        elif not args.quiet:
            print(f"\nProcessing: {audio_file}\n" + "=" * 80)

        results.append(process_file(audio_file, detector, verbose=not args.quiet))

    if pbar:
        pbar.close()

    # Summary
    successful_raw = [r for r in results if r[1] is not None]
    failed = [r for r in results if r[1] is None]

    if not args.quiet:
        # Modify reporting if needed, or keep simple count
        print("\n--- Batch Processing Summary ---")
        print(f"Total files processed: {len(results)}")
        print(f"Successful detections: {len(successful_raw)}")
        print(f"Failed detections: {len(failed)}")
        # reporting.print_batch_summary(results) # Old reporting might expect Beats object
        if failed:
            print("\nFailed files:")
            for fname, _ in failed:
                print(f"  - {fname}")


if __name__ == "__main__":
    main() 