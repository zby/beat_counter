#!/usr/bin/env python3
"""Batch processing tool for the Beat Detection Web App.

This script allows you to run both beat detection and video generation on
multiple files at once. It can find files in a directory, process them,
and wait for each task to complete before starting the next one.
"""

import argparse
import os
import pathlib
import sys
import time
import uuid
from typing import List, Optional, Tuple

# Add the parent directory to the Python path so we can import from beat_detection
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from beat_detection.utils.constants import SUPPORTED_AUDIO_EXTENSIONS
from web_app.config import Config, StorageConfig
from web_app.storage import FileMetadataStorage

# Import the tools directly to reuse their functionality
from run_beat_detection import run_beat_detection
from run_video_generation import run_video_generation

# Directory structure constants
DEFAULT_UPLOAD_DIR = pathlib.Path(__file__).parent.parent / "web_app" / "uploads"


def find_upload_directories(upload_dir: pathlib.Path) -> List[Tuple[str, pathlib.Path]]:
    """
    Find all upload directories containing audio files.

    Args:
        upload_dir: Root directory containing upload directories

    Returns:
        List of tuples (file_id, audio_file_path)
    """
    result = []

    # Walk through the upload directory to find all subdirectories
    for item in upload_dir.iterdir():
        if not item.is_dir():
            continue

        file_id = item.name

        # Look for audio files in this directory
        audio_path = None
        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            potential_path = item / f"original{ext}"
            if potential_path.exists():
                audio_path = potential_path
                break

        if audio_path:
            result.append((file_id, audio_path))

    return result


def process_file(
    file_id: str,
    upload_dir: pathlib.Path,
    wait_for_beat_detection: bool,
    wait_for_video: bool,
    skip_existing: bool,
) -> bool:
    """
    Process a single file with beat detection and optionally video generation.

    Args:
        file_id: Unique identifier for the file
        upload_dir: Directory containing upload subdirectories
        wait_for_beat_detection: Whether to wait for beat detection to complete
        wait_for_video: Whether to wait for video generation to complete
        skip_existing: Whether to skip files that already have beat or video files

    Returns:
        True if processing was successful, False otherwise
    """
    # Load configuration
    config = Config.from_env()

    # Create storage with configuration
    storage = FileMetadataStorage(config.storage)

    # Check for existing beat file
    beats_file = storage.get_beats_file_path(file_id)
    if skip_existing and beats_file.exists():
        print(f"Skipping beat detection for {file_id} - beats file already exists")
    else:
        try:
            print(f"\n{'='*80}\nProcessing beat detection for {file_id}\n{'='*80}")
            run_beat_detection(file_id, upload_dir, wait_for_beat_detection)

            # Wait for the beat file to appear if we're waiting for beat detection
            if wait_for_beat_detection and not beats_file.exists():
                max_attempts = 10
                print(f"Waiting for beats file to be created: {beats_file}")
                for i in range(max_attempts):
                    if beats_file.exists():
                        print(f"Beats file found after {i+1} attempts!")
                        break
                    print(
                        f"Attempt {i+1}/{max_attempts}: Beats file not found, waiting..."
                    )
                    time.sleep(2)  # Wait for 2 seconds before checking again
        except Exception as e:
            print(f"Error in beat detection for {file_id}: {str(e)}")
            return False

    # Check for existing video file
    video_file = storage.get_video_file_path(file_id)
    if skip_existing and video_file.exists():
        print(f"Skipping video generation for {file_id} - video file already exists")
    else:
        try:
            # Only proceed with video generation if beats file exists
            if beats_file.exists():
                print(
                    f"\n{'='*80}\nProcessing video generation for {file_id}\n{'='*80}"
                )
                run_video_generation(file_id, upload_dir, wait_for_video)
            else:
                print(f"Skipping video generation for {file_id} - no beats file found")
        except Exception as e:
            print(f"Error in video generation for {file_id}: {str(e)}")
            # Don't return False here, consider partial success if beat detection worked

    return True


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Process multiple files with beat detection and video generation"
    )
    parser.add_argument(
        "--upload-dir",
        "-d",
        help=f"Directory containing upload subdirectories (default: {DEFAULT_UPLOAD_DIR})",
    )
    parser.add_argument(
        "--wait-beats",
        "-wb",
        action="store_true",
        help="Wait for beat detection tasks to complete before proceeding",
    )
    parser.add_argument(
        "--wait-video",
        "-wv",
        action="store_true",
        help="Wait for video generation tasks to complete before proceeding",
    )
    parser.add_argument(
        "--skip-existing",
        "-s",
        action="store_true",
        help="Skip files that already have beat or video files",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="Limit the number of files to process (0 = no limit)",
    )

    args = parser.parse_args()

    try:
        # Determine the upload directory
        upload_dir = (
            pathlib.Path(args.upload_dir) if args.upload_dir else DEFAULT_UPLOAD_DIR
        )

        # Find all upload directories with audio files
        print(f"Scanning {upload_dir} for upload directories...")
        file_ids = find_upload_directories(upload_dir)

        if not file_ids:
            print(f"No files found in {upload_dir}")
            sys.exit(0)

        print(f"Found {len(file_ids)} files to process")

        # Apply limit if specified
        if args.limit > 0:
            file_ids = file_ids[: args.limit]
            print(f"Limited to {len(file_ids)} files")

        # Process each file
        success_count = 0
        for idx, (file_id, audio_path) in enumerate(file_ids):
            print(
                f"\nProcessing file {idx+1}/{len(file_ids)}: {file_id} ({audio_path})"
            )
            success = process_file(
                file_id,
                upload_dir,
                args.wait_beats,
                args.wait_video,
                args.skip_existing,
            )
            if success:
                success_count += 1

        # Print summary
        print(f"\n{'='*80}")
        print(f"Processing complete: {success_count}/{len(file_ids)} files successful")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
