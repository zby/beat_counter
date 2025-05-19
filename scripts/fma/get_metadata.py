#!/usr/bin/env python3
"""Script to get file status for a given file_id."""

import argparse
from pathlib import Path
from beat_counter.web_app.storage import FileMetadataStorage
from beat_counter.web_app.config import get_config
from beat_counter.web_app.app import UPLOAD_DIR


def get_file_status(file_id: str) -> None:
    """Get and display file status for the given file_id."""
    # Use the same storage configuration as the app
    config = get_config()
    storage = FileMetadataStorage(
        base_dir=str(UPLOAD_DIR),
        max_audio_duration=config.get("queue", {}).get("max_duration", 60),
    )

    try:
        metadata = storage.get_file_metadata(file_id)
        if not metadata:
            print(f"Error: No metadata found for file_id {file_id}")
            return

        print(f"\nFile Status for {file_id}:")
        print("-" * 50)

        # Display basic file information
        print(f"Original Filename: {metadata.get('original_filename', 'N/A')}")
        print(f"Status: {metadata.get('status', 'N/A')}")
        print(f"Duration: {metadata.get('duration', 'N/A')} seconds")

        # Display beat detection information if available
        if "beats" in metadata:
            print(f"\nBeat Detection:")
            print(f"Number of beats: {len(metadata['beats'])}")
            print(f"BPM: {metadata.get('bpm', 'N/A')}")

        # Display video generation status if available
        if "video_file" in metadata:
            print(f"\nVideo Generation:")
            print(f"Video file: {metadata['video_file']}")

        # Display any error messages if present
        if "error" in metadata:
            print(f"\nError:")
            print(metadata["error"])

    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Get file status for a given file_id")
    parser.add_argument("file_id", help="The file_id to get status for")

    args = parser.parse_args()

    get_file_status(args.file_id)


if __name__ == "__main__":
    main()
