#!/usr/bin/env python3
"""Beat detection tool for the Beat Detection Web App.

This script allows you to run beat detection on audio files directly from the command line.
It can process a file by its ID or by its file path.
"""

import argparse
import os
import pathlib
import sys
import uuid
from typing import Optional

# Add the parent directory to the Python path so we can import from beat_detection
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from beat_detection.utils.constants import SUPPORTED_AUDIO_EXTENSIONS
from web_app.tasks import detect_beats_task
from web_app.celery_app import app
from web_app.storage import FileMetadataStorage

# Directory structure constants
DEFAULT_UPLOAD_DIR = pathlib.Path(__file__).parent.parent / "web_app" / "uploads"


def extract_file_id_from_path(file_path: str) -> str:
    """
    Extract the file ID from a file path.
    Assumes the file ID is the last part of the directory path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        The file ID
    """
    path = pathlib.Path(file_path)
    
    # If the path is a file, get its parent directory
    if path.is_file():
        path = path.parent
    
    # Return the name of the deepest directory
    return path.name


def run_beat_detection(file_id: str, upload_dir: pathlib.Path = DEFAULT_UPLOAD_DIR, 
                      wait: bool = False) -> Optional[dict]:
    """
    Run beat detection task on a file using standardized directory structure.
    
    Args:
        file_id: Unique identifier for the file
        upload_dir: Directory containing upload subdirectories
        wait: Whether to wait for the task to complete
        
    Returns:
        Task result if wait=True, None otherwise
    """
    # Create a storage instance to get standardized paths
    storage = FileMetadataStorage(str(upload_dir))
    
    # Get the audio file path using the storage
    audio_path = None
    job_dir = upload_dir / file_id
    
    # Check if the directory exists
    if not job_dir.exists():
        raise FileNotFoundError(f"Directory not found for file_id: {file_id}")
        
    # Find audio file in the job directory
    for ext in SUPPORTED_AUDIO_EXTENSIONS:
        potential_path = job_dir / f"original{ext}"
        if potential_path.exists():
            audio_path = str(potential_path.resolve())
            break
    
    # If no audio file found, raise an error
    if not audio_path:
        raise FileNotFoundError(f"No audio file found in directory for file_id: {file_id}")
    
    print(f"Running beat detection on file: {audio_path}")
    print(f"File ID: {file_id}")
    
    # Run the task with only the file_id
    task = detect_beats_task.delay(file_id)
    print(f"Task ID: {task.id}")
    
    if wait:
        print("Waiting for task to complete...")
        try:
            result = task.get()  # This will wait for the task to complete
            print("\nTask completed!")
            print("\nResult:")
            print(result)
            return result
        except Exception as e:
            print(f"Task failed: {str(e)}")
            return None
    else:
        print("\nTask started. Use the task ID to check its status.")
        return None


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run beat detection on an audio file by ID or path"
    )
    parser.add_argument(
        "file", 
        help="File ID or path to the file/directory containing the audio file"
    )
    parser.add_argument(
        "--wait", "-w", 
        action="store_true", 
        help="Wait for the task to complete and show results"
    )
    parser.add_argument(
        "--upload-dir", "-d",
        help=f"Directory containing upload subdirectories (default: {DEFAULT_UPLOAD_DIR})"
    )
    
    args = parser.parse_args()
    
    try:
        # Determine if the input is a file ID or path
        if os.path.exists(args.file):
            # It's a path, extract the file ID
            file_id = extract_file_id_from_path(args.file)
            print(f"Extracted file ID from path: {file_id}")
        else:
            # Assume it's a file ID
            file_id = args.file
            
            # Validate that it looks like a UUID
            try:
                uuid.UUID(file_id)
            except ValueError:
                print(f"Warning: '{file_id}' doesn't appear to be a valid UUID. "
                      f"This might be fine if you're using custom IDs.")
        
        # Determine the upload directory
        upload_dir = pathlib.Path(args.upload_dir) if args.upload_dir else DEFAULT_UPLOAD_DIR
        
        # Run the beat detection
        run_beat_detection(file_id, upload_dir, args.wait)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main() 