#!/usr/bin/env python3
"""Video generation tool for the Beat Detection Web App.

This script allows you to run video generation on files with detected beats
directly from the command line. It can process a file by its ID or by its file path.
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
from web_app.tasks import generate_video_task
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


def run_video_generation(file_id: str, upload_dir: pathlib.Path = DEFAULT_UPLOAD_DIR, 
                        wait: bool = False) -> Optional[dict]:
    """
    Run video generation task on a file with detected beats.
    
    Args:
        file_id: Unique identifier for the file
        upload_dir: Directory containing upload subdirectories
        wait: Whether to wait for the task to complete
        
    Returns:
        Task result if wait=True, None otherwise
    """
    # Create a storage instance to get standardized paths
    storage = FileMetadataStorage(str(upload_dir))
    
    # Get the beats file path
    beats_file = storage.get_beats_file_path(file_id)
    
    # Check if beats file exists
    if not beats_file.exists():
        raise FileNotFoundError(
            f"No beats file found for file_id: {file_id}. "
            f"Please run beat detection first."
        )
    
    # Get the audio file path
    job_dir = upload_dir / file_id
    
    # Check if the directory exists
    if not job_dir.exists():
        raise FileNotFoundError(f"Directory not found for file_id: {file_id}")
        
    # Find audio file in the job directory
    audio_path = None
    for ext in SUPPORTED_AUDIO_EXTENSIONS:
        potential_path = job_dir / f"original{ext}"
        if potential_path.exists():
            audio_path = str(potential_path.resolve())
            break
    
    # If no audio file found, raise an error
    if not audio_path:
        raise FileNotFoundError(f"No audio file found in directory for file_id: {file_id}")
    
    print(f"Running video generation for file ID: {file_id}")
    print(f"Audio file: {audio_path}")
    print(f"Beats file: {beats_file}")
    
    # Run the task with only the file_id
    task = generate_video_task.delay(file_id)
    print(f"Task ID: {task.id}")
    
    if wait:
        print("Waiting for task to complete...")
        try:
            result = task.get()  # This will wait for the task to complete
            print("\nTask completed!")
            print("\nResult:")
            print(result)
            
            # Get the path to the generated video
            video_file = storage.get_video_file_path(file_id)
            if video_file.exists():
                print(f"\nVideo file created: {video_file}")
            
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
        description="Run video generation on a file with detected beats by ID or path"
    )
    parser.add_argument(
        "file", 
        help="File ID or path to the file/directory containing the audio and beats files"
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
    parser.add_argument(
        "--after-beat-detection", "-a",
        action="store_true",
        help="Run this immediately after beat detection (checks if beat file exists first)"
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
        
        # Check if beats file exists if we're running after beat detection
        if args.after_beat_detection:
            storage = FileMetadataStorage(str(upload_dir))
            beats_file = storage.get_beats_file_path(file_id)
            max_attempts = 10
            
            if not beats_file.exists():
                print(f"Waiting for beats file to be created: {beats_file}")
                import time
                for i in range(max_attempts):
                    if beats_file.exists():
                        print(f"Beats file found after {i+1} attempts!")
                        break
                    print(f"Attempt {i+1}/{max_attempts}: Beats file not found, waiting...")
                    time.sleep(2)  # Wait for 2 seconds before checking again
                
                if not beats_file.exists():
                    print(f"Beats file not found after {max_attempts} attempts. "
                          f"Please check that beat detection completed successfully.")
                    sys.exit(1)
        
        # Run the video generation
        run_video_generation(file_id, upload_dir, args.wait)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main() 