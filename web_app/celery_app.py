"""
Celery configuration for the beat detection application.

This module sets up Celery for handling long-running tasks like beat detection
and video generation in a distributed manner.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any

from celery import Celery, states
from web_app.config import Config
from web_app.storage import FileMetadataStorage
from web_app.context import AppContext
from web_app.utils.task_utils import (
    IOCapture, create_progress_updater, safe_error, safe_info
)
from beat_detection.core.detector import BeatDetector
from beat_detection.core.video import BeatVideoGenerator
from beat_detection.utils import reporting
from beat_detection.utils.beat_file import load_beat_data

# Load configuration
config = Config.from_env()

# Create the Celery app with configuration from config
app = Celery(**config.celery.__dict__)

# Get max duration from config
MAX_AUDIO_DURATION = config.storage.max_audio_secs

# Configure Celery to log to a file instead of stdout/stderr
log_file = os.path.join(config.storage.upload_dir, 'celery.log')
os.environ['CELERY_LOG_FILE'] = log_file

# Initialize application context with storage
app.context = AppContext(
    storage=FileMetadataStorage(config.storage)
)

# Auto-discover tasks in the web_app package
app.autodiscover_tasks(['web_app'])

@app.task(
    bind=True,
    name='detect_beats_task',
    queue='beat_detection'
)
def detect_beats_task(
    self,
    file_id: str
) -> Dict[str, Any]:
    """Celery task for detecting beats in an audio file.
    
    Args:
        file_id: Unique identifier for the file
    
    Returns
    -------
    Dictionary with task results
    """
    # Setup task info
    task_info = {
        'file_id': file_id
    }
    
    # Setup IO capture
    self._io_capture = IOCapture()
    
    with self._io_capture:
        try:
            # Construct job directory path
            job_dir = self.app.context.storage.get_job_directory(file_id)
            
            # Create a dedicated 'tmp' subdirectory for temporary files
            tmp_dir = job_dir / "tmp"
            tmp_dir.mkdir(exist_ok=True)
            
            # Create metadata.json path
            metadata_path = job_dir / "metadata.json"
            
            # Load metadata if exists
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Get the audio file path using storage interface
            audio_file_path = self.app.context.storage.get_audio_file_path(file_id)
            if not audio_file_path.exists():
                raise FileNotFoundError(f"No audio file found for file_id: {file_id}")
            
            # Convert to string for compatibility with downstream code
            audio_path = str(audio_file_path.resolve())
            
            # Initialize task state
            update_progress = create_progress_updater(
                self, task_info, 'beat_detection_output'
            )
            update_progress("Initializing beat detection", 0)
            
            # Initialize beat detector with progress callback
            detector = BeatDetector(
                min_bpm=60,
                max_bpm=240,
                tolerance_percent=10.0,
                progress_callback=update_progress
            )
            
            # Detect beats
            try:
                # Perform beat detection
                detection_result = detector.detect_beats(
                    audio_path,
                    skip_intro=True,
                    skip_ending=True
                )
                
                # Unpack results
                (
                    beat_timestamps, stats, irregular_beats, downbeats,
                    intro_end_idx, ending_start_idx, detected_meter
                ) = detection_result
                
                # Generate standardized output file paths
                beats_file = self.app.context.storage.get_beats_file_path(file_id)
                stats_file = self.app.context.storage.get_beat_stats_file_path(file_id)
                
                # Save beat timestamps and statistics
                reporting.save_beat_timestamps(
                    beat_timestamps, beats_file, downbeats, 
                    intro_end_idx=intro_end_idx, ending_start_idx=ending_start_idx,
                    detected_meter=detected_meter
                )
                
                reporting.save_beat_statistics(
                    stats, irregular_beats, stats_file, 
                    filename=os.path.basename(audio_path),
                    detected_meter=detected_meter,
                    duration=beat_timestamps[-1] if len(beat_timestamps) > 0 else 0
                )
                
                # Update metadata with beat stats
                with open(stats_file, "r") as f:
                    beat_stats = json.load(f)
                
                # Update metadata using the storage method with proper locking
                self.app.context.storage.update_metadata(file_id, {
                    "beat_stats": beat_stats,
                    "beats_file": str(beats_file),
                    "stats_file": str(stats_file)
                })
                
                # Final progress update
                update_progress("Beat detection complete", 1.0)
                
                # Get final stdout and stderr
                final_stdout, final_stderr = self._io_capture.get_output()
                
                # Return the results
                return {
                    "file_id": file_id,
                    "file_path": audio_path,
                    "beats_file": str(beats_file),
                    "stats_file": str(stats_file),
                    "progress": {
                        "status": "Beat detection complete",
                        "percent": 100
                    },
                    "beat_detection_output": {
                        "stdout": final_stdout,
                        "stderr": final_stderr
                    }
                }
                
            except Exception as e:
                # Log the error
                error_msg = f"Beat detection error: {str(e)}"
                safe_error(error_msg)
                self._io_capture.write_stderr(error_msg)
                
                # Get current stdout and stderr
                current_stdout, current_stderr = self._io_capture.get_output()
                
                # Update task state with error information
                self.update_state(state=states.FAILURE, meta={
                    "file_id": file_id,
                    "exc_type": type(e).__name__,
                    "exc_message": str(e),
                    "exc_module": type(e).__module__,
                    "beat_detection_output": {
                        "stdout": current_stdout,
                        "stderr": current_stderr
                    }
                })
                
                # Re-raise the exception with proper Celery exception info
                raise
                
        except Exception as e:
            # Log the exception
            safe_error(f"Error in beat detection task: {str(e)}")
            
            # If we have an IO capture, get its output
            stdout, stderr = "", ""
            if hasattr(self, '_io_capture'):
                stdout, stderr = self._io_capture.get_output()
            
            # Return error information with proper Celery exception format
            return {
                "file_id": file_id,
                "exc_type": type(e).__name__,
                "exc_message": str(e),
                "exc_module": type(e).__module__,
                "beat_detection_output": {
                    "stdout": stdout,
                    "stderr": stderr
                }
            }

@app.task(
    bind=True,
    name='generate_video_task',
    queue='video_generation'
)
def generate_video_task(
    self,
    file_id: str
) -> Dict[str, Any]:
    """Celery task for generating a beat visualization video.
    
    Args:
        file_id: Unique identifier for the file
    
    Returns
    -------
    Dictionary with task results
    """
    # Setup task info
    task_info = {
        'file_id': file_id
    }
    
    # Setup IO capture
    self._io_capture = IOCapture()
    
    with self._io_capture:
        try:
            # Construct job directory path
            job_dir = self.app.context.storage.get_job_directory(file_id)
            
            # Create metadata.json path
            metadata_path = job_dir / "metadata.json"
            
            # Load metadata if exists
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Get the audio file path using storage interface
            audio_file_path = self.app.context.storage.get_audio_file_path(file_id)
            if not audio_file_path.exists():
                raise FileNotFoundError(f"No audio file found for file_id: {file_id}")
            
            # Convert to string for compatibility with downstream code
            audio_path = str(audio_file_path.resolve())
            
            # Get the beats file
            beats_file = self.app.context.storage.get_beats_file_path(file_id)
            if not beats_file.exists():
                raise FileNotFoundError(f"Beats file not found for file_id: {file_id}")
                
            # Standardized output video path
            video_output = self.app.context.storage.get_video_file_path(file_id)
            
            # MoviePy creates temporary files relative to the current working directory
            # Change the directory to the job directory to control where temp files are created
            old_cwd = os.getcwd()
            os.chdir(str(job_dir))
            safe_info(f"Changed working directory to {str(job_dir)}")
            
            # Initialize task state
            update_progress = create_progress_updater(
                self, task_info, 'video_generation_output'
            )
            update_progress("Initializing video generation", 0)
            
            # Load beat data from the text file
            beat_timestamps, downbeats, intro_end_idx, ending_start_idx, detected_meter = load_beat_data(str(beats_file))
            
            # Check if beat data is valid
            if beat_timestamps is None or len(beat_timestamps) == 0:
                raise ValueError("No beat data found")
            
            # Create video generator
            def sync_video_progress_callback(status, progress):
                update_progress(status, progress)
            
            # Generate the visualization video
            update_progress("Initializing video generation", 0.05)
            
            try:
                # Initialize video generator
                video_generator = BeatVideoGenerator(
                    progress_callback=sync_video_progress_callback
                )
                
                # Generate video
                video_file = video_generator.generate_video(
                    audio_path, 
                    beat_timestamps,
                    output_path=str(video_output),
                    downbeats=downbeats,
                    detected_meter=detected_meter
                )
                
                # Update metadata using the storage method with proper locking
                self.app.context.storage.update_metadata(file_id, {"video_file": str(video_output)})
                
                # Final progress update
                update_progress("Video generation complete", 1.0)
                
                # Get final stdout and stderr
                final_stdout, final_stderr = self._io_capture.get_output()
                
                # Return the results
                return {
                    "file_id": file_id,
                    "audio_file": audio_path,
                    "beats_file": str(beats_file),
                    "video_file": str(video_output),
                    "progress": {
                        "status": "Video generation complete",
                        "percent": 100
                    },
                    "video_generation_output": {
                        "stdout": final_stdout,
                        "stderr": final_stderr
                    }
                }
            finally:
                # Restore the original working directory if it was changed
                if 'old_cwd' in locals():
                    try:
                        os.chdir(old_cwd)
                        safe_info("Restored original working directory")
                    except Exception as e:
                        safe_error(f"Error restoring original working directory: {str(e)}")
            
        except Exception as e:
            # Log the error
            error_msg = f"Video generation error: {str(e)}"
            safe_error(error_msg)
            if hasattr(self, '_io_capture'):
                self._io_capture.write_stderr(error_msg)
            
            # Get current stdout and stderr
            stdout, stderr = "", ""
            if hasattr(self, '_io_capture'):
                stdout, stderr = self._io_capture.get_output()
            
            # Update task state with error information
            self.update_state(state=states.FAILURE, meta={
                "file_id": file_id,
                "exc_type": type(e).__name__,
                "exc_message": str(e),
                "exc_module": type(e).__module__,
                "video_generation_output": {
                    "stdout": stdout,
                    "stderr": stderr
                }
            })
            
            # Return error information
            return {
                "file_id": file_id,
                "exc_type": type(e).__name__,
                "exc_message": str(e),
                "exc_module": type(e).__module__,
                "video_generation_output": {
                    "stdout": stdout,
                    "stderr": stderr
                }
            }

# Optional: Add some debugging info
@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
