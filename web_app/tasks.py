"""
Celery tasks for beat detection and video generation.

This module contains the Celery tasks for handling long-running operations
like beat detection and video generation.

Result Format:
-------------
All tasks return results in a consistent format:
- Progress information is always in a 'progress' dictionary with 'status' and 'percent'
- No top-level 'status' field is used (to avoid redundancy with Celery task state)
- Error information is captured in stdout/stderr in the output dictionaries
"""

# Standard library imports
import io
import os
import pathlib
import sys
from typing import Any, Dict, Callable, Tuple, List, Optional

# Third-party imports
from celery import states
from celery.utils.log import get_task_logger

# Local imports
from beat_detection.core.detector import BeatDetector
from beat_detection.core.video import BeatVideoGenerator
from beat_detection.utils import reporting
from beat_detection.utils.beat_file import load_beat_data
from web_app.celery_app import app

# Set up task logger
logger = get_task_logger(__name__)

# Configure Celery to log to a file instead of stdout/stderr
# This is done via environment variables that Celery will read
log_file = str(pathlib.Path(__file__).parent.absolute() / 'celery.log')
os.environ['CELERY_LOG_FILE'] = log_file

# Configure logging to handle I/O errors gracefully
class SafeLogHandler:
    """A wrapper for logger that catches I/O errors."""
    
    @staticmethod
    def safe_log(log_func, message, *args, **kwargs):
        """Log a message safely, catching any I/O errors."""
        try:
            log_func(message, *args, **kwargs)
        except OSError:
            # Silently ignore I/O errors during logging
            pass
        except Exception as e:
            # For other exceptions, try to log them but don't raise
            try:
                print(f"Logging error: {str(e)}")
            except Exception:  # Changed from bare except
                pass

# Create safe logging functions
def safe_info(message, *args, **kwargs):
    """Log an info message safely."""
    SafeLogHandler.safe_log(logger.info, message, *args, **kwargs)
    
def safe_error(message, *args, **kwargs):
    """Log an error message safely."""
    SafeLogHandler.safe_log(logger.error, message, *args, **kwargs)
    
def safe_warning(message, *args, **kwargs):
    """Log a warning message safely."""
    SafeLogHandler.safe_log(logger.warning, message, *args, **kwargs)
    
def safe_debug(message, *args, **kwargs):
    """Log a debug message safely."""
    SafeLogHandler.safe_log(logger.debug, message, *args, **kwargs)

# We'll use Celery's built-in task state management instead of a separate metadata store

# Maximum size for log output (10KB)
MAX_LOG_SIZE = 10000

def truncate_output(output: str, max_size: int = MAX_LOG_SIZE) -> str:
    """Truncate output to prevent memory issues.
    
    Args:
        output: The output string to truncate
        max_size: Maximum size in characters (default: 10KB)
    
    Returns
    -------
        Truncated output string
    """
    if len(output) > max_size:
        return "[...truncated...]\n" + output[-max_size:]
    return output

def safe_print(message: str) -> None:
    """Print a message, ignoring I/O errors."""
    try:
        print(message)
    except OSError:
        pass

class IOCapture:
    """Context manager for capturing stdout and stderr."""
    
    def __init__(self):
        self.stdout_capture = io.StringIO()
        self.stderr_capture = io.StringIO()
        self.original_stdout = None
        self.original_stderr = None
        
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.stdout_capture
        sys.stderr = self.stderr_capture
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
    def get_output(self) -> Tuple[str, str]:
        """Get the captured stdout and stderr, truncated if necessary."""
        stdout = truncate_output(self.stdout_capture.getvalue())
        stderr = truncate_output(self.stderr_capture.getvalue())
        return stdout, stderr
    
    def write_stderr(self, message: str) -> None:
        """Write a message to stderr capture."""
        self.stderr_capture.write(message + "\n")

def create_progress_updater(celery_task, task_info: dict, output_key: str):
    """Create a progress update function for a task.
    
    Args:
        celery_task: The Celery task instance
        task_info: Dictionary with task information
        output_key: Key for the output dictionary in the task metadata
        
    Returns:
        A function that can be called to update progress
    """
    def update_progress(status: str, progress: float) -> None:
        try:
            # Calculate progress percentage
            percent = progress * 100
            
            # Get current stdout and stderr from celery_task
            if hasattr(celery_task, '_io_capture'):
                current_stdout, current_stderr = celery_task._io_capture.get_output()
            else:
                current_stdout, current_stderr = "", ""
            
            # Update task state with comprehensive metadata
            update_meta = task_info.copy()
            update_meta.update({
                'progress': {
                    'status': status,
                    'percent': percent
                },
                output_key: {
                    'stdout': current_stdout,
                    'stderr': current_stderr
                }
            })
            
            celery_task.update_state(state=states.STARTED, meta=update_meta)
            
            # Log progress using safe logging
            task_type = output_key.replace('_output', '').upper()
            safe_info(f"{task_type} progress: {status} - {percent:.1f}%")
            safe_print(f"{task_type}: {status} - {percent:.1f}%")
            
        except Exception as e:
            # If there's an error updating progress, log it but don't fail the task
            safe_error(f"Error updating progress: {str(e)}")
            safe_print(f"Error updating progress: {str(e)}")
    
    return update_progress

@app.task(
    bind=True,
    name='detect_beats_task',
    queue='beat_detection'
)
def detect_beats_task(
    self,
    file_id: str,
    file_path: str
) -> Dict[str, Any]:
    """Celery task for detecting beats in an audio file.
    
    Args:
        file_id: Unique identifier for the file
        file_path: Path to the audio file
    
    Returns
    -------
    Dictionary with task results
    """
    # Setup task info
    task_info = {
        'file_id': file_id,
        'file_path': file_path,
    }
    
    # Setup IO capture
    self._io_capture = IOCapture()
    
    with self._io_capture:
        try:
            # Get the directory of the uploaded file to store output in the same place
            input_path = pathlib.Path(file_path)
            output_dir = input_path.parent
            
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
                    file_path,
                    skip_intro=True,
                    skip_ending=True
                )
                
                # Unpack results
                (
                    beat_timestamps, stats, irregular_beats, downbeats,
                    intro_end_idx, ending_start_idx, detected_meter
                ) = detection_result
                
                # Generate output file paths in the same directory as the input file
                beats_file = output_dir / f"{input_path.stem}_beats.txt"
                stats_file = output_dir / f"{input_path.stem}_beat_stats.json"
                
                # Save beat timestamps and statistics
                reporting.save_beat_timestamps(
                    beat_timestamps, beats_file, downbeats, 
                    intro_end_idx=intro_end_idx, ending_start_idx=ending_start_idx,
                    detected_meter=detected_meter
                )
                
                reporting.save_beat_statistics(
                    stats, irregular_beats, stats_file, 
                    filename=input_path.name,
                    detected_meter=detected_meter,
                    duration=beat_timestamps[-1] if len(beat_timestamps) > 0 else 0
                )
                
                # Final progress update
                update_progress("Beat detection complete", 1.0)
                
                # Get final stdout and stderr
                final_stdout, final_stderr = self._io_capture.get_output()
                
                # Return the results
                return {
                    "file_id": file_id,
                    "file_path": file_path,
                    "beats_file": str(beats_file),
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
                    "file_path": file_path,
                    "error": str(e),
                    "beat_detection_output": {
                        "stdout": current_stdout,
                        "stderr": current_stderr
                    }
                })
                
                # Re-raise the exception with proper formatting for Celery
                raise type(e)(str(e)) from e
                
        except Exception:
            # This will be caught by the outer try-except in the context manager
            raise

@app.task(
    bind=True,
    name='generate_video_task',
    queue='video_generation'
)
def generate_video_task(
    self,
    file_id: str,
    file_path: str,
    beats_file: str
) -> Dict[str, Any]:
    """Celery task for generating a beat visualization video.
    
    Args:
        file_id: Unique identifier for the file
        file_path: Path to the audio file
        beats_file: Path to the beats file
    
    Returns
    -------
    Dictionary with task results
    """
    # Setup task info
    task_info = {
        'file_id': file_id,
        'file_path': file_path,
        'beats_file': beats_file,
    }
    
    # Setup IO capture
    self._io_capture = IOCapture()
    
    with self._io_capture:
        try:
            # Get the directory of the uploaded file to store output in the same place
            input_path = pathlib.Path(file_path)
            output_dir = input_path.parent
            
            # Initialize task state
            update_progress = create_progress_updater(
                self, task_info, 'video_generation_output'
            )
            update_progress("Initializing video generation", 0)
            
            # Generate output video path in the same directory as the input file
            video_file = output_dir / f"{input_path.stem}_counter.mp4"
            
            # Load beat data
            update_progress("Loading beat data", 0.1)  # 10%
            
            try:
                # The load_beat_data function returns a tuple directly
                (
                    beat_timestamps, downbeats, intro_end_idx,
                    ending_start_idx, detected_meter
                ) = load_beat_data(beats_file)
                
                # Update progress
                update_progress("Preparing video generation", 0.3)  # 30%
                
                # Create a synchronous wrapper for the video progress callback
                def sync_video_progress_callback(status, progress):
                    # Calculate adjusted progress (50% to 90% range)
                    adjusted_progress = 0.5 + (progress * 0.4)  # Map 0-1 to 0.5-0.9
                    update_progress(status, adjusted_progress)
                
                # Generate video
                try:
                    # Create a video generator instance
                    video_generator = BeatVideoGenerator()
                    
                    # Generate the counter video
                    success = video_generator.create_counter_video(
                        audio_file=input_path,
                        output_file=video_file,
                        beat_timestamps=beat_timestamps,
                        downbeats=downbeats,
                        meter=detected_meter,
                        progress_callback=sync_video_progress_callback
                    )
                    
                    # Update progress to finalizing
                    update_progress("Finalizing video", 0.9)  # 90%
                    
                    # Check if video was created despite warnings
                    if success or video_file.exists():
                        # Final progress update
                        update_progress("Video generation complete", 1.0)  # 100%

                        # Get final stdout and stderr
                        final_stdout, final_stderr = self._io_capture.get_output()

                        # Log success
                        safe_info(f"Video generation complete: {video_file}")

                        # Return success result
                        return {
                            'file_id': file_id,
                            'file_path': file_path,
                            'beats_file': beats_file,
                            'video_file': str(video_file),
                            'progress': {
                                'status': 'Video generation complete',
                                'percent': 100
                            },
                            'video_generation_output': {
                                'stdout': final_stdout,
                                'stderr': final_stderr
                            }
                        }

                    else:
                        # Get final stdout and stderr
                        final_stdout, final_stderr = self._io_capture.get_output()

                        # Log success with warning
                        warning_msg = "Video generation completed with warnings. Video file not found."
                        safe_warning(warning_msg)

                        # Return warning result
                        return {
                            'file_id': file_id,
                            'file_path': file_path,
                            'beats_file': beats_file,
                            'video_file': str(video_file),
                            'warning': warning_msg,
                            'progress': {
                                'status': 'Video generation completed with warnings',
                                'percent': 100
                            },
                            'video_generation_output': {
                                'stdout': final_stdout,
                                'stderr': final_stderr
                            }
                        }

                except Exception as video_error:
                    # Check if the video file was created despite the error
                    if video_file.exists() and video_file.stat().st_size > 0:
                        warning_msg = (
                            f"Warning during video generation: {video_error}, "
                            "but video file was created successfully"
                        )
                        safe_print(warning_msg)
                        self._io_capture.stdout_capture.write(warning_msg + "\n")
                        
                        # Make sure the final progress update is sent
                        update_progress(
                            "Video generation complete (with warnings)",
                            1.0  # 100%
                        )
                        
                        # Get final stdout and stderr
                        final_stdout, final_stderr = self._io_capture.get_output()
                            
                        # Log success with warning
                        safe_info(f"Successfully generated video for file {file_id} (with warning)")
                        safe_print(f"Successfully generated video for file {file_id} (with warning)")
                        
                        # Return success with warning
                        return {
                            'file_id': file_id,
                            'file_path': file_path,
                            'beats_file': beats_file,
                            'video_file': str(video_file),
                            'warning': str(video_error),
                            'progress': {
                                'status': 'Video generation complete (with warnings)',
                                'percent': 100
                            },
                            'video_generation_output': {
                                'stdout': final_stdout,
                                'stderr': final_stderr
                            }
                        }
                    else:
                        # Re-raise the exception if no video was created
                        raise
                        
            except Exception as e:
                # Log the error
                error_msg = f"Video generation error: {str(e)}"
                safe_error(error_msg)
                self._io_capture.write_stderr(error_msg)
                
                # Get current stdout and stderr
                current_stdout, current_stderr = self._io_capture.get_output()
                
                # Update task state with error information
                self.update_state(state=states.FAILURE, meta={
                    "file_id": file_id,
                    "file_path": file_path,
                    "beats_file": beats_file,
                    "error": str(e),
                    "progress": {
                        "status": "Error: " + str(e),
                        "percent": 0
                    },
                    "video_generation_output": {
                        "stdout": current_stdout,
                        "stderr": current_stderr
                    }
                })
                
                # Log error
                safe_error(f"Video generation error for file {file_id}: {str(e)}")
                safe_print(f"Video generation error for file {file_id}: {str(e)}")
                
                # Re-raise the exception
                raise
                
        except Exception as e:
            # Get final stdout and stderr
            final_stdout, final_stderr = self._io_capture.get_output()
            
            # Return failure result with comprehensive information
            return {
                'file_id': file_id,
                'file_path': file_path,
                'beats_file': beats_file,
                'error': str(e),
                'progress': {
                    'status': 'Error: ' + str(e),
                    'percent': 0
                },
                'video_generation_output': {
                    'stdout': final_stdout,
                    'stderr': final_stderr
                }
            }
