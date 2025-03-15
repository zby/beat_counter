"""
Celery tasks for beat detection and video generation.

This module contains the Celery tasks for handling long-running operations
like beat detection and video generation.
"""

import pathlib
import io
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import os

from web_app.celery_app import app
from celery import states
from celery.utils.log import get_task_logger

from beat_detection.core.detector import BeatDetector
from beat_detection.core.video import BeatVideoGenerator
from beat_detection.utils import reporting

# Set up task logger
logger = get_task_logger(__name__)

# Configure Celery to log to a file instead of stdout/stderr
# This is done via environment variables that Celery will read
os.environ['CELERY_LOG_FILE'] = str(pathlib.Path(__file__).parent.absolute() / 'celery.log')

# Configure logging to handle I/O errors gracefully
class SafeLogHandler:
    """A wrapper for logger that catches I/O errors"""
    
    @staticmethod
    def safe_log(log_func, message, *args, **kwargs):
        """Log a message safely, catching any I/O errors"""
        try:
            log_func(message, *args, **kwargs)
        except OSError:
            # Silently ignore I/O errors during logging
            pass
        except Exception as e:
            # For other exceptions, try to log them but don't raise
            try:
                print(f"Logging error: {str(e)}")
            except:
                pass

# Create safe logging functions
def safe_info(message, *args, **kwargs):
    SafeLogHandler.safe_log(logger.info, message, *args, **kwargs)
    
def safe_error(message, *args, **kwargs):
    SafeLogHandler.safe_log(logger.error, message, *args, **kwargs)
    
def safe_warning(message, *args, **kwargs):
    SafeLogHandler.safe_log(logger.warning, message, *args, **kwargs)
    
def safe_debug(message, *args, **kwargs):
    SafeLogHandler.safe_log(logger.debug, message, *args, **kwargs)

# We'll use Celery's built-in task state management instead of a separate metadata store

# Set output directory
OUTPUT_DIR = pathlib.Path(__file__).parent.absolute() / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Import the beat_file module for loading beat data
from beat_detection.utils.beat_file import load_beat_data

@app.task(bind=True, name='detect_beats_task', queue='beat_detection')
def detect_beats_task(self, file_id: str, file_path: str) -> Dict[str, Any]:
    """
    Celery task for detecting beats in an audio file.
    
    Args:
        file_id: Unique identifier for the file
        file_path: Path to the audio file
    
    Returns:
        Dictionary with task results
    """
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    # Store original stdout and stderr
    import sys
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Update task state to started with comprehensive metadata
    self.update_state(state=states.STARTED, meta={
        'status': 'analyzing',
        'file_id': file_id,
        'file_path': file_path,
        'progress': {
            'status': 'Initializing beat detection',
            'percent': 0
        },
        'beat_detection_output': {
            'stdout': '',
            'stderr': ''
        }
    })
    
    # Redirect stdout and stderr to our capture objects
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    try:
        # Create output directory
        output_path = OUTPUT_DIR / file_id
        output_path.mkdir(exist_ok=True)
        
        # Create a progress callback function
        def update_progress(status, progress):
            try:
                # Calculate progress percentage
                percent = progress * 100
                
                # Capture current stdout and stderr
                current_stdout = stdout_capture.getvalue()
                current_stderr = stderr_capture.getvalue()
                
                # Limit the size of stdout/stderr to prevent memory issues
                max_output_size = 10000  # 10KB limit for logs
                if len(current_stdout) > max_output_size:
                    current_stdout = "[...truncated...]\n" + current_stdout[-max_output_size:]
                if len(current_stderr) > max_output_size:
                    current_stderr = "[...truncated...]\n" + current_stderr[-max_output_size:]
                
                # Update task state with comprehensive metadata
                self.update_state(state=states.STARTED, meta={
                    'status': 'analyzing',
                    'file_id': file_id,
                    'file_path': file_path,
                    'progress': {
                        'status': status,
                        'percent': percent
                    },
                    'beat_detection_output': {
                        'stdout': current_stdout,
                        'stderr': current_stderr
                    }
                })
                
                # Log progress using safe logging
                safe_info(f"Beat detection progress: {status} - {percent:.1f}%")
                
                # Capture to stdout (this will be captured by our StringIO)
                try:
                    print(f"BEAT DETECTION: {status} - {percent:.1f}%")
                except OSError:
                    # Silently ignore I/O errors when writing to stdout
                    pass
            except Exception as e:
                # If there's an error updating progress, log it but don't fail the task
                safe_error(f"Error updating progress: {str(e)}")
                try:
                    print(f"Error updating progress: {str(e)}")
                except OSError:
                    # Silently ignore I/O errors when writing to stdout
                    pass
        
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
            detection_result = detector.detect_beats(file_path, skip_intro=True, skip_ending=True)
            
            # Unpack results
            beat_timestamps, stats, irregular_beats, downbeats, intro_end_idx, ending_start_idx, detected_meter = detection_result
            
            # Generate output file paths
            input_path = pathlib.Path(file_path)
            beats_file = output_path / f"{input_path.stem}_beats.txt"
            stats_file = output_path / f"{input_path.stem}_beat_stats.txt"
            
            # Save beat timestamps and statistics
            reporting.save_beat_timestamps(
                beat_timestamps, beats_file, downbeats, 
                intro_end_idx=intro_end_idx, ending_start_idx=ending_start_idx,
                detected_meter=detected_meter
            )
            
            reporting.save_beat_statistics(
                stats, irregular_beats, stats_file, 
                filename=input_path.name
            )
            
            # Make sure the final progress update is sent through the progress callback
            # This ensures the UI receives the final status update through the same channel
            update_progress("Beat detection complete", 1.0)
            
            # Capture final stdout and stderr
            final_stdout = stdout_capture.getvalue()
            final_stderr = stderr_capture.getvalue()
            
            # Limit the size of stdout/stderr to prevent memory issues
            max_output_size = 10000  # 10KB limit for logs
            if len(final_stdout) > max_output_size:
                final_stdout = "[...truncated...]\n" + final_stdout[-max_output_size:]
            if len(final_stderr) > max_output_size:
                final_stderr = "[...truncated...]\n" + final_stderr[-max_output_size:]
            
            # Return the results with comprehensive information
            # This will be stored in Celery's result backend and accessible via AsyncResult
            return {
                "status": "analyzed",  # Use 'analyzed' instead of 'success' for consistency
                "file_id": file_id,
                "file_path": file_path,
                "beats_file": str(beats_file),
                "stats": {
                    "bpm": stats.tempo_bpm,
                    "total_beats": len(beat_timestamps),
                    "duration": beat_timestamps[-1] if len(beat_timestamps) > 0 else 0,
                    "irregularity_percent": stats.irregularity_percent,
                    "detected_meter": detected_meter
                },
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
            stderr_capture.write(error_msg + "\n")
            
            # Capture current stdout and stderr
            current_stdout = stdout_capture.getvalue()
            current_stderr = stderr_capture.getvalue()
            
            # Limit the size of stdout/stderr to prevent memory issues
            max_output_size = 10000  # 10KB limit for logs
            if len(current_stdout) > max_output_size:
                current_stdout = "[...truncated...]\n" + current_stdout[-max_output_size:]
            if len(current_stderr) > max_output_size:
                current_stderr = "[...truncated...]\n" + current_stderr[-max_output_size:]
            
            # Update task state with error information
            self.update_state(state=states.FAILURE, meta={
                "status": "error",
                "file_id": file_id,
                "error": str(e),
                "beat_detection_output": {
                    "stdout": current_stdout,
                    "stderr": current_stderr
                }
            })
            
            # Re-raise the exception
            raise
            
    except Exception as e:
        # Capture final stdout and stderr even in case of exception
        final_stdout = stdout_capture.getvalue() if 'stdout_capture' in locals() else ''
        final_stderr = stderr_capture.getvalue() if 'stderr_capture' in locals() else ''
        
        # Limit the size of stdout/stderr to prevent memory issues
        max_output_size = 10000  # 10KB limit for logs
        if len(final_stdout) > max_output_size:
            final_stdout = "[...truncated...]\n" + final_stdout[-max_output_size:]
        if len(final_stderr) > max_output_size:
            final_stderr = "[...truncated...]\n" + final_stderr[-max_output_size:]
        
        # Return failure result with more comprehensive information
        return {
            'status': 'failure',
            'file_id': file_id,
            'file_path': file_path,
            'error': str(e),
            'progress': {
                'status': 'Failed',
                'percent': 0
            },
            'beat_detection_output': {
                'stdout': final_stdout,
                'stderr': final_stderr
            }
        }
    finally:
        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr


@app.task(bind=True, name='generate_video_task', queue='video_generation')
def generate_video_task(self, file_id: str, file_path: str, beats_file: str) -> Dict[str, Any]:
    """
    Celery task for generating a beat visualization video.
    
    Args:
        file_id: Unique identifier for the file
        file_path: Path to the audio file
        beats_file: Path to the beats file
    
    Returns:
        Dictionary with task results
    """
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    # Store original stdout and stderr
    import sys
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Update task state to started with comprehensive metadata
    self.update_state(state=states.STARTED, meta={
        'status': 'generating_video',
        'file_id': file_id,
        'file_path': file_path,
        'beats_file': beats_file,
        'progress': {
            'status': 'Initializing video generation',
            'percent': 0
        },
        'video_generation_output': {
            'stdout': '',
            'stderr': ''
        }
    })
    
    # Redirect stdout and stderr to our capture objects
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    try:
        # Create output directory
        output_path = OUTPUT_DIR / file_id
        output_path.mkdir(exist_ok=True)
        
        # Create a progress update function
        def update_progress(status, progress):
            try:
                # Calculate progress percentage
                percent = progress * 100
                
                # Capture current stdout and stderr
                current_stdout = stdout_capture.getvalue()
                current_stderr = stderr_capture.getvalue()
                
                # Limit the size of stdout/stderr to prevent memory issues
                max_output_size = 10000  # 10KB limit for logs
                if len(current_stdout) > max_output_size:
                    current_stdout = "[...truncated...]\n" + current_stdout[-max_output_size:]
                if len(current_stderr) > max_output_size:
                    current_stderr = "[...truncated...]\n" + current_stderr[-max_output_size:]
                
                # Update task state with comprehensive metadata
                self.update_state(state=states.STARTED, meta={
                    'status': 'generating_video',
                    'file_id': file_id,
                    'file_path': file_path,
                    'beats_file': beats_file,
                    'progress': {
                        'status': status,
                        'percent': percent
                    },
                    'video_generation_output': {
                        'stdout': current_stdout,
                        'stderr': current_stderr
                    }
                })
                
                # Log progress
                safe_info(f"Video generation progress: {status} - {percent:.1f}%")
                
                # Capture to stdout (this will be captured by our StringIO)
                try:
                    print(f"VIDEO GENERATION: {status} - {percent:.1f}%")
                except OSError:
                    # Silently ignore I/O errors when writing to stdout
                    pass
            except Exception as e:
                # If there's an error updating progress, log it but don't fail the task
                safe_error(f"Error updating progress: {str(e)}")
                try:
                    print(f"Error updating progress: {str(e)}")
                except OSError:
                    # Silently ignore I/O errors when writing to stdout
                    pass
        
        # Update progress
        update_progress("Loading beat data", 0.1)  # 10%
        
        # Generate output video path
        input_path = pathlib.Path(file_path)
        video_file = output_path / f"{input_path.stem}_counter.mp4"
        
        # Load beat data
        try:
            # The load_beat_data function returns a tuple directly
            beat_timestamps, downbeats, intro_end_idx, ending_start_idx, detected_meter = load_beat_data(beats_file)
            
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
                
                # Check if the video file was actually created despite potential warnings
                if success or video_file.exists():
                    # Make sure the final progress update is sent through the progress callback
                    # This ensures the UI receives the final status update through the same channel
                    update_progress("Video generation complete", 1.0)  # 100%
                    
                    # Capture final stdout and stderr
                    final_stdout = stdout_capture.getvalue()
                    final_stderr = stderr_capture.getvalue()
                    
                    # Limit the size of stdout/stderr to prevent memory issues
                    max_output_size = 10000  # 10KB limit for logs
                    if len(final_stdout) > max_output_size:
                        final_stdout = "[...truncated...]\n" + final_stdout[-max_output_size:]
                    if len(final_stderr) > max_output_size:
                        final_stderr = "[...truncated...]\n" + final_stderr[-max_output_size:]
                        
                    # Log success
                    safe_info(f"Successfully generated video for file {file_id}")
                    try:
                        print(f"Successfully generated video for file {file_id}")
                    except OSError:
                        # Silently ignore I/O errors when writing to stdout
                        pass
                    
                    # Return the results with comprehensive information
                    return {
                        'status': 'COMPLETED',  # Using uppercase for consistency
                        'file_id': file_id,
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
                    error_msg = "Failed to generate video"
                    stderr_capture.write(error_msg + "\n")
                    raise Exception(error_msg)
                    
            except Exception as video_error:
                # Check if the video file was created despite the error
                if video_file.exists() and video_file.stat().st_size > 0:
                    warning_msg = f"Warning during video generation: {video_error}, but video file was created successfully"
                    print(warning_msg)
                    stdout_capture.write(warning_msg + "\n")
                    
                    # Make sure the final progress update is sent
                    update_progress("Video generation complete (with warnings)", 1.0)  # 100%
                    
                    # Update metadata with warning
                    # Capture final stdout and stderr
                    final_stdout = stdout_capture.getvalue()
                    final_stderr = stderr_capture.getvalue()
                    
                    # Limit the size of stdout/stderr to prevent memory issues
                    max_output_size = 10000  # 10KB limit for logs
                    if len(final_stdout) > max_output_size:
                        final_stdout = "[...truncated...]\n" + final_stdout[-max_output_size:]
                    if len(final_stderr) > max_output_size:
                        final_stderr = "[...truncated...]\n" + final_stderr[-max_output_size:]
                        
                    # Log success with warning
                    safe_info(f"Successfully generated video for file {file_id} (with warning)")
                    try:
                        print(f"Successfully generated video for file {file_id} (with warning)")
                    except OSError:
                        # Silently ignore I/O errors when writing to stdout
                        pass
                    
                    # Return success with warning
                    return {
                        'status': 'completed',
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
            stderr_capture.write(error_msg + "\n")
            
            # Capture current stdout and stderr
            current_stdout = stdout_capture.getvalue()
            current_stderr = stderr_capture.getvalue()
            
            # Limit the size of stdout/stderr to prevent memory issues
            max_output_size = 10000  # 10KB limit for logs
            if len(current_stdout) > max_output_size:
                current_stdout = "[...truncated...]\n" + current_stdout[-max_output_size:]
            if len(current_stderr) > max_output_size:
                current_stderr = "[...truncated...]\n" + current_stderr[-max_output_size:]
            
            # Update task state with error information
            self.update_state(state=states.FAILURE, meta={
                "status": "error",
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
            try:
                print(f"Video generation error for file {file_id}: {str(e)}")
            except OSError:
                # Silently ignore I/O errors when writing to stdout
                pass
            
            # Re-raise the exception
            raise
            
    except Exception as e:
        # Capture final stdout and stderr even in case of exception
        final_stdout = stdout_capture.getvalue() if 'stdout_capture' in locals() else ''
        final_stderr = stderr_capture.getvalue() if 'stderr_capture' in locals() else ''
        
        # Limit the size of stdout/stderr to prevent memory issues
        max_output_size = 10000  # 10KB limit for logs
        if len(final_stdout) > max_output_size:
            final_stdout = "[...truncated...]\n" + final_stdout[-max_output_size:]
        if len(final_stderr) > max_output_size:
            final_stderr = "[...truncated...]\n" + final_stderr[-max_output_size:]
        
        # Return failure result with comprehensive information
        return {
            'status': 'error',
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
    finally:
        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
