"""
Celery configuration for the beat detection application.

This module sets up Celery for handling long-running tasks like beat detection
and video generation in a distributed manner.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import logging

# Import Task for type hints
from celery import Celery, states, Task
from web_app.config import Config
from web_app.storage import FileMetadataStorage
from web_app.utils.task_utils import (
    IOCapture, create_progress_updater, safe_error
)
from beat_detection.core.detector import BeatDetector
from beat_detection.core.video import BeatVideoGenerator
from beat_detection.core.beats import Beats # Import the Beats class

# Set up logger
logger = logging.getLogger(__name__)

# Load configuration
try:
    config = Config.from_env()
except FileNotFoundError as e:
    logger.error(f"Configuration error: {e}")
    raise SystemExit(f"Error: Configuration file not found. {e}") from e
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    raise SystemExit(f"Error loading configuration: {e}") from e


class AppContext:
    """Application context for storing shared services and dependencies."""
    
    def __init__(self, storage: FileMetadataStorage):
        """Initialize the application context.
        
        Args:
            storage: The file metadata storage service
        """
        self.storage = storage 


# Create the Celery app instance
app = Celery(
    config.app.name or 'beat_detection_app',
    broker=config.celery.broker_url,
    backend=config.celery.result_backend,
    include=['web_app.celery_app']
)

# Apply other Celery settings from the Config object
app.conf.update(
    task_serializer=config.celery.task_serializer,
    accept_content=config.celery.accept_content,
    result_serializer=config.celery.task_serializer,
    task_ignore_result=config.celery.task_ignore_result,
    result_extended=config.celery.result_extended,
    task_track_started=config.celery.task_track_started,
)

# Configure Celery logging
log_file_path = config.storage.upload_dir / 'celery.log'
log_file_path.parent.mkdir(parents=True, exist_ok=True)
app.conf.update(
    worker_log_format="%(asctime)s - %(levelname)s - %(message)s",
    worker_task_log_format="%(asctime)s - %(levelname)s - Task:%(task_name)s[%(task_id)s] - %(message)s",
    worker_log_file=str(log_file_path),
    worker_redirect_stdouts=False
)
logger.info(f"Celery worker log file configured at: {log_file_path}")

# Initialize application context with storage
app.context = AppContext(
    storage=FileMetadataStorage(config.storage)
)

# --- Celery Tasks (Base class removed) ---

# Core beat detection logic (separated for testability)

def _perform_beat_detection(
    storage: FileMetadataStorage,
    file_id: str,
    min_bpm: int,
    max_bpm: int,
    tolerance_percent: float,
    min_measures: int,
    beats_per_bar: Optional[int],
    update_progress: Callable[[str, float], None]
) -> dict:
    """Performs the actual beat detection, saving, and metadata update."""
    # Configure detector
    detector = BeatDetector(
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        tolerance_percent=tolerance_percent,
        min_measures=min_measures,
        beats_per_bar=beats_per_bar,
        progress_callback=update_progress
    )
    
    # Detect beats, ensuring the path is a string
    audio_path_str = str(storage.get_audio_file_path(file_id))
    beats = detector.detect_beats(audio_path_str)
    
    # Create metadata for the beat detection result
    metadata = {
        'beat_detection_status': 'success',
        'detected_beats_per_bar': beats.beats_per_bar,
        'total_beats': len(beats.timestamps),
        'detected_tempo_bpm': beats.overall_stats.tempo_bpm,
        'irregularity_percent': beats.overall_stats.irregularity_percent,
        'irregular_beats_count': len(beats.get_irregular_beats())
    }
    
    # Save beat data to JSON
    beat_data = {
        'timestamps': beats.timestamps.tolist(),
        'beats_per_bar': beats.beats_per_bar,
        'downbeats': [i for i, beat in enumerate(beats.beat_list) if beat.beat_count == 1],
        'intro_end_idx': beats.start_regular_beat_idx,
        'ending_start_idx': beats.end_regular_beat_idx
    }

    # Save beat data
    beats_file_path = storage.get_beats_file_path(file_id)
    beats.save_to_file(beats_file_path)

    # Add beat_file path before updating metadata
    metadata['beat_file'] = str(beats_file_path)

    # --- DEBUG PRINT --- #
    logger.info(f"DEBUG: Metadata being saved by Celery task for {file_id}: {metadata}")
    # --- END DEBUG PRINT --- #

    # Update the central metadata store
    storage.update_metadata(file_id, metadata)
    logger.info(f"Metadata updated for file_id {file_id} after beat detection.")

    # Return success information
    return {
        'status': 'success',
        'beat_file': str(beats_file_path), # Return string path
        'total_beats': len(beats.timestamps),
        'beats_per_bar': beats.beats_per_bar,
        'irregular_beats': len(beats.get_irregular_beats()),
        'tempo_bpm': beats.overall_stats.tempo_bpm
    }

# Celery Task Definition
@app.task(bind=True, name='detect_beats_task', queue='beat_detection')
def detect_beats_task(self: Task, file_id: str, min_bpm: int = 60, max_bpm: int = 200, 
                     tolerance_percent: float = 10.0, min_measures: int = 1, 
                     beats_per_bar: int = None) -> dict:
    """
    Celery task wrapper for beat detection.
    Handles context retrieval, progress callback creation, calling the core logic,
    and error handling.
    
    Parameters are passed to the core beat detection function.
        
    Returns:
    --------
    dict
        Task result with status and file paths or error info.
    """
    storage: FileMetadataStorage = None # Initialize to allow use in except block
    try:
        # Access storage directly via app context
        storage = self.app.context.storage
        if not storage:
            logger.error("Storage context not found on Celery app!")
            raise RuntimeError("Storage context unavailable.")

        # Create output directory if it doesn't exist
        storage.ensure_job_directory(file_id)
        
        # Create progress updater callback
        update_progress = create_progress_updater(self, {"file_id": file_id}, 'beat_detection_progress')
        
        # Call the core processing function
        result = _perform_beat_detection(
            storage=storage,
            file_id=file_id,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            tolerance_percent=tolerance_percent,
            min_measures=min_measures,
            beats_per_bar=beats_per_bar,
            update_progress=update_progress
        )
        return result

    except Exception as e:
        logger.error(f"Error processing {file_id} in detect_beats_task: {str(e)}", exc_info=True)
        # Attempt to update metadata with error status
        try:
            # Storage might not have been initialized if error occurred early
            if storage:
                 storage.update_metadata(file_id, {'beat_detection_status': 'error', 'beat_detection_error': str(e)})
            else:
                logger.warning(f"Cannot update metadata with error for {file_id} as storage context was not available.")
        except Exception as meta_err_e:
            logger.error(f"Failed even to update metadata with error for {file_id}: {meta_err_e}")

        return {
            'status': 'error',
            'error': str(e)
        }


def _perform_video_generation(
    storage: FileMetadataStorage,
    file_id: str,
    update_progress: Callable[[str, float], None]
) -> Dict[str, Any]:
    """Performs the actual video generation, handling file paths, CWD changes, and metadata update."""
    old_cwd = None
    try:
        logger.info(f"Starting video generation logic for file_id: {file_id}")

        # --- File Paths & Pre-checks ---
        job_dir = storage.get_job_directory(file_id)
        job_dir.mkdir(exist_ok=True, parents=True)

        audio_file_path = storage.get_audio_file_path(file_id)
        beats_file_path = storage.get_beats_file_path(file_id)
        video_output_path = storage.get_video_file_path(file_id)

        if not audio_file_path or not audio_file_path.exists():
            raise FileNotFoundError(f"No audio file found for file_id: {file_id} at path: {audio_file_path}")
        if not beats_file_path or not beats_file_path.exists():
            raise FileNotFoundError(f"No beats file found for file_id: {file_id} at path: {beats_file_path}")

        audio_path = str(audio_file_path.resolve())
        beats_path = str(beats_file_path.resolve())
        video_output = str(video_output_path.resolve())

        logger.debug(f"Audio path: {audio_path}")
        logger.debug(f"Beats path: {beats_path}")
        logger.debug(f"Video output path: {video_output}")

        # --- Change CWD for MoviePy ---
        # It's crucial this happens within the function that *uses* MoviePy
        old_cwd = Path.cwd()
        os.chdir(str(job_dir))
        logger.info(f"Changed working directory to: {job_dir}")

        # --- Load Beat Data ---
        update_progress("Loading beat data", 0.01)
        try:
            # Load the full Beats object
            beats = Beats.load_from_file(beats_path)
            logger.info(f"Loaded Beats object from {beats_path}")
            if not beats.timestamps.size > 0:
                 raise ValueError("Loaded Beats object contains no timestamps.")
        except FileNotFoundError:
            logger.error(f"Beats file not found for loading: {beats_path}")
            raise # Re-raise the specific error
        except Exception as load_e:
             logger.error(f"Failed to load Beats object from {beats_path}: {load_e}")
             raise ValueError(f"Failed to load Beats object: {load_e}") from load_e

        # --- Generate Video ---
        # Instantiate generator (progress callback not passed via constructor anymore based on test example)
        video_generator = BeatVideoGenerator()
        
        update_progress("Starting video rendering", 0.05)
        logger.info("Starting actual video generation process using Beats object...")

        # Call generate_video with audio path, Beats object, and output path
        generated_video_file = video_generator.generate_video(
            audio_path=audio_path,
            beats=beats, # Pass the loaded Beats object
            output_path=video_output
        )
        logger.info(f"Video generation process finished. Output: {generated_video_file}")

        # --- Update Metadata ---
        storage.update_metadata(file_id, {"video_file": video_output, "video_generation_status": "success"})
        logger.info("Metadata updated with video generation results.")

        update_progress("Video generation complete", 1.0)
        return {
            "status": "success",
            "file_id": file_id,
            "video_file": video_output
        }

    finally:
        # Restore CWD if it was changed
        if old_cwd and Path.cwd() != old_cwd:
            try:
                os.chdir(str(old_cwd))
                logger.info(f"Restored original working directory: {old_cwd}")
            except Exception as chdir_e:
                logger.error(f"Error restoring original working directory ({old_cwd}): {chdir_e}")


@app.task(bind=True, name='generate_video_task', queue='video_generation')
def generate_video_task(self: Task, file_id: str) -> Dict[str, Any]:
    """
    Celery task wrapper for generating a beat visualization video.
    Handles context retrieval, progress callback creation, calling the core logic,
    and error handling.
    
    Parameters:
    -----------
    file_id : str
        Identifier for the file to process
        
    Returns:
    --------
    dict
        Task result with status and file paths or error info.
    """
    task_info = {'file_id': file_id}
    io_capture = IOCapture()
    storage: FileMetadataStorage = None # Allow use in except block

    with io_capture:
        try:
            logger.info(f"Starting video generation task for file_id: {file_id}")
            # Access storage directly via app context
            storage = self.app.context.storage
            if not storage:
                 logger.error("Storage context not found on Celery app!")
                 raise RuntimeError("Storage context unavailable.")

            # Create progress updater callback (captures stdout/stderr)
            update_progress = create_progress_updater(self, task_info, 'video_generation_output')
            update_progress("Initializing video generation task", 0)

            # Call the core processing function
            result = _perform_video_generation(
                storage=storage,
                file_id=file_id,
                update_progress=update_progress
            )

            # Add captured output to the success result
            final_stdout, final_stderr = io_capture.get_output()
            result["video_generation_output"] = {"stdout": final_stdout, "stderr": final_stderr}
            return result

        except Exception as e:
            error_msg = f"Video generation task error for {file_id}: {type(e).__name__} - {e}"
            logger.error(error_msg, exc_info=True)
            safe_error(error_msg) # Log safely
            # Attempt to capture error output
            io_capture.write_stderr(error_msg + "\n") 
            final_stdout, final_stderr = io_capture.get_output()

            # Attempt to update metadata with error status
            try:
                if storage:
                    storage.update_metadata(file_id, {'video_generation_status': 'error', 'video_generation_error': str(e)})
                else:
                    logger.warning(f"Cannot update metadata with error for {file_id} as storage context was not available.")
            except Exception as meta_err_e:
                logger.error(f"Failed even to update metadata with error for {file_id}: {meta_err_e}")

            return {
                "status": "error",
                "file_id": file_id,
                "error": str(e),
                "video_generation_output": {"stdout": final_stdout, "stderr": final_stderr}
            }
        # Note: The finally block for restoring CWD is inside _perform_video_generation


@app.task(bind=True, name='debug_task')
def debug_task(self: Task):
    """
    Simple task for debugging worker setup.
    
    Returns:
    --------
    dict
        Debug information about the task execution
    """
    task_id = self.request.id if self.request else "unknown"
    logger.info(f'Debug task executed. Request: {self.request!r}')
    
    return {
        "status": "ok", 
        "request_id": task_id,
        "worker_hostname": self.request.hostname if self.request else "unknown"
    }