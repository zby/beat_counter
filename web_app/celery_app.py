"""
Celery configuration for the beat detection application.

This module sets up Celery for handling long-running tasks like beat detection
and video generation in a distributed manner.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
import logging

# Import Task for type hints
from celery import Celery, states, Task
from web_app.config import Config
from web_app.storage import FileMetadataStorage
from web_app.utils.task_utils import (
    IOCapture, create_progress_updater, safe_error, safe_info, safe_print
)
from beat_detection.core.detector import BeatDetector
from beat_detection.core.video import BeatVideoGenerator
from beat_detection.utils import reporting
from beat_detection.utils.beat_file import load_beat_data

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

# Get max duration from config
MAX_AUDIO_DURATION = config.storage.max_audio_secs

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

@app.task(bind=True, name='detect_beats_task', queue='beat_detection')
def detect_beats_task(self: Task, file_id: str) -> Dict[str, Any]: # self is now celery.Task
    """Celery task for detecting beats in an audio file."""
    task_info = {'file_id': file_id}
    # Use __dict__ to store custom attributes on the task instance if needed, or manage separately
    # self._io_capture = IOCapture() # Avoid modifying 'self' directly if possible
    io_capture = IOCapture() # Manage IOCapture separately

    with io_capture:
        try:
            logger.info(f"Starting beat detection for file_id: {file_id}")
            # Access storage directly via app context
            storage: FileMetadataStorage = self.app.context.storage
            if not storage:
                 logger.error("Storage context not found on Celery app!")
                 raise RuntimeError("Storage context unavailable.")


            # --- File Paths & Pre-checks ---
            job_dir = storage.get_job_directory(file_id)
            job_dir.mkdir(exist_ok=True, parents=True)
            audio_file_path = storage.get_audio_file_path(file_id)

            if not audio_file_path or not audio_file_path.exists():
                logger.error(f"Audio file not found for file_id: {file_id} at path: {audio_file_path}")
                raise FileNotFoundError(f"No audio file found for file_id: {file_id}")

            audio_path = str(audio_file_path.resolve())
            logger.debug(f"Audio path resolved to: {audio_path}")

            # --- Progress & Detection ---
            # Pass io_capture explicitly if needed by updater, or manage output differently
            update_progress = create_progress_updater(self, task_info, 'beat_detection_output')
            update_progress("Initializing beat detection", 0)

            detector = BeatDetector(progress_callback=update_progress)

            logger.info("Starting actual beat detection process...")
            detection_result = detector.detect_beats(audio_path, skip_intro=True, skip_ending=True)
            logger.info("Beat detection process finished.")

            (beat_timestamps, stats, irregular_beats, downbeats,
             intro_end_idx, ending_start_idx, detected_meter) = detection_result

            # --- Save Results ---
            beats_file = storage.get_beats_file_path(file_id)
            stats_file = storage.get_beat_stats_file_path(file_id)

            reporting.save_beat_timestamps(
                beat_timestamps, beats_file, downbeats,
                intro_end_idx=intro_end_idx, ending_start_idx=ending_start_idx,
                detected_meter=detected_meter
            )
            logger.info(f"Beat timestamps saved to: {beats_file}")
            
            # --- FIX: Check NumPy array size instead of truthiness ---
            duration = beat_timestamps[-1] if beat_timestamps is not None and beat_timestamps.size > 0 else 0
            
            reporting.save_beat_statistics(
                stats, irregular_beats, stats_file,
                filename=audio_file_path.name,
                detected_meter=detected_meter,
                duration=duration
            )
            logger.info(f"Beat statistics saved to: {stats_file}")

            # --- Update Metadata ---
            try:
                with open(stats_file, "r") as f:
                    beat_stats = json.load(f)
                storage.update_metadata(file_id, {
                    "beat_stats": beat_stats,
                    "beats_file": str(beats_file),
                    "stats_file": str(stats_file)
                })
                logger.info("Metadata updated with beat detection results.")
            except FileNotFoundError:
                 logger.error(f"Failed to read stats file {stats_file} after saving.")
                 raise
            except json.JSONDecodeError:
                 logger.error(f"Failed to decode stats file {stats_file} as JSON.")
                 raise
            except Exception as meta_e:
                 logger.error(f"Failed to update metadata for {file_id}: {meta_e}")
                 raise

            update_progress("Beat detection complete", 1.0)
            final_stdout, final_stderr = io_capture.get_output() # Use separate capture object

            logger.info(f"Beat detection successful for file_id: {file_id}")
            return {
                "file_id": file_id,
                "beats_file": str(beats_file),
                "stats_file": str(stats_file),
                "beat_detection_output": {"stdout": final_stdout, "stderr": final_stderr}
            }

        except Exception as e:
            error_msg = f"Beat detection error for {file_id}: {type(e).__name__} - {e}"
            logger.error(error_msg, exc_info=True)
            safe_error(error_msg)
            # Use separate io_capture object
            io_capture.write_stderr(error_msg + "\n")
            final_stdout, final_stderr = io_capture.get_output()

            try:
                self.update_state(state=states.FAILURE, meta={
                    'file_id': file_id,
                    'exc_type': type(e).__name__,
                    'exc_message': str(e),
                    'beat_detection_output': {'stdout': final_stdout, 'stderr': final_stderr}
                })
            except Exception as update_state_e:
                 logger.error(f"Failed to update Celery task state for {file_id} during error handling: {update_state_e}")

            raise e


@app.task(bind=True, name='generate_video_task', queue='video_generation')
def generate_video_task(self: Task, file_id: str) -> Dict[str, Any]: # self is now celery.Task
    """Celery task for generating a beat visualization video."""
    task_info = {'file_id': file_id}
    # self._io_capture = IOCapture() # Avoid modifying 'self' directly
    io_capture = IOCapture() # Manage IOCapture separately
    old_cwd = None

    with io_capture:
        try:
            logger.info(f"Starting video generation for file_id: {file_id}")
            # Access storage directly via app context
            storage: FileMetadataStorage = self.app.context.storage
            if not storage:
                 logger.error("Storage context not found on Celery app!")
                 raise RuntimeError("Storage context unavailable.")

            # --- File Paths & Pre-checks ---
            job_dir = storage.get_job_directory(file_id)
            job_dir.mkdir(exist_ok=True, parents=True)

            audio_file_path = storage.get_audio_file_path(file_id)
            beats_file_path = storage.get_beats_file_path(file_id)
            video_output_path = storage.get_video_file_path(file_id)

            if not audio_file_path or not audio_file_path.exists():
                logger.error(f"Audio file not found for file_id: {file_id} at path: {audio_file_path}")
                raise FileNotFoundError(f"No audio file found for file_id: {file_id} at path: {audio_file_path}")
            if not beats_file_path or not beats_file_path.exists():
                logger.error(f"Beats file not found for file_id: {file_id} at path: {beats_file_path}")
                raise FileNotFoundError(f"No beats file found for file_id: {file_id}")

            audio_path = str(audio_file_path.resolve())
            beats_path = str(beats_file_path.resolve())
            video_output = str(video_output_path.resolve())

            logger.debug(f"Audio path: {audio_path}")
            logger.debug(f"Beats path: {beats_path}")
            logger.debug(f"Video output path: {video_output}")

            # --- Change CWD for MoviePy ---
            old_cwd = Path.cwd()
            os.chdir(str(job_dir))
            logger.info(f"Changed working directory to: {job_dir}")

            # --- Progress & Generation ---
            # Pass io_capture explicitly if needed by updater, or manage output differently
            update_progress = create_progress_updater(self, task_info, 'video_generation_output')
            update_progress("Initializing video generation", 0)

            try:
                beat_timestamps, downbeats, _, _, detected_meter = load_beat_data(beats_path)
                # --- FIX: Check for None or empty array explicitly ---
                if beat_timestamps is None or beat_timestamps.size == 0:
                    raise ValueError("No valid beat timestamps found in beats file.")
                logger.info(f"Loaded {len(beat_timestamps)} beats from {beats_path}")
            except Exception as load_e:
                 logger.error(f"Failed to load beat data from {beats_path}: {load_e}")
                 raise ValueError(f"Failed to load beat data: {load_e}") from load_e

            video_generator = BeatVideoGenerator(progress_callback=update_progress)

            update_progress("Starting video rendering", 0.05)
            logger.info("Starting actual video generation process...")

            generated_video_file = video_generator.generate_video(
                audio_path,
                beat_timestamps,
                output_path=video_output,
                downbeats=downbeats,
                detected_meter=detected_meter
            )
            logger.info(f"Video generation process finished. Output: {generated_video_file}")

            # --- Update Metadata ---
            try:
                storage.update_metadata(file_id, {"video_file": video_output})
                logger.info("Metadata updated with video generation results.")
            except Exception as meta_e:
                 logger.error(f"Failed to update metadata for {file_id} after video generation: {meta_e}")
                 raise

            update_progress("Video generation complete", 1.0)
            final_stdout, final_stderr = io_capture.get_output() # Use separate capture object

            logger.info(f"Video generation successful for file_id: {file_id}")
            return {
                "file_id": file_id,
                "video_file": video_output,
                "video_generation_output": {"stdout": final_stdout, "stderr": final_stderr}
            }

        except Exception as e:
            error_msg = f"Video generation error for {file_id}: {type(e).__name__} - {e}"
            logger.error(error_msg, exc_info=True)
            safe_error(error_msg)
            # Use separate io_capture object
            io_capture.write_stderr(error_msg + "\n")
            final_stdout, final_stderr = io_capture.get_output()

            try:
                self.update_state(state=states.FAILURE, meta={
                    'file_id': file_id,
                    'exc_type': type(e).__name__,
                    'exc_message': str(e),
                    'video_generation_output': {'stdout': final_stdout, 'stderr': final_stderr}
                })
            except Exception as update_state_e:
                 logger.error(f"Failed to update Celery task state for {file_id} during error handling: {update_state_e}")

            raise e

        finally:
            if old_cwd and Path.cwd() != old_cwd:
                try:
                    os.chdir(str(old_cwd))
                    logger.info(f"Restored original working directory: {old_cwd}")
                except Exception as chdir_e:
                    logger.error(f"Error restoring original working directory ({old_cwd}): {chdir_e}")


@app.task(bind=True)
def debug_task(self: Task): # self is now celery.Task
    """Simple task for debugging worker setup."""
    logger.info(f'Debug task executed. Request: {self.request!r}')
    print(f'Debug task executed. Request: {self.request!r}')
    return {"status": "ok", "request_id": self.request.id}