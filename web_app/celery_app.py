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
from beat_detection.utils.beat_file import load_beat_data, save_beats
from beat_detection.utils.file_utils import ensure_output_dir

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
def detect_beats_task(self: Task, audio_file: str, output_dir: str, min_bpm: int = 60, max_bpm: int = 200, 
                     tolerance_percent: float = 10.0, min_measures: int = 1, 
                     beats_per_bar: int = None) -> dict:
    """
    Detect beats in an audio file.
    
    Parameters:
    -----------
    audio_file : str
        Path to input audio file
    output_dir : str
        Path to output directory
    min_bpm : int
        Minimum BPM to detect (default: 60)
    max_bpm : int
        Maximum BPM to detect (default: 200)
    tolerance_percent : float
        Percentage tolerance for beat intervals (default: 10.0)
    min_measures : int
        Minimum number of consistent measures for stable section analysis (default: 1)
    beats_per_bar : int
        Number of beats per bar for downbeat detection (default: None, will try all supported meters)
        
    Returns:
    --------
    dict
        Task result with status and file paths
    """
    try:
        # Create output directory if it doesn't exist
        ensure_output_dir(output_dir)
        
        # Create progress updater callback
        update_progress = create_progress_updater(self, {"file": audio_file}, 'beat_detection_progress')
        
        # Configure detector
        detector = BeatDetector(
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            tolerance_percent=tolerance_percent,
            min_measures=min_measures,
            beats_per_bar=beats_per_bar,
            progress_callback=update_progress
        )
        
        # Detect beats
        beats = detector.detect_beats(audio_file)
        
        # Save beat data
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        beat_file = os.path.join(output_dir, f"{base_name}.beats")
        save_beats(beat_file, beats)
        
        return {
            'status': 'success',
            'beat_file': beat_file,
            'total_beats': len(beats.timestamps),
            'meter': beats.meter,
            'irregular_beats': len(beats.irregular_beats),
            'tempo_bpm': beats.stats.tempo_bpm
        }
        
    except Exception as e:
        logger.error(f"Error processing {audio_file}: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e)
        }


@app.task(bind=True, name='generate_video_task', queue='video_generation')
def generate_video_task(self: Task, file_id: str) -> Dict[str, Any]:
    """
    Celery task for generating a beat visualization video.
    
    Parameters:
    -----------
    file_id : str
        Identifier for the file to process
        
    Returns:
    --------
    dict
        Task result with status and file paths
    """
    task_info = {'file_id': file_id}
    io_capture = IOCapture()
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
            update_progress = create_progress_updater(self, task_info, 'video_generation_output')
            update_progress("Initializing video generation", 0)

            try:
                beat_timestamps, downbeats, _, _, detected_meter = load_beat_data(beats_path)
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
            final_stdout, final_stderr = io_capture.get_output()

            return {
                "status": "success",
                "file_id": file_id,
                "video_file": video_output,
                "video_generation_output": {"stdout": final_stdout, "stderr": final_stderr}
            }

        except Exception as e:
            error_msg = f"Video generation error for {file_id}: {type(e).__name__} - {e}"
            logger.error(error_msg, exc_info=True)
            safe_error(error_msg)
            io_capture.write_stderr(error_msg + "\n")
            final_stdout, final_stderr = io_capture.get_output()

            return {
                "status": "error",
                "file_id": file_id,
                "error": str(e),
                "video_generation_output": {"stdout": final_stdout, "stderr": final_stderr}
            }

        finally:
            if old_cwd and Path.cwd() != old_cwd:
                try:
                    os.chdir(str(old_cwd))
                    logger.info(f"Restored original working directory: {old_cwd}")
                except Exception as chdir_e:
                    logger.error(f"Error restoring original working directory ({old_cwd}): {chdir_e}")


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