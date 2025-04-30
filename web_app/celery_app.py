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
from web_app.utils.task_utils import IOCapture, create_progress_updater, safe_error
from beat_detection.core.factory import get_beat_detector
from beat_detection.core.video import BeatVideoGenerator
from beat_detection.core.beats import Beats, BeatCalculationError, RawBeats

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
    config.app.name or "beat_detection_app",
    broker=config.celery.broker_url,
    backend=config.celery.result_backend,
    include=["web_app.celery_app"],
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
log_file_path = config.storage.upload_dir / "celery.log"
log_file_path.parent.mkdir(parents=True, exist_ok=True)
app.conf.update(
    worker_log_format="%(asctime)s - %(levelname)s - %(message)s",
    worker_task_log_format="%(asctime)s - %(levelname)s - Task:%(task_name)s[%(task_id)s] - %(message)s",
    worker_log_file=str(log_file_path),
    worker_redirect_stdouts=False,
)
logger.info(f"Celery worker log file configured at: {log_file_path}")

# Initialize application context with storage
app.context = AppContext(storage=FileMetadataStorage(config.storage))

# --- Celery Tasks (Base class removed) ---

# Core beat detection logic (separated for testability)


def _perform_beat_detection(
    storage: FileMetadataStorage,
    file_id: str,
    algorithm: str,
    min_bpm: int,
    max_bpm: int,
    tolerance_percent: float,
    min_measures: int,
    beats_per_bar: Optional[int], # Optional override
    update_progress: Callable[[str, float], None],
) -> None:
    """Performs the actual beat detection, saving, and metadata update."""
    # Get detector from factory
    detector = get_beat_detector(
        algorithm=algorithm,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        progress_callback=update_progress,
    )

    # Detect beats, get simplified RawBeats object
    audio_path_str = str(storage.get_audio_file_path(file_id))
    raw_beats = detector.detect(audio_path_str)
    logger.info(f"Raw beats detected for {file_id}")

    # Create the full Beats object using RawBeats data and optional beats_per_bar override
    try:
        beats = Beats(
            raw_beats=raw_beats,
            beats_per_bar=beats_per_bar,  # Use the optional override if provided
            tolerance_percent=tolerance_percent,
            min_measures=min_measures,
        )
        logger.info(f"Created Beats object for {file_id} with beats_per_bar={beats.beats_per_bar}")
    except BeatCalculationError as e:
        logger.error(f"Failed to create Beats object for {file_id} from RawBeats: {e}")
        raise  # Re-raise the critical error

    # Create metadata using the Beats object
    beats_file_path = storage.get_beats_file_path(file_id)
    metadata_update = {
        "beat_detection_status": "success",
        "beat_detection_error": None,
        "algorithm": algorithm,
        "detected_beats_per_bar": beats.beats_per_bar,
        "total_beats": len(beats.timestamps),
        "detected_tempo_bpm": beats.overall_stats.tempo_bpm,
        "irregularity_percent": beats.overall_stats.irregularity_percent,
        "irregular_beats_count": len(beats.irregular_beat_indices),
        "beats_file": str(beats_file_path),
        # Store parameters used for beat analysis in metadata
        "analysis_params": {
            "beats_per_bar_override": beats_per_bar,  # None if inferred
            "tolerance_percent": tolerance_percent,
            "min_measures": min_measures,
        }
    }

    # Save the simplified RawBeats object
    raw_beats.save_to_file(beats_file_path)
    logger.info(f"Raw beat data saved to {beats_file_path}")

    # Update the central metadata store
    storage.update_metadata(file_id, metadata_update)
    logger.info(
        f"Metadata updated for file_id {file_id} after successful beat detection."
    )

    return


# Celery Task Definition
@app.task(bind=True, name="detect_beats_task", queue="beat_detection")
def detect_beats_task(
    self: Task,
    file_id: str,
    algorithm: str = "madmom",
    min_bpm: int = 60,
    max_bpm: int = 200,
    tolerance_percent: float = 10.0,
    min_measures: int = 1,
    beats_per_bar: int = None,
) -> None:
    """
    Celery task wrapper for beat detection.
    Handles context retrieval, progress callback creation, calling the core logic,
    and error handling.

    Parameters are passed to the core beat detection function.

    Returns:
    --------
    None
        No return needed here, status is written to metadata by _perform_beat_detection
    """
    storage: FileMetadataStorage = None  # Initialize to allow use in except block
    try:
        # Access storage directly via app context
        storage = self.app.context.storage
        if not storage:
            logger.error("Storage context not found on Celery app!")
            raise RuntimeError("Storage context unavailable.")

        # Create output directory if it doesn't exist
        storage.ensure_job_directory(file_id)

        # Create progress updater callback
        update_progress = create_progress_updater(
            self, {"file_id": file_id}, "beat_detection_progress"
        )

        # Call the core processing function
        _perform_beat_detection(
            storage=storage,
            file_id=file_id,
            algorithm=algorithm,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            tolerance_percent=tolerance_percent,
            min_measures=min_measures,
            beats_per_bar=beats_per_bar,
            update_progress=update_progress,
        )
        return

    except Exception as e:
        logger.error(
            f"Error processing {file_id} in detect_beats_task: {str(e)}", exc_info=True
        )
        # Attempt to update metadata with error status
        try:
            # Storage might not have been initialized if error occurred early
            if storage:
                # Ensure error message is also stored
                storage.update_metadata(
                    file_id,
                    {"beat_detection_status": "error", "beat_detection_error": str(e)},
                )
            else:
                logger.warning(
                    f"Cannot update metadata with error for {file_id} as storage context was not available."
                )
        except Exception as meta_err_e:
            logger.error(
                f"Failed even to update metadata with error for {file_id}: {meta_err_e}"
            )

        # Re-raise the exception so Celery marks the task as FAILED
        # The state will be FAILURE, but the detailed reason is in metadata.json
        self.update_state(
            state=states.FAILURE,
            meta={"exc_type": type(e).__name__, "exc_message": str(e)},
        )
        raise  # Important: re-raise after updating state/metadata


def _perform_video_generation(
    storage: FileMetadataStorage,
    file_id: str,
    update_progress: Callable[[str, float], None],
) -> None:
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
            raise FileNotFoundError(
                f"No audio file found for file_id: {file_id} at path: {audio_file_path}"
            )
        if not beats_file_path or not beats_file_path.exists():
            raise FileNotFoundError(
                f"No beats file found for file_id: {file_id} at path: {beats_file_path}"
            )

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

        # --- Load Data (Raw Beats and Reconstruction Params) ---
        update_progress("Loading beat data and parameters", 0.01)
        raw_beats: RawBeats = None
        tolerance_percent: float = None
        min_measures: int = None
        try:
            # Load Raw Beats (contains beats_per_bar)
            raw_beats = RawBeats.load_from_file(beats_path)
            logger.info(f"Loaded RawBeats object from {beats_path} (bpb={raw_beats.beats_per_bar})")
            if not raw_beats.timestamps.size > 0:
                raise ValueError("Loaded RawBeats object contains no timestamps.")

            # Load metadata to get reconstruction parameters (tolerance, min_measures)
            metadata = storage.get_metadata(file_id)
            if not metadata:
                raise ValueError(f"Could not load metadata for file_id {file_id}")

            recon_params = metadata.get("reconstruction_params")
            if not recon_params:
                raise ValueError(
                    f"'reconstruction_params' not found in metadata for {file_id}. Cannot reconstruct Beats."
                )

            # Extract parameters
            tolerance_percent = recon_params.get("tolerance_percent")
            min_measures = recon_params.get("min_measures")

            if tolerance_percent is None or min_measures is None:
                raise ValueError(
                    f"Missing tolerance_percent or min_measures within 'reconstruction_params' in metadata for {file_id}"
                )

            logger.info(
                f"Loaded reconstruction params: tol%={tolerance_percent}, min_meas={min_measures}"
            )

        except FileNotFoundError:
            logger.error(f"Raw Beats file not found for loading: {beats_path}")
            raise  # Re-raise the specific error
        except Exception as load_e:
            logger.error(f"Failed to load data or parameters from {beats_path} or metadata: {load_e}")
            raise ValueError(f"Failed to load data/params: {load_e}") from load_e

        # --- Reconstruct Beats Object ---
        update_progress("Reconstructing beat analysis", 0.03)
        try:
            beats = Beats.from_timestamps(
                timestamps=raw_beats.timestamps,
                beat_counts=raw_beats.beat_counts,
                beats_per_bar=raw_beats.beats_per_bar, # From RawBeats file
                tolerance_percent=float(tolerance_percent), # From metadata
                min_measures=int(min_measures) # From metadata
            )
            logger.info(f"Successfully reconstructed Beats object for {file_id}")
        except Exception as recon_e:
            logger.error(f"Failed to reconstruct Beats object: {recon_e}")
            raise ValueError(f"Beats reconstruction failed: {recon_e}") from recon_e

        # --- Generate Video ---
        video_generator = BeatVideoGenerator()

        update_progress("Starting video rendering", 0.05)
        logger.info("Starting actual video generation process using reconstructed Beats object...")

        # Call generate_video with audio path, RECONSTRUCTED Beats object, and output path
        generated_video_file = video_generator.generate_video(
            audio_path=audio_path,
            beats=beats,  # Pass the RECONSTRUCTED Beats object
            output_path=video_output,
        )
        logger.info(
            f"Video generation process finished. Output: {generated_video_file}"
        )

        update_progress("Video generation complete", 1.0)
        return

    finally:
        # Restore CWD if it was changed
        if old_cwd and Path.cwd() != old_cwd:
            try:
                os.chdir(str(old_cwd))
                logger.info(f"Restored original working directory: {old_cwd}")
                # --- Update Metadata ---
                storage.update_metadata(
                    file_id,
                    {
                        "video_file": str(
                            video_output
                        ),  # Ensure path is stored as string
                        "video_generation_status": "success",
                        "video_generation_error": None,
                    },
                )
                logger.info("Metadata updated with video generation results.")
            except Exception as chdir_e:
                logger.error(
                    f"Error restoring original working directory ({old_cwd}): {chdir_e}"
                )


@app.task(bind=True, name="generate_video_task", queue="video_generation")
def generate_video_task(self: Task, file_id: str) -> None:
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
    None
        No return needed here, status is written by _perform_video_generation
    """
    task_info = {"file_id": file_id}
    io_capture = IOCapture()
    storage: FileMetadataStorage = None  # Allow use in except block

    with io_capture:
        try:
            logger.info(f"Starting video generation task for file_id: {file_id}")
            # Access storage directly via app context
            storage = self.app.context.storage
            if not storage:
                logger.error("Storage context not found on Celery app!")
                raise RuntimeError("Storage context unavailable.")

            # Create progress updater callback (captures stdout/stderr)
            update_progress = create_progress_updater(
                self, task_info, "video_generation_output"
            )
            update_progress("Initializing video generation task", 0)

            # Call the core processing function
            _perform_video_generation(
                storage=storage, file_id=file_id, update_progress=update_progress
            )

            # No return needed, status is written by _perform_video_generation
            # Add captured output to task info (transient, not for persistent state)
            final_stdout, final_stderr = io_capture.get_output()
            self.update_state(
                state=states.SUCCESS,
                meta={"stdout": final_stdout, "stderr": final_stderr},
            )
            return

        except Exception as e:
            error_msg = (
                f"Video generation task error for {file_id}: {type(e).__name__} - {e}"
            )
            logger.error(error_msg, exc_info=True)
            safe_error(error_msg)  # Log safely
            # Attempt to capture error output
            io_capture.write_stderr(error_msg + "\n")
            final_stdout, final_stderr = io_capture.get_output()

            # Attempt to update metadata with error status
            try:
                if storage:
                    storage.update_metadata(
                        file_id,
                        {
                            "video_generation_status": "error",
                            "video_generation_error": str(e),
                        },
                    )
                else:
                    logger.warning(
                        f"Cannot update metadata with error for {file_id} as storage context was not available."
                    )
            except Exception as meta_err_e:
                logger.error(
                    f"Failed even to update metadata with error for {file_id}: {meta_err_e}"
                )

            # Re-raise the exception so Celery marks the task as FAILED
            self.update_state(
                state=states.FAILURE,
                meta={
                    "exc_type": type(e).__name__,
                    "exc_message": str(e),
                    "stdout": final_stdout,
                    "stderr": final_stderr,
                },
            )
            raise  # Important: re-raise after updating state/metadata
        # Note: The finally block for restoring CWD is inside _perform_video_generation


@app.task(bind=True, name="debug_task")
def debug_task(self: Task):
    """
    Simple task for debugging worker setup.

    Returns:
    --------
    dict
        Debug information about the task execution
    """
    task_id = self.request.id if self.request else "unknown"
    logger.info(f"Debug task executed. Request: {self.request!r}")

    return {
        "status": "ok",
        "request_id": task_id,
        "worker_hostname": self.request.hostname if self.request else "unknown",
    }
