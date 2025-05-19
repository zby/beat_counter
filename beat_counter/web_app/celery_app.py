"""
Celery configuration for the beat detection application.

This module sets up Celery for handling long-running tasks like beat detection
and video generation in a distributed manner.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING
import logging

from celery import Celery, states, Task
from celery.signals import worker_init, worker_process_init
from beat_counter.web_app.config import Config, ConfigurationError # Ensure Config and ConfigurationError are imported
from beat_counter.web_app.storage import FileMetadataStorage
from beat_counter.web_app.utils.task_utils import IOCapture, create_progress_updater, safe_error
from beat_counter.core import extract_beats
from beat_counter.core.video import BeatVideoGenerator
from beat_counter.core.beats import Beats, BeatCalculationError, RawBeats
from beat_counter.core.detector_protocol import BeatDetector
from beat_counter.genre_db import GenreDB, parse_genre_from_path
import numpy as np

# Set up logger
logger = logging.getLogger(__name__)

# Removed import-time configuration loading
# try:
#     config = Config.from_env()
# except FileNotFoundError as e:
#     logger.error(f"Configuration error: {e}")
#     raise SystemExit(f"Error: Configuration file not found. {e}") from e
# except Exception as e:
#     logger.error(f"Error loading configuration: {e}")
#     raise SystemExit(f"Error loading configuration: {e}") from e


class AppContext:
    """Application context for storing shared services and dependencies."""

    def __init__(self, storage: FileMetadataStorage):
        """Initialize the application context.

        Args:
            storage: The file metadata storage service
        """
        self.storage = storage


# Create a basic Celery app instance globally. It will be configured later.
# The name here is a placeholder and will be updated during initialization.
# `include` ensures tasks in this module are found.
app = Celery("beat_detection_app_unconfigured", include=["beat_counter.web_app.celery_app"])

# Global variable to track if initialization has occurred, primarily for safety/assertion.
_celery_app_initialized = False

def initialize_celery_app(app_config: Config) -> Celery:
    """Configures the global Celery app instance with settings from the Config object."""
    global app, _celery_app_initialized

    if not isinstance(app_config, Config):
        raise TypeError("app_config must be an instance of beat_counter.web_app.config.Config")

    if _celery_app_initialized:
        # Optionally, decide how to handle re-initialization. For now, log a warning.
        logger.warning(
            f"Celery app already initialized. Attempting to reconfigure with new config: {app_config.app.name}"
        )
        # Consider if app.main or other critical parts should be protected or reset explicitly.

    # Update the Celery app's main name (used in task names, logging, etc.)
    app.main = app_config.app.name or "beat_detection_app"

    # Configure broker and backend
    app.conf.broker_url = app_config.celery.broker_url
    app.conf.result_backend = app_config.celery.result_backend

    # Apply other Celery settings from the Config object
    app.conf.update(
        task_serializer=app_config.celery.task_serializer,
        accept_content=app_config.celery.accept_content,
        result_serializer=app_config.celery.task_serializer,  # Often same as task_serializer
        task_ignore_result=app_config.celery.task_ignore_result,
        result_extended=app_config.celery.result_extended,
        task_track_started=app_config.celery.task_track_started,
    )

    # Configure Celery logging - Note: direct file logging via conf is often less flexible.
    # It's generally better to configure Python's standard logging module, which Celery uses.
    # However, if specific worker log file behavior from conf is desired:
    # log_file_path = Path(app_config.storage.upload_dir) / "celery_worker.log"
    # log_file_path.parent.mkdir(parents=True, exist_ok=True)
    # app.conf.worker_log_file = str(log_file_path)
    # app.conf.worker_log_format = "%(asctime)s - %(levelname)s - %(message)s"
    # app.conf.worker_task_log_format = "%(asctime)s - %(levelname)s - Task:%(task_name)s[%(task_id)s] - %(message)s"
    
    # Standard practice: redirect Celery's stdout/stderr to its logger.
    # False means it uses Python's logging, True means it redirects to its own log files if configured.
    app.conf.worker_redirect_stdouts = False
    logger.info(
        "Celery logging will use standard Python logging. Ensure handlers are configured appropriately."
    )

    # Initialize application context with storage
    try:
        storage_service = FileMetadataStorage(app_config.storage)
        app.context = AppContext(storage=storage_service)
    except Exception as e:
        logger.error(f"Failed to initialize FileMetadataStorage or AppContext: {e}")
        raise ConfigurationError(f"Error initializing storage for Celery context: {e}") from e

    _celery_app_initialized = True
    logger.info(f"Celery app '{app.main}' has been configured.")
    return app


@worker_init.connect
def celery_worker_init_handler(**kwargs):
    """Signal handler for Celery worker initialization."""
    logger.info("Celery worker_init signal received. Initializing application configuration...")
    try:
        # Step 1: Get the application root directory from the environment variable
        app_root_dir = Config.get_app_dir_from_env()
        logger.info(f"Application root directory for config: {app_root_dir}")

        # Step 2: Load the configuration from the specified directory
        loaded_config = Config.from_dir(app_root_dir)
        logger.info(f"Configuration loaded successfully for app: {loaded_config.app.name}")

        # Step 3: Initialize the global Celery app instance with this configuration
        initialize_celery_app(loaded_config)
        # The global 'app' in this module is now fully configured.
        logger.info("Global Celery app instance has been initialized and configured via worker_init.")

    except ConfigurationError as e:
        logger.critical(f"CRITICAL: Configuration failed during worker_init: {e}. Worker cannot operate.", exc_info=True)
        # Fail Fast: Exit the worker process if configuration is faulty.
        raise SystemExit(f"FATAL: Worker could not initialize due to configuration error: {e}") from e
    except FileNotFoundError as e: # Raised by Config.from_dir if files are missing
        logger.critical(f"CRITICAL: Configuration file not found during worker_init: {e}. Worker cannot start.", exc_info=True)
        raise SystemExit(f"FATAL: Worker could not initialize, essential configuration file not found: {e}") from e
    except Exception as e: # Catch any other unexpected errors during this critical phase
        logger.critical(f"CRITICAL: An unexpected error occurred during worker_init configuration: {e}", exc_info=True)
        raise SystemExit(f"FATAL: Worker could not initialize due to an unexpected error: {e}") from e


# --- Celery Tasks ---
# Task definitions remain below. They use `@app.task` which now works because `app` is a Celery instance.
# Inside tasks, `self.app` will refer to this globally configured `app`.

# Ensure that tasks can access the app context IF the app has been initialized.
# This is a runtime check that could be added to tasks if necessary, or rely on entry points to initialize.

# Example: (This check would be inside a task if we were very defensive,
# but typically initialization is guaranteed by the worker startup process)
# if not _celery_app_initialized:
#     raise RuntimeError("Celery app accessed before initialization with config.")

# Core beat detection logic (separated for testability)

def _perform_beat_detection(
    storage: FileMetadataStorage,
    file_id: str,
    detector_name: str,
    min_bpm: int,
    max_bpm: int,
    tolerance_percent: float,
    min_measures: int,
    beats_per_bar: Optional[int], # Optional override
    genre: Optional[str] = None,  # Optional genre for defaults
    use_genre_defaults: bool = False  # Whether to try to detect genre from path
) -> None:
    """Performs beat detection using extract_beats, saves Beats object, and updates metadata."""

    audio_path_str = str(storage.get_audio_file_path(file_id))
    beats_file_path = storage.get_beats_file_path(file_id)
    output_path_str = str(beats_file_path)

    # Prepare base arguments
    detector_kwargs = {
        "min_bpm": min_bpm,
        "max_bpm": max_bpm,
    }
    beats_constructor_args = {
        "beats_per_bar": beats_per_bar,
        "tolerance_percent": tolerance_percent,
        "min_measures": min_measures,
    }

    # Check for genre-based defaults
    detected_genre = None
    if genre or use_genre_defaults:
        genre_db = GenreDB()  # Instantiate genre database
        
        # Use explicit genre if provided
        if genre:
            detected_genre = genre
            logger.info(f"Using provided genre '{genre}' for file_id {file_id}")
        # Otherwise try to detect from path if enabled
        elif use_genre_defaults:
            try:
                detected_genre = parse_genre_from_path(audio_path_str)
                logger.info(f"Detected genre '{detected_genre}' from path for file_id {file_id}")
            except ValueError:
                logger.info(f"No genre detected in path for file_id {file_id}")
        
        # Apply genre defaults if we have a genre
        if detected_genre:
            # Apply genre defaults to detector kwargs and beats args
            detector_kwargs = genre_db.detector_kwargs_for_genre(detected_genre, existing=detector_kwargs)
            beats_constructor_args = genre_db.beats_kwargs_for_genre(detected_genre, existing=beats_constructor_args)
            logger.info(f"Applied genre defaults for '{detected_genre}' to file_id {file_id}")

    try:
        # Call extract_beats - handles detection, Beats creation, and saving
        beats_obj = extract_beats(
            audio_file_path=audio_path_str,
            output_path=output_path_str, 
            detector_name=detector_name,
            beats_args=beats_constructor_args,
            **detector_kwargs
        )
        logger.info(f"extract_beats succeeded for {file_id}. Beats saved to {output_path_str}")

    except (BeatCalculationError, FileNotFoundError, ValueError, IOError) as e:
        # Catch specific errors from extract_beats or its callees
        logger.error(f"Beat detection failed during extract_beats for {file_id}: {e}")
        raise # Re-raise critical errors
    except Exception as e:
        # Catch unexpected errors
        logger.exception(f"Unexpected error during extract_beats for {file_id}: {e}")
        raise

    # --- Metadata Update (using the returned beats_obj) ---
    try:
        metadata_update = {
            "beat_detection_status": "success",
            "beat_detection_error": None,
            "detector_name": detector_name,
            "detected_beats_per_bar": beats_obj.beats_per_bar,
            "total_beats": len(beats_obj.timestamps),
            "detected_tempo_bpm": beats_obj.overall_stats.tempo_bpm,
            "irregularity_percent": beats_obj.overall_stats.irregularity_percent,
            "irregular_beats_count": len(beats_obj.irregular_beat_indices),
            "clip_length": beats_obj.clip_length,  # Add clip_length to metadata
            "beats_file": output_path_str, # Use the path where extract_beats saved the file
            "analysis_params": {
                "beats_per_bar_override": beats_per_bar,
                "tolerance_percent": tolerance_percent,
                "min_measures": min_measures,
                "detected_genre": detected_genre,  # Add genre to metadata if detected
                "genre_provided": genre,  # Add provided genre to metadata
            }
        }
        # Update the central metadata store
        storage.update_metadata(file_id, metadata_update)
        logger.info(
            f"Metadata updated for file_id {file_id} after successful beat detection."
        )
    except Exception as e:
        # If metadata update fails AFTER successful detection/saving, log but don't fail task?
        # Or should this also cause a task failure?
        logger.exception(f"Failed to update metadata for {file_id} after successful beat detection: {e}")
        # Decide whether to re-raise here based on desired behavior
        # raise 

    # No explicit return needed, success/failure handled via metadata and exceptions


# Celery Task Definition
@app.task(bind=True, name="detect_beats_task", queue="beat_detection")
def detect_beats_task(
    self: Task,
    file_id: str,
    detector_name: str = "madmom",
    min_bpm: int = 60,
    max_bpm: int = 200,
    tolerance_percent: float = 10.0,
    min_measures: int = 1,
    beats_per_bar: int = None,
    genre: str = None,  # Optional explicit genre
    use_genre_defaults: bool = False,  # Whether to detect genre from path
) -> None:
    """
    Celery task wrapper for beat detection.
    Handles context retrieval, progress callback creation, calling the core logic,
    and error handling.

    Parameters are passed to the core beat detection function.

    The task supports genre-based defaults in two ways:
    1. Explicit genre parameter
    2. Auto-detection from path when use_genre_defaults=True

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

        # Call the core processing function
        _perform_beat_detection(
            storage=storage,
            file_id=file_id,
            detector_name=detector_name,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            tolerance_percent=tolerance_percent,
            min_measures=min_measures,
            beats_per_bar=beats_per_bar,
            genre=genre,
            use_genre_defaults=use_genre_defaults,
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

        # --- Load Data (Simplified Raw Beats and Analysis Params) ---
        update_progress("Loading beat data and parameters", 0.01)
        raw_beats: RawBeats = None
        beats_per_bar_override: Optional[int] = None
        tolerance_percent: Optional[float] = None
        min_measures: Optional[int] = None
        try:
            # Load Simplified Raw Beats (no beats_per_bar)
            raw_beats = RawBeats.load_from_file(beats_path)
            logger.info(f"Loaded simplified RawBeats object from {beats_path}")
            if not raw_beats.timestamps.size > 0:
                raise ValueError("Loaded RawBeats object contains no timestamps.")

            # Load metadata to get analysis parameters
            metadata = storage.get_metadata(file_id)
            if not metadata:
                raise ValueError(f"Could not load metadata for file_id {file_id}")

            analysis_params = metadata.get("analysis_params")
            if not analysis_params:
                # If analysis_params are missing, maybe log a warning and try defaults?
                # For now, let's enforce the fail-fast principle.
                logger.error(
                    f"CRITICAL: 'analysis_params' not found in metadata for {file_id}. Cannot reconstruct Beats. Beat detection task likely failed to save them."
                )
                raise ValueError(
                    f"'analysis_params' missing in metadata for {file_id}."
                )

            # Extract parameters
            beats_per_bar_override = analysis_params.get("beats_per_bar_override") # Optional
            tolerance_percent = analysis_params.get("tolerance_percent")
            min_measures = analysis_params.get("min_measures")

            # tolerance_percent and min_measures are required by Beats constructor
            if tolerance_percent is None:
                 raise ValueError(
                     f"Missing 'tolerance_percent' within 'analysis_params' in metadata for {file_id}"
                 )
            if min_measures is None:
                 raise ValueError(
                     f"Missing 'min_measures' within 'analysis_params' in metadata for {file_id}"
                 )

            logger.info(
                f"Loaded analysis params: bpb_override={beats_per_bar_override}, tol%={tolerance_percent}, min_meas={min_measures}"
            )

        except FileNotFoundError:
            logger.error(f"Raw Beats file not found for loading: {beats_path}")
            raise  # Re-raise the specific error
        except Exception as load_e:
            logger.error(f"Failed to load data or parameters from {beats_path} or metadata: {load_e}")
            raise ValueError(f"Failed to load data/params: {load_e}") from load_e

        # --- Construct Beats Object ---
        update_progress("Constructing beat analysis object", 0.03)
        try:
            # Use the analysis parameters from metadata when constructing Beats
            beats = Beats(
                raw_beats=raw_beats, # Pass the loaded simplified RawBeats
                beats_per_bar=beats_per_bar_override, # Pass the optional override
                tolerance_percent=float(tolerance_percent),
                min_measures=int(min_measures)
            )
            logger.info(f"Successfully constructed Beats object for {file_id}")
        except Exception as construct_e:
            logger.error(f"Failed to construct Beats object: {construct_e}", exc_info=True)
            raise ValueError(f"Beats construction failed: {construct_e}") from construct_e

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

    if not _celery_app_initialized:
        logger.error("Debug task: Celery app not initialized!")
        self.update_state(state=states.FAILURE, meta="Celery app not initialized")
        raise RuntimeError("Celery app not initialized at time of debug_task execution")

    try:
        # Try to access context to see if it's there
        app_context = self.app.context
        if app_context and app_context.storage:
            logger.info(f"Debug task: App context and storage seem available. Storage config: {app_context.storage.config}")
            # You could add a more specific check here, e.g., try a benign storage operation
        else:
            logger.warning("Debug task: App context or storage is missing.")
            # Potentially update state to reflect this issue

        # Example of using IOCapture if needed for a command
        # with IOCapture() as output:
        #     print("This is some stdout from the debug task")
        # logger.info(f"Debug task captured output: {output.stdout}")

        return {"status": "Debug task completed successfully", "file_id": None}
    except Exception as e:
        logger.exception(f"Error during debug_task: {e}")
        self.update_state(state=states.FAILURE, meta=safe_error(e))
        # According to fail-fast, we should re-raise to make the failure clear
        raise
