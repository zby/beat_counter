"""Storage implementations for metadata management."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, BinaryIO
import json
import logging
import os
import pathlib
from datetime import datetime
import aiofiles
import asyncio
import fcntl
import time
import tempfile
from pydub import AudioSegment

# Set up logger
logger = logging.getLogger(__name__)

# Define file processing states as string constants
ANALYZING = "ANALYZING"
ANALYZED = "ANALYZED"
ANALYZING_FAILURE = "ANALYZING_FAILURE"
GENERATING_VIDEO = "GENERATING_VIDEO"
COMPLETED = "COMPLETED"
VIDEO_ERROR = "VIDEO_ERROR"
ERROR = "ERROR"

# Set of all valid states for validation
VALID_STATES = {
    ANALYZING, ANALYZED, ANALYZING_FAILURE,
    GENERATING_VIDEO, COMPLETED, VIDEO_ERROR, ERROR
}

def is_in_progress(state: str) -> bool:
    """Check if a state indicates work is in progress.
    
    Args:
        state: The state to check
        
    Returns:
        bool: True if the state indicates work is in progress
    """
    return state in {ANALYZING, GENERATING_VIDEO}

class MetadataStorage(ABC):
    """Abstract base class for metadata storage."""
    
    def __init__(self, max_audio_duration: int = 60):
        """Initialize the storage with configuration.
        
        Args:
            max_audio_duration: Maximum audio duration in seconds (default: 60)
        """
        self.max_audio_duration = max_audio_duration * 1000  # Convert to milliseconds
    
    @abstractmethod
    async def get_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific file."""
        pass
    
    @abstractmethod
    def update_metadata(self, file_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a specific file."""
        pass
    
    @abstractmethod
    async def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all files."""
        pass

    @abstractmethod
    def delete_metadata(self, file_id: str) -> bool:
        """Delete metadata for a specific file."""
        pass

    @abstractmethod
    async def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """Get the file metadata and file existence information."""
        pass
    
    # Path management methods - all implementations should provide these
    @abstractmethod
    def get_job_directory(self, file_id: str) -> pathlib.Path:
        """Get the standardized job directory for a file ID."""
        pass
    
    @abstractmethod
    def get_audio_file_path(self, file_id: str, file_extension: str = None) -> pathlib.Path:
        """Get the standardized path for the audio file."""
        pass
    
    @abstractmethod
    def get_beats_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the beats file."""
        pass
    
    @abstractmethod
    def get_beat_stats_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the beat statistics file."""
        pass
    
    @abstractmethod
    def get_video_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the visualization video."""
        pass
    
    @abstractmethod
    def ensure_job_directory(self, file_id: str) -> pathlib.Path:
        """Ensure the job directory exists and return its path."""
        pass
        
    @abstractmethod
    def save_audio_file(self, file_id: str, file_extension: str, file_obj: BinaryIO, filename: str = None) -> pathlib.Path:
        """Save an uploaded audio file to the storage and return its path.
        Also creates and saves basic metadata about the file.
        
        Args:
            file_id: The unique ID for the file
            file_extension: The file extension of the audio file
            file_obj: A binary file-like object that supports read operations
            filename: Original filename (optional)
            
        Returns:
            Path to the saved audio file
        """
        pass

class TaskExecutor(ABC):
    """Abstract base class for task execution."""
    
    @abstractmethod
    def execute_beat_detection(self, file_id: str) -> Any:
        """Execute beat detection task."""
        pass
    
    @abstractmethod
    def execute_video_generation(self, file_id: str) -> Any:
        """Execute video generation task."""
        pass
    
    @abstractmethod
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task."""
        pass

class FileMetadataStorage(MetadataStorage):
    """File-based implementation of metadata storage with standardized directory structure."""
    
    def __init__(self, base_dir: str = "web_app/uploads", max_audio_duration: int = 60):
        """Initialize the storage with a base directory.
        
        Args:
            base_dir: Base directory for file storage
            max_audio_duration: Maximum audio duration in seconds (default: 60)
        """
        super().__init__(max_audio_duration)
        self.base_upload_dir = pathlib.Path(base_dir)
        # Ensure the base directory exists
        self.base_upload_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"File metadata storage initialized with base directory: {self.base_upload_dir}")
    
    def get_metadata_file_path(self, file_id: str) -> pathlib.Path:
        """Get the path to the metadata JSON file for a file ID."""
        job_dir = self.get_job_directory(file_id)
        return job_dir / "metadata.json"
    
    async def get_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific file."""
        metadata_file = self.get_metadata_file_path(file_id)
        
        if not metadata_file.exists():
            return None
        
        try:
            async with aiofiles.open(metadata_file, 'r') as f:
                metadata_json = await f.read()
                return json.loads(metadata_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metadata file as JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading metadata file: {e}")
            return None
    
    def get_metadata_sync(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous version of get_metadata for internal use."""
        metadata_file = self.get_metadata_file_path(file_id)
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metadata file as JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading metadata file: {e}")
            return None
    
    def update_metadata(self, file_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a specific file with file locking for atomicity."""
        # Ensure job directory exists
        job_dir = self.ensure_job_directory(file_id)
        metadata_file = self.get_metadata_file_path(file_id)
        lock_file = metadata_file.with_suffix('.lock')
        
        # Create lock file if it doesn't exist
        if not lock_file.exists():
            try:
                with open(lock_file, 'w') as f:
                    pass  # Just create an empty file
            except Exception as e:
                logger.warning(f"Failed to create lock file, proceeding without locking: {e}")
        
        # Acquire lock for atomic update
        try:
            # Open the lock file 
            with open(lock_file, 'r+') as lock_f:
                # Try to acquire an exclusive lock with timeout
                max_wait = 10  # Maximum wait time in seconds
                start_time = time.time()
                
                while True:
                    try:
                        # Try to get an exclusive lock
                        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break  # Lock acquired
                    except IOError:
                        # Lock not acquired, wait and retry
                        elapsed = time.time() - start_time
                        if elapsed > max_wait:
                            logger.warning(f"Timeout waiting for lock, proceeding without lock: {file_id}")
                            break
                        time.sleep(0.1)
                
                try:
                    # Read the latest metadata to ensure we have the most up-to-date version
                    existing = {}
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            try:
                                existing = json.load(f)
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON in metadata file, starting with empty metadata: {file_id}")
                    
                    # Deep update the metadata
                    self._deep_update(existing, metadata)
                    
                    # Write directly to the file since we have an exclusive lock
                    with open(metadata_file, 'w') as f:
                        json.dump(existing, f, indent=2)
                    
                finally:
                    # Release the lock
                    fcntl.flock(lock_f, fcntl.LOCK_UN)
        
        except Exception as e:
            logger.error(f"Error updating metadata file: {e}")
            
            # Fall back to non-atomic update on error
            try:
                logger.warning(f"Falling back to non-atomic update for {file_id}")
                existing = self.get_metadata_sync(file_id) or {}
                self._deep_update(existing, metadata)
                with open(metadata_file, 'w') as f:
                    json.dump(existing, f, indent=2)
            except Exception as inner_e:
                logger.error(f"Fallback update also failed: {inner_e}")
                raise inner_e  # Re-raise the inner exception
            
            raise  # Re-raise the original exception
    
    def check_ready_for_confirmation(self, file_id: str, beat_task_status: Dict[str, Any]) -> bool:
        """Check if a file is ready for video generation confirmation.
        
        Args:
            file_id: The ID of the file to check
            beat_task_status: The status of the beat detection task
            
        Returns:
            bool: True if ready for confirmation, False otherwise
        """
        try:
            # Check if beat detection task was successful
            if beat_task_status.get("state") != "SUCCESS":
                logger.info(f"Beat detection task for {file_id} not successful: {beat_task_status.get('state')}")
                return False
                
            # Check if beats file exists
            beats_file_path = self.get_beats_file_path(file_id)
            if not beats_file_path.exists():
                logger.info(f"Beats file for {file_id} does not exist")
                return False
                
            # Get current metadata
            metadata = self.get_metadata_sync(file_id)
            if not metadata:
                logger.info(f"No metadata found for {file_id}")
                return False
                
            # If beats_file is not in metadata but exists on disk, update metadata
            if 'beats_file' not in metadata and beats_file_path.exists():
                logger.info(f"Updating metadata with beats_file for {file_id}")
                self.update_metadata(file_id, {"beats_file": str(beats_file_path)})
                
            # If beat_stats_file exists but not in metadata, update metadata
            stats_file_path = self.get_beat_stats_file_path(file_id)
            if 'stats_file' not in metadata and stats_file_path.exists():
                logger.info(f"Updating metadata with stats_file for {file_id}")
                
                try:
                    # Try to parse the beat stats file
                    with open(stats_file_path, 'r') as f:
                        beat_stats = json.load(f)
                    
                    # Update metadata with stats and file path
                    self.update_metadata(file_id, {
                        "stats_file": str(stats_file_path),
                        "beat_stats": beat_stats
                    })
                except Exception as e:
                    logger.error(f"Error parsing beat stats file: {e}")
            
            # Check again after potential updates
            metadata = self.get_metadata_sync(file_id)
            if 'beats_file' not in metadata:
                logger.info(f"Beats file still not in metadata for {file_id} after update attempt")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error checking if file {file_id} is ready for confirmation: {e}")
            return False
    
    async def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all files."""
        all_metadata = {}
        
        # Find all job directories
        try:
            for job_dir in self.base_upload_dir.iterdir():
                if job_dir.is_dir():
                    file_id = job_dir.name
                    metadata = await self.get_metadata(file_id)
                    if metadata:
                        all_metadata[file_id] = metadata
        except Exception as e:
            logger.error(f"Error getting all metadata: {e}")
        
        return all_metadata
    
    def delete_metadata(self, file_id: str) -> bool:
        """Delete metadata for a specific file."""
        try:
            metadata_file = self.get_metadata_file_path(file_id)
            if metadata_file.exists():
                metadata_file.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting metadata file: {e}")
            return False
    
    def _deep_update(self, target: Dict, source: Dict) -> None:
        """Helper function for deep updating dictionaries."""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _parse_beat_stats_file(self, stats_file_path: str) -> Dict[str, Any]:
        """Parse beat statistics from a JSON file."""
        if not os.path.exists(stats_file_path):
            logger.warning(f"Beat stats file not found: {stats_file_path}")
            return {}
        
        try:
            with open(stats_file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse beat stats file as JSON: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error reading beat stats file: {e}")
            return {}
    
    async def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """Get the file metadata and file existence information."""
        metadata = await self.get_metadata(file_id)
        if not metadata:
            return None

        # Create response with basic file information
        status_data = {
            "file_id": file_id,
            "original_filename": metadata.get("original_filename"),
            "audio_file_path": metadata.get("audio_file_path"),
            "user_ip": metadata.get("user_ip"),
            "upload_timestamp": metadata.get("upload_timestamp"),
            "uploaded_by": metadata.get("uploaded_by"),
            "original_duration": metadata.get("original_duration"),
            "duration_limit": metadata.get("duration_limit", 60)
        }

        # Add task IDs to the status data
        beat_task_id = metadata.get("beat_detection")
        video_task_id = metadata.get("video_generation")
        
        if beat_task_id:
            status_data["beat_detection"] = beat_task_id
            
        if video_task_id:
            status_data["video_generation"] = video_task_id

        # Attempt to get beat statistics if they exist
        beats_file = self.get_beats_file_path(file_id)
        if beats_file.exists():
            beat_stats_file = self.get_beat_stats_file_path(file_id)
            if beat_stats_file.exists():
                try:
                    beat_stats = self._parse_beat_stats_file(str(beat_stats_file))
                    status_data["beat_stats"] = beat_stats
                except Exception as e:
                    logger.error(f"Error parsing beat stats: {e}")
        
        return status_data

    # Path management methods
    def get_job_directory(self, file_id: str) -> pathlib.Path:
        """Get the standardized job directory for a file ID."""
        return self.base_upload_dir / file_id
    
    def get_job_directory_creation_time(self, file_id: str) -> float:
        """Get the creation time of the job directory.
        
        Returns:
            Float timestamp representing creation time, or 0 if directory doesn't exist
        """
        try:
            job_dir = self.get_job_directory(file_id)
            if job_dir.exists():
                return job_dir.stat().st_ctime
            return 0
        except Exception as e:
            logger.error(f"Error getting job directory creation time: {e}")
            return 0
    
    def get_audio_file_path(self, file_id: str, file_extension: str = None) -> pathlib.Path:
        """Get the standardized path for the audio file."""
        # If extension provided, return direct path
        if file_extension:
            return self.get_job_directory(file_id) / f"audio{file_extension}"
        
        # If no extension provided, attempt to get from metadata
        metadata = self.get_metadata_sync(file_id)
        if metadata and "file_extension" in metadata:
            return self.get_job_directory(file_id) / f"audio{metadata['file_extension']}"
        
        # Fallback - return path without extension (caller must handle this)
        return self.get_job_directory(file_id) / "audio"
    
    def get_beats_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the beats file."""
        return self.get_job_directory(file_id) / "beats.txt"
    
    def get_beat_stats_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the beat statistics file."""
        return self.get_job_directory(file_id) / "beat_stats.json"
    
    def get_video_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the visualization video."""
        return self.get_job_directory(file_id) / "visualization.mp4"
    
    def ensure_job_directory(self, file_id: str) -> pathlib.Path:
        """Ensure the job directory exists and return its path."""
        job_dir = self.get_job_directory(file_id)
        job_dir.mkdir(exist_ok=True, parents=True)
        return job_dir
        
    def save_audio_file(self, file_id: str, file_extension: str, file_obj: BinaryIO, filename: str = None) -> pathlib.Path:
        """Save an uploaded audio file to the storage and return its path.
        Also creates and saves basic metadata about the file.
        
        Args:
            file_id: The unique ID for the file
            file_extension: The file extension of the audio file
            file_obj: A binary file-like object that supports read operations
            filename: Original filename (optional)
            
        Returns:
            Path to the saved audio file
        """
        # Ensure job directory exists
        self.ensure_job_directory(file_id)
        
        # Create a temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save the uploaded file temporarily
            with open(temp_path, "wb") as f:
                import shutil
                shutil.copyfileobj(file_obj, f)
            
            # Load and truncate the audio file
            audio = AudioSegment.from_file(temp_path)
            original_duration = len(audio) / 1000  # Convert to seconds
            truncated_audio = audio[:self.max_audio_duration]
            
            # Get standardized path for the audio file
            audio_file_path = self.get_audio_file_path(file_id, file_extension)
            
            try:
                # For M4A files, we need to use specific parameters
                if file_extension.lower() == '.m4a':
                    truncated_audio.export(audio_file_path, format='mp4', codec='aac')
                else:
                    # For other formats, use the original format
                    truncated_audio.export(audio_file_path, format=file_extension[1:])
            except Exception as e:
                # If original format fails, try MP3 as fallback
                logger.warning(f"Failed to save in original format {file_extension}, falling back to MP3: {e}")
                audio_file_path = self.get_audio_file_path(file_id, '.mp3')
                truncated_audio.export(audio_file_path, format='mp3')
                file_extension = '.mp3'  # Update extension to match actual format
            
            # Create and save metadata
            metadata = {
                "audio_file_path": str(audio_file_path),
                "file_extension": file_extension,  # Store the actual extension used
                "upload_time": datetime.now().isoformat(),
                "duration_limit": self.max_audio_duration / 1000,  # Store in seconds
                "original_duration": original_duration,  # Store original duration
                "original_filename": filename
            }
            
            self.update_metadata(file_id, metadata)
            
            return audio_file_path
            
        finally:
            # Clean up temporary file
            os.remove(temp_path)

class CeleryTaskExecutor(TaskExecutor):
    """Celery implementation of task execution."""
    
    def __init__(self):
        """Initialize the task executor."""
        pass
    
    def execute_beat_detection(self, file_id: str) -> Any:
        """Execute beat detection task."""
        from web_app.tasks import detect_beats_task
        return detect_beats_task.delay(file_id)
    
    def execute_video_generation(self, file_id: str) -> Any:
        """Execute video generation task."""
        from web_app.tasks import generate_video_task
        return generate_video_task.delay(file_id)
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task."""
        try:
            from celery.result import AsyncResult
            # Create AsyncResult
            async_result = AsyncResult(task_id)
            
            # Get task state
            state = async_result.state
            result = None
            
            # Get result data if available
            if async_result.ready():
                if async_result.successful():
                    result = async_result.result
                else:
                    # For failed tasks, get the error
                    result = str(async_result.result)
            
            return {"state": state, "result": result}
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return {"state": ERROR, "error": str(e)} 