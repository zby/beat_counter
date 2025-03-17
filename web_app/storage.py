"""Storage implementations for metadata management."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import logging
import os
import pathlib
from datetime import datetime
import aiofiles
import asyncio

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
    async def get_file_status(self, file_id: str, executor: 'TaskExecutor') -> Dict[str, Any]:
        """Get the processing status for a file."""
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
    """File-based metadata storage implementation."""
    
    def __init__(self, base_dir: str):
        """Initialize the file-based metadata storage.
        
        Args:
            base_dir: Base directory for file storage
        """
        self.base_upload_dir = pathlib.Path(base_dir)
        self.base_upload_dir.mkdir(parents=True, exist_ok=True)
        self._stats_cache = {}
    
    # Metadata management methods
    async def get_metadata(self, file_id: str) -> Dict[str, Any]:
        """Get metadata for a specific file."""
        metadata_file = self.get_job_directory(file_id) / "metadata.json"
        if not metadata_file.exists():
            return None
        
        async with aiofiles.open(metadata_file, "r") as f:
            content = await f.read()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse metadata for file {file_id}")
                return None
    
    def get_metadata_sync(self, file_id: str) -> Dict[str, Any]:
        """Synchronous version of get_metadata."""
        metadata_file = self.get_job_directory(file_id) / "metadata.json"
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.error(f"Failed to parse metadata for file {file_id}")
            return None
    
    def update_metadata(self, file_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a specific file."""
        job_dir = self.get_job_directory(file_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = job_dir / "metadata.json"
        
        # If metadata file exists, merge with existing data
        existing = self.get_metadata_sync(file_id) or {}
        
        # Update existing metadata with new data
        for key, value in metadata.items():
            existing[key] = value
        
        try:
            with open(metadata_file, "w") as f:
                json.dump(existing, f, indent=2)
        except IOError:
            logger.exception(f"Failed to write metadata for file {file_id}")
            raise
    
    async def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all files."""
        all_metadata = {}
        
        # Find all job directories in the base directory
        try:
            # List all directories in the base directory
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
        metadata_file = self.get_job_directory(file_id) / "metadata.json"
        if metadata_file.exists():
            try:
                metadata_file.unlink()
                return True
            except IOError:
                logger.exception(f"Failed to delete metadata for file {file_id}")
        return False
    
    async def get_file_status(self, file_id: str, executor: 'TaskExecutor') -> Dict[str, Any]:
        """Get the processing status for a file."""
        metadata = await self.get_metadata(file_id)
        if not metadata:
            return None

        # Get task statuses
        beat_task_id = metadata.get("beat_detection")
        video_task_id = metadata.get("video_generation")

        status_data = {
            "file_id": file_id,
            "filename": metadata.get("original_filename"),
            "audio_file_path": metadata.get("audio_file_path")
        }

        # Determine overall status based on task states
        overall_status = ERROR

        if beat_task_id:
            beat_task_status = executor.get_task_status(beat_task_id)
            # Ensure task ID is included
            beat_task_status["id"] = beat_task_id
            status_data["beat_detection_task"] = beat_task_status

            if beat_task_status["state"] == "SUCCESS":
                overall_status = ANALYZED
            elif beat_task_status["state"] == "FAILURE":
                overall_status = ANALYZING_FAILURE
            else:
                overall_status = ANALYZING

        if video_task_id:
            video_task_status = executor.get_task_status(video_task_id)
            # Ensure task ID is included
            video_task_status["id"] = video_task_id
            status_data["video_generation_task"] = video_task_status

            if video_task_status["state"] == "SUCCESS":
                overall_status = COMPLETED
            elif video_task_status["state"] == "FAILURE":
                overall_status = VIDEO_ERROR
            else:
                overall_status = GENERATING_VIDEO

        # Add overall status to response
        status_data["status"] = overall_status

        # Check for beat stats file and load it if it exists (regardless of task status)
        beat_task_status = status_data.get("beat_detection_task", {})
        if beat_task_status.get("state") == "SUCCESS":
            stats_file = self.get_beat_stats_file_path(file_id)
            if stats_file.exists():
                stats_data = self._parse_beat_stats_file(str(stats_file))
                if stats_data:
                    status_data["beat_stats"] = stats_data

        return status_data
    
    def _parse_beat_stats_file(self, filepath: str) -> Dict[str, Any]:
        """Parse a beats statistics file and return the data.
        
        The stats file is expected to be a JSON file with beat statistics.
        """
        # Check cache first
        if filepath in self._stats_cache:
            return self._stats_cache[filepath]
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Cache the parsed data
                self._stats_cache[filepath] = data
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to parse beat stats file {filepath}: {e}")
            return None
    
    # Path management methods
    def get_job_directory(self, file_id: str) -> pathlib.Path:
        """Get the standardized job directory for a file ID."""
        return self.base_upload_dir / file_id
    
    def get_audio_file_path(self, file_id: str, file_extension: str = None) -> pathlib.Path:
        """Get the standardized path for the audio file."""
        job_dir = self.get_job_directory(file_id)
        
        # If extension not provided, try to get from metadata
        if file_extension is None:
            metadata = self.get_metadata_sync(file_id) or {}
            file_extension = metadata.get("file_extension", ".mp3")  # Default to .mp3 if not found
        
        return job_dir / f"original{file_extension}"
    
    def get_beats_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the beats file."""
        job_dir = self.get_job_directory(file_id)
        return job_dir / "beats.txt"
    
    def get_beat_stats_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the beat statistics file."""
        job_dir = self.get_job_directory(file_id)
        return job_dir / "beat_stats.json"
    
    def get_video_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the visualization video."""
        job_dir = self.get_job_directory(file_id)
        return job_dir / "video.mp4"
    
    def ensure_job_directory(self, file_id: str) -> pathlib.Path:
        """Ensure the job directory exists and return its path."""
        job_dir = self.get_job_directory(file_id)
        job_dir.mkdir(exist_ok=True, parents=True)
        return job_dir

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
        async_result = TaskStatusManager.create_async_result(task_id)
        if not async_result:
            return {"state": ERROR, "result": None, "error": "Failed to create AsyncResult"}
        
        try:
            # Get state and result without accessing backend methods directly
            state = "UNKNOWN"
            
            # First try to get the state from AsyncResult.state
            try:
                state = async_result.state
            except Exception as e:
                logger.error(f"Error getting task state: {e}")
                state = ERROR
            
            result = None
            
            # Only try to get result if task appears to be ready
            # and avoid calling .ready() which might cause another backend error
            if state in ["SUCCESS", "FAILURE"]:
                try:
                    if state == "SUCCESS":
                        result = async_result.result
                    else:
                        # For failed tasks, get the error message
                        error = str(async_result.result)
                        return {"state": state, "error": error}
                except Exception as e:
                    logger.error(f"Error getting task result: {e}")
                    return {"state": ERROR, "error": str(e)}
            
            # Create a standardized response
            result_dict = {"state": state}
            
            # Add result or structured data if available
            if result is not None:
                # Handle dictionary results
                if isinstance(result, dict):
                    # Copy any relevant fields from the result dict
                    if "error" in result:
                        result_dict["error"] = result["error"]
                    if "progress" in result:
                        result_dict["progress"] = result["progress"]
                    if "file_id" in result:
                        result_dict["file_id"] = result["file_id"]
                    if "video_file" in result:
                        result_dict["video_file"] = result["video_file"]
                    # Include the entire result dict
                    result_dict["result"] = result
                else:
                    # For non-dict results, just set as result
                    result_dict["result"] = result
            
            # Attempt to check Redis directly as a fallback
            if state == "UNKNOWN" or (state == "SUCCESS" and result is None):
                try:
                    from redis import Redis
                    redis_client = Redis(host='localhost', port=6379, db=0, decode_responses=True)
                    redis_key = f"celery-task-meta-{task_id}"
                    
                    raw_result = redis_client.get(redis_key)
                    if raw_result:
                        try:
                            import json
                            parsed = json.loads(raw_result)
                            if "status" in parsed:
                                result_dict["state"] = parsed["status"]
                            if "result" in parsed:
                                result_dict["result"] = parsed["result"]
                            logger.info(f"Got task result from Redis directly for task {task_id}")
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse Redis result: {raw_result}")
                except Exception as e:
                    logger.error(f"Error accessing Redis directly: {e}")
            
            return result_dict
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return {"state": ERROR, "error": str(e)}

class TaskStatusManager:
    """Class to manage task status retrieval."""
    
    @staticmethod
    def create_async_result(task_id: str):
        """Create an AsyncResult object for a task."""
        try:
            from celery.result import AsyncResult
            # Make sure the backend is properly specified
            from web_app.celery_app import app
            # Create AsyncResult using our app instance that has properly configured backend
            result = app.AsyncResult(task_id)
            return result
        except Exception as e:
            logger.error(f"Error creating AsyncResult: {e}")
            return None

    @staticmethod
    def extract_task_data(task_result) -> Dict[str, Any]:
        """Extract basic data from a task result."""
        try:
            # Check if task_result is None
            if task_result is None:
                return {"state": ERROR, "error": "Task result is None"}
                
            # Return minimal task information
            result = {"state": task_result.state}
            
            # Add result or error if available
            if task_result.ready():
                if task_result.successful():
                    result["result"] = task_result.result
                else:
                    result["error"] = str(task_result.result) 
            
            return result
        except Exception as e:
            logger.error(f"Error extracting task data: {e}")
            return {"state": ERROR, "error": str(e)}

    @staticmethod
    async def get_beat_detection_status(beat_detection_task_id: str) -> Dict[str, Any]:
        """Get the status of a beat detection task."""
        # Create AsyncResult
        task_result = TaskStatusManager.create_async_result(beat_detection_task_id)
        if not task_result:
            return {
                "id": beat_detection_task_id,
                "type": "beat_detection",
                "state": ERROR
            }
        
        # Get basic task data
        task_data = TaskStatusManager.extract_task_data(task_result)
        
        # Add task identification
        task_data["id"] = beat_detection_task_id
        task_data["type"] = "beat_detection"
        
        return task_data

    @staticmethod
    async def get_video_generation_status(video_task_id: str) -> Dict[str, Any]:
        """Get the status of a video generation task."""
        # Create AsyncResult
        task_result = TaskStatusManager.create_async_result(video_task_id)
        if not task_result:
            return {
                "id": video_task_id,
                "type": "video_generation",
                "state": ERROR
            }
        
        # Get basic task data
        task_data = TaskStatusManager.extract_task_data(task_result)
        
        # Add task identification
        task_data["id"] = video_task_id
        task_data["type"] = "video_generation"
        
        return task_data 