"""Storage implementations for metadata management."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import logging
import redis
import os

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

class TaskExecutor(ABC):
    """Abstract base class for task execution."""
    
    @abstractmethod
    def execute_beat_detection(self, file_id: str, file_path: str) -> Any:
        """Execute beat detection task."""
        pass
    
    @abstractmethod
    def execute_video_generation(self, file_id: str, file_path: str, beats_file: str) -> Any:
        """Execute video generation task."""
        pass
    
    @abstractmethod
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task."""
        pass

class FileIDNotFoundError(Exception):
    """Exception raised when a file is not found in the storage."""
    def __init__(self, file_id: str):
        self.file_id = file_id
        super().__init__(f"File with ID {file_id} not found.")

class RedisMetadataStorage(MetadataStorage):
    """Redis implementation of metadata storage."""
    
    def __init__(self, host='localhost', port=6379, db=0, prefix='file_metadata:'):
        """Initialize Redis connection."""
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.prefix = prefix
        
        # Test the connection
        try:
            self.client.ping()
            logger.info("Redis connection successful")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise RuntimeError("Redis connection failed. Make sure Redis is running.")

    def _get_key(self, file_id: str) -> str:
        """Get the Redis key for a file ID."""
        return f"{self.prefix}{file_id}"

    def _get_metadata_sync(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous version of get_metadata for internal use."""
        try:
            redis_key = self._get_key(file_id)
            metadata_json = self.client.get(redis_key)
            
            if metadata_json:
                return json.loads(metadata_json)
            return None
        except Exception as e:
            logger.error(f"Error getting metadata from Redis: {e}")
            return None

    async def get_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific file."""
        return self._get_metadata_sync(file_id)

    def update_metadata(self, file_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a specific file."""
        try:
            redis_key = self._get_key(file_id)
            # Get existing metadata or initialize empty dict
            existing = self._get_metadata_sync(file_id) or {}
            
            # Deep update the metadata
            self._deep_update(existing, metadata)
            
            # Save updated metadata to Redis
            self.client.set(redis_key, json.dumps(existing), ex=604800)  # 7 days expiration
        except Exception as e:
            logger.error(f"Error updating metadata in Redis: {e}")
            raise

    async def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all files."""
        try:
            all_metadata = {}
            
            # Get all keys matching the file metadata prefix
            for key in self.client.keys(f"{self.prefix}*"):
                file_id = key.replace(self.prefix, '')
                metadata = await self.get_metadata(file_id)
                if metadata:
                    all_metadata[file_id] = metadata
            
            return all_metadata
        except Exception as e:
            logger.error(f"Error getting all metadata from Redis: {e}")
            return {}

    def delete_metadata(self, file_id: str) -> bool:
        """Delete metadata for a specific file."""
        try:
            redis_key = self._get_key(file_id)
            return self.client.delete(redis_key) > 0
        except Exception as e:
            logger.error(f"Error deleting metadata from Redis: {e}")
            return False

    def _deep_update(self, target: Dict, source: Dict) -> None:
        """Helper function for deep updating dictionaries."""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value

    def _parse_beat_stats_file(self, stats_file_path: str) -> Dict[str, Any]:
        """Parse beat statistics from a JSON file.
        
        Args:
            stats_file_path: Path to the beat stats JSON file
            
        Returns:
            Dict containing the parsed beat statistics
        """
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

    async def get_file_status(self, file_id: str, executor: 'TaskExecutor') -> Dict[str, Any]:
        """Get the processing status for a file."""
        metadata = await self.get_metadata(file_id)
        if not metadata:
            raise FileIDNotFoundError(file_id)

        # Get task statuses
        beat_task_id = metadata.get("beat_detection")
        video_task_id = metadata.get("video_generation")

        status_data = {
            "file_id": file_id,
            "filename": metadata.get("filename"),
            "file_path": metadata.get("file_path")
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
        file_path = metadata.get("file_path")
        if file_path:
            # Construct the expected path to the beat stats file
            filename_with_ext = os.path.basename(file_path)
            filename_without_ext = os.path.splitext(filename_with_ext)[0]
            upload_dir = os.path.dirname(file_path)
            stats_filename = f"{filename_without_ext}_beat_stats.json"
            stats_file_path = os.path.join(upload_dir, stats_filename)
            
            # Parse the beat stats file
            stats = self._parse_beat_stats_file(stats_file_path)
            if stats:
                status_data["beat_stats"] = stats

        return status_data

class CeleryTaskExecutor(TaskExecutor):
    """Celery implementation of task executor."""
    def __init__(self):
        from web_app.tasks import detect_beats_task, generate_video_task
        self.detect_beats_task = detect_beats_task
        self.generate_video_task = generate_video_task
    
    def execute_beat_detection(self, file_id: str, file_path: str) -> Any:
        return self.detect_beats_task.delay(file_id, file_path)
    
    def execute_video_generation(self, file_id: str, file_path: str, beats_file: str) -> Any:
        return self.generate_video_task.delay(file_id, file_path, beats_file)
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the raw status of a Celery task.
        
        Returns essential information from the Celery AsyncResult.
        """
        from celery.result import AsyncResult
        task_result = AsyncResult(task_id)
        
        # Return basic task information
        return {
            "state": task_result.state,
            "result": task_result.result if task_result.ready() and task_result.successful() else None,
            "error": str(task_result.result) if task_result.ready() and not task_result.successful() else None
        }

class TaskStatusManager:
    """Class to manage task status retrieval."""
    
    @staticmethod
    def create_async_result(task_id: str):
        """Create an AsyncResult object for a task."""
        try:
            from celery.result import AsyncResult
            return AsyncResult(task_id)
        except Exception as e:
            logger.error(f"Error creating AsyncResult: {e}")
            return None

    @staticmethod
    def extract_task_data(task_result) -> Dict[str, Any]:
        """Extract basic data from a task result."""
        try:
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