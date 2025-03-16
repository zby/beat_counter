"""Storage implementations for metadata management."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import logging
import redis

# Set up logger
logger = logging.getLogger(__name__)

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
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all files."""
        pass

    @abstractmethod
    def delete_metadata(self, file_id: str) -> bool:
        """Delete metadata for a specific file."""
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

    async def get_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific file."""
        try:
            redis_key = self._get_key(file_id)
            metadata_json = self.client.get(redis_key)
            
            if metadata_json:
                return json.loads(metadata_json)
            return None
        except Exception as e:
            logger.error(f"Error getting metadata from Redis: {e}")
            return None

    def update_metadata(self, file_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a specific file."""
        try:
            redis_key = self._get_key(file_id)
            # Get existing metadata or initialize empty dict
            existing = self.get_metadata(file_id) or {}
            
            # Deep update the metadata
            self._deep_update(existing, metadata)
            
            # Save updated metadata to Redis
            self.client.set(redis_key, json.dumps(existing), ex=604800)  # 7 days expiration
        except Exception as e:
            logger.error(f"Error updating metadata in Redis: {e}")
            raise

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all files."""
        try:
            all_metadata = {}
            
            # Get all keys matching the file metadata prefix
            for key in self.client.keys(f"{self.prefix}*"):
                file_id = key.replace(self.prefix, '')
                metadata = self.get_metadata(file_id)
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
        from celery.result import AsyncResult
        return AsyncResult(task_id)

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
        """Extract metadata from a task result."""
        try:
            # Get the task metadata
            task_metadata = {}
            
            # Add the task state
            task_metadata["state"] = task_result.state
            
            # Add the task result if available
            if task_result.ready():
                if task_result.successful():
                    task_metadata["result"] = task_result.result
                else:
                    task_metadata["error"] = str(task_result.result)
            
            return task_metadata
            
        except Exception as e:
            logger.error(f"Error extracting task metadata: {e}")
            return {}

    @staticmethod
    def get_attribute(task_result, attr: str) -> Any:
        """Safely get an attribute from a task result."""
        try:
            return getattr(task_result, attr)
        except Exception as e:
            logger.error(f"Error getting task attribute {attr}: {e}")
            return None

    @staticmethod
    async def get_beat_detection_status(beat_detection_task_id: str) -> Dict[str, Any]:
        """Get the status of a beat detection task."""
        # Create AsyncResult
        task_result = TaskStatusManager.create_async_result(beat_detection_task_id)
        if not task_result:
            # Return minimal information if task result cannot be created
            return {
                "id": beat_detection_task_id,
                "type": "beat_detection",
                "state": "UNKNOWN"
            }
        
        # Extract the task metadata
        try:
            # Get the task metadata
            task_metadata = TaskStatusManager.extract_task_data(task_result)
            
            # Ensure the metadata is a dictionary and has basic required fields
            if not isinstance(task_metadata, dict):
                task_metadata = {}
                
            # Add essential fields if they don't exist
            task_metadata["id"] = beat_detection_task_id
            task_metadata["type"] = "beat_detection"
            
            # Add the Celery state - this is the only status we need
            task_metadata["state"] = TaskStatusManager.get_attribute(task_result, "state")
                
            return task_metadata
            
        except Exception as e:
            logger.error(f"Error extracting beat detection metadata: {e}")
            # Return minimal information on error
            return {
                "id": beat_detection_task_id,
                "type": "beat_detection",
                "state": "FAILURE",
                "error": str(e)
            }

    @staticmethod
    async def get_video_generation_status(video_task_id: str) -> Dict[str, Any]:
        """Get the status of a video generation task."""
        # Create AsyncResult
        task_result = TaskStatusManager.create_async_result(video_task_id)
        if not task_result:
            # Return minimal information if task result cannot be created
            return {
                "id": video_task_id,
                "type": "video_generation",
                "state": "UNKNOWN"
            }
        
        # Extract the task metadata
        try:
            # Get the task metadata
            task_metadata = TaskStatusManager.extract_task_data(task_result)
            
            # Ensure the metadata is a dictionary and has basic required fields
            if not isinstance(task_metadata, dict):
                task_metadata = {}
                
            # Add essential fields if they don't exist
            task_metadata["id"] = video_task_id
            task_metadata["type"] = "video_generation"
            
            # Add the Celery state - this is the only status we need
            task_metadata["state"] = TaskStatusManager.get_attribute(task_result, "state")
                
            return task_metadata
            
        except Exception as e:
            logger.error(f"Error extracting video generation metadata: {e}")
            # Return minimal information on error
            return {
                "id": video_task_id,
                "type": "video_generation",
                "state": "FAILURE",
                "error": str(e)
            } 