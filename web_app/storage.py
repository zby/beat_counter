from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

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
    def __init__(self, redis_client=None):
        # Import here to avoid circular imports
        from web_app.metadata import get_file_metadata, get_all_file_metadata, update_file_metadata
        self._get_file_metadata = get_file_metadata
        self._get_all_file_metadata = get_all_file_metadata
        self._update_file_metadata = update_file_metadata
    
    async def get_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        return await self._get_file_metadata(file_id)
    
    def update_metadata(self, file_id: str, metadata: Dict[str, Any]) -> None:
        self._update_file_metadata(file_id, metadata)
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        return self._get_all_file_metadata()

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