"""Mock implementations of storage and task executor for testing."""

from typing import Any, Dict, Optional
from web_app.storage import MetadataStorage, TaskExecutor

class MockMetadataStorage(MetadataStorage):
    """In-memory implementation of metadata storage for testing."""
    
    def __init__(self):
        self.storage = {}
    
    async def get_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific file."""
        return self.storage.get(file_id)
    
    def update_metadata(self, file_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a specific file."""
        if file_id not in self.storage:
            self.storage[file_id] = {}
        self._deep_update(self.storage[file_id], metadata)
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all files."""
        return self.storage.copy()

    def delete_metadata(self, file_id: str) -> bool:
        """Delete metadata for a specific file."""
        if file_id in self.storage:
            del self.storage[file_id]
            return True
        return False

    def _deep_update(self, target: Dict, source: Dict) -> None:
        """Helper function for deep updating dictionaries."""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value

class MockTask:
    """Mock implementation of a task."""
    
    def __init__(self):
        self.id = str(id(self))
        self.state = "PENDING"
        self.result = None
    
    def set_state(self, state: str, result: Any = None) -> None:
        """Set the task state and result."""
        self.state = state
        self.result = result

class MockTaskExecutor(TaskExecutor):
    """Mock implementation of task executor for testing."""
    
    def __init__(self):
        self.tasks = {}
    
    def _create_task(self) -> MockTask:
        """Create a new mock task."""
        task = MockTask()
        self.tasks[task.id] = task
        return task
    
    def execute_beat_detection(self, file_id: str, file_path: str) -> MockTask:
        """Execute beat detection on a file."""
        task = self._create_task()
        task.set_state("STARTED")
        return task
    
    def execute_video_generation(self, file_id: str, file_path: str, beats_file: str) -> MockTask:
        """Execute video generation for a file."""
        task = self._create_task()
        task.set_state("STARTED")
        return task
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a task."""
        task = self.tasks.get(task_id)
        if not task:
            return {"state": "UNKNOWN"}
        
        status = {"state": task.state}
        if task.result is not None:
            status["result"] = task.result
        
        return status
    
    def complete_task(self, task_id: str, result: Any = None) -> None:
        """Complete a task with success state."""
        task = self.tasks.get(task_id)
        if task:
            task.set_state("SUCCESS", result)
    
    def fail_task(self, task_id: str, error: str = "Task failed") -> None:
        """Fail a task with error state."""
        task = self.tasks.get(task_id)
        if task:
            task.set_state("FAILURE", error) 