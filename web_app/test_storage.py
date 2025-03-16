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
        self.storage[file_id].update(metadata)
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all files."""
        return self.storage.copy()

class MockTask:
    """Mock task result for testing."""
    def __init__(self, task_id: str, state: str = "PENDING"):
        self.id = task_id
        self._state = state
        self._result = None
    
    @property
    def state(self) -> str:
        return self._state
    
    def set_state(self, state: str, result: Any = None) -> None:
        self._state = state
        self._result = result
    
    @property
    def result(self) -> Any:
        return self._result

class MockTaskExecutor(TaskExecutor):
    """Mock implementation of task executor for testing."""
    
    def __init__(self):
        self.tasks = {}
        self.task_counter = 0
    
    def _create_task(self) -> MockTask:
        """Create a new mock task."""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        task = MockTask(task_id)
        self.tasks[task_id] = task
        return task
    
    def execute_beat_detection(self, file_id: str, file_path: str) -> MockTask:
        """Execute beat detection task."""
        task = self._create_task()
        # Simulate task execution
        task.set_state("STARTED")
        return task
    
    def execute_video_generation(self, file_id: str, file_path: str, beats_file: str) -> MockTask:
        """Execute video generation task."""
        task = self._create_task()
        # Simulate task execution
        task.set_state("STARTED")
        return task
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task."""
        task = self.tasks.get(task_id)
        if not task:
            return {"state": "UNKNOWN"}
        return {
            "id": task.id,
            "state": task.state,
            "result": task.result
        }
    
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