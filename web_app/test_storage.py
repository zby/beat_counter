"""Mock implementations of storage and task executor for testing."""

import pytest
from typing import Any, Dict, Optional
from web_app.storage import MetadataStorage, TaskExecutor, FileIDNotFoundError

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
    
    async def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
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

    async def get_file_status(self, file_id: str, executor: TaskExecutor) -> Dict[str, Any]:
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
        overall_status = "UNKNOWN"

        if beat_task_id:
            beat_task_status = executor.get_task_status(beat_task_id)
            status_data["beat_detection_task"] = beat_task_status

            if beat_task_status["state"] == "SUCCESS":
                overall_status = "ANALYZED"
            elif beat_task_status["state"] == "FAILURE":
                overall_status = "FAILED"
            else:
                overall_status = "ANALYZING"

        if video_task_id:
            video_task_status = executor.get_task_status(video_task_id)
            status_data["video_generation_task"] = video_task_status

            if video_task_status["state"] == "SUCCESS":
                overall_status = "COMPLETED"
            elif video_task_status["state"] == "FAILURE":
                overall_status = "FAILED"
            else:
                overall_status = "GENERATING_VIDEO"

        # Add overall status to response
        status_data["status"] = overall_status

        return status_data

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
            return {"state": "UNKNOWN", "result": None}
        
        return {
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

@pytest.mark.asyncio
async def test_mock_metadata_storage():
    """Test the methods of MockMetadataStorage class."""
    storage = MockMetadataStorage()
    
    # Test initial state
    assert storage.storage == {}
    
    # Test get_metadata for non-existent file
    metadata = await storage.get_metadata("nonexistent")
    assert metadata is None
    
    # Test update_metadata for new file
    file_id = "test_file"
    initial_metadata = {
        "filename": "test.mp3",
        "file_path": "/path/to/test.mp3"
    }
    storage.update_metadata(file_id, initial_metadata)
    
    # Verify metadata was stored
    metadata = await storage.get_metadata(file_id)
    assert metadata == initial_metadata
    
    # Test deep update of metadata
    update_metadata = {
        "task_info": {
            "beat_detection": "task1",
            "status": "STARTED"
        }
    }
    storage.update_metadata(file_id, update_metadata)
    
    # Verify deep update worked
    metadata = await storage.get_metadata(file_id)
    assert metadata["filename"] == "test.mp3"  # Original data preserved
    assert metadata["task_info"]["beat_detection"] == "task1"  # New data added
    
    # Test get_all_metadata
    all_metadata = await storage.get_all_metadata()
    assert len(all_metadata) == 1
    assert all_metadata[file_id] == metadata
    
    # Test delete_metadata
    assert storage.delete_metadata(file_id) is True
    assert storage.delete_metadata("nonexistent") is False
    assert await storage.get_metadata(file_id) is None

def test_mock_task_methods():
    """Test the methods of MockTask class."""
    task = MockTask()
    
    # Test initial state
    assert task.state == "PENDING"
    assert task.result is None
    assert isinstance(task.id, str)
    
    # Test state transitions
    task.set_state("STARTED")
    assert task.state == "STARTED"
    assert task.result is None
    
    task.set_state("SUCCESS", {"data": "test"})
    assert task.state == "SUCCESS"
    assert task.result == {"data": "test"}
    
    task.set_state("FAILURE", "error")
    assert task.state == "FAILURE"
    assert task.result == "error"

def test_mock_task_executor_methods():
    """Test the helper methods of MockTaskExecutor."""
    executor = MockTaskExecutor()
    
    # Test task creation
    task = executor._create_task()
    assert task.id in executor.tasks
    assert executor.tasks[task.id] is task
    
    # Test task completion helper
    executor.complete_task(task.id, {"result": "success"})
    assert task.state == "SUCCESS"
    assert task.result == {"result": "success"}
    
    # Test task failure helper
    task2 = executor._create_task()
    executor.fail_task(task2.id, "test error")
    assert task2.state == "FAILURE"
    assert task2.result == "test error"
    
    # Test with non-existent task
    executor.complete_task("nonexistent")  # Should not raise error
    executor.fail_task("nonexistent")  # Should not raise error

def test_task_status_structure():
    """Test that get_task_status returns a properly structured response."""
    executor = MockTaskExecutor()
    
    # Create a task and test different states
    task = executor._create_task()
    
    # Test PENDING state (initial state)
    status = executor.get_task_status(task.id)
    assert isinstance(status, dict)
    assert "state" in status
    assert status["state"] == "PENDING"
    assert "result" in status
    assert status["result"] is None
    
    # Test STARTED state
    task.set_state("STARTED")
    status = executor.get_task_status(task.id)
    assert status["state"] == "STARTED"
    assert status["result"] is None
    
    # Test SUCCESS state with result
    result_data = {"beats_file": "beats.json"}
    task.set_state("SUCCESS", result_data)
    status = executor.get_task_status(task.id)
    assert status["state"] == "SUCCESS"
    assert status["result"] == result_data
    
    # Test FAILURE state with error message
    error_msg = "Task failed with error"
    task.set_state("FAILURE", error_msg)
    status = executor.get_task_status(task.id)
    assert status["state"] == "FAILURE"
    assert status["result"] == error_msg
    
    # Test non-existent task
    status = executor.get_task_status("nonexistent")
    assert isinstance(status, dict)
    assert "state" in status
    assert status["state"] == "UNKNOWN"
    assert "result" in status
    assert status["result"] is None

@pytest.mark.asyncio
async def test_get_file_status(mock_storage: MockMetadataStorage, mock_executor: MockTaskExecutor):
    """Test the get_file_status method in MockMetadataStorage."""
    file_id = "test_file"
    metadata = {
        "filename": "test.mp3",
        "file_path": "/path/to/test.mp3"
    }
    
    # Test with no tasks
    mock_storage.update_metadata(file_id, metadata)
    status_data = await mock_storage.get_file_status(file_id, mock_executor)
    assert status_data["status"] == "UNKNOWN"
    
    # Test with beat detection task in progress
    beat_task = mock_executor._create_task()
    beat_task.set_state("STARTED")
    metadata["beat_detection"] = beat_task.id
    mock_storage.update_metadata(file_id, metadata)
    status_data = await mock_storage.get_file_status(file_id, mock_executor)
    assert status_data["status"] == "ANALYZING"
    
    # Test with beat detection task completed
    beat_task.set_state("SUCCESS")
    status_data = await mock_storage.get_file_status(file_id, mock_executor)
    assert status_data["status"] == "ANALYZED"
    
    # Test with video generation task in progress
    video_task = mock_executor._create_task()
    video_task.set_state("STARTED")
    metadata["video_generation"] = video_task.id
    mock_storage.update_metadata(file_id, metadata)
    status_data = await mock_storage.get_file_status(file_id, mock_executor)
    assert status_data["status"] == "GENERATING_VIDEO"
    
    # Test with video generation task completed
    video_task.set_state("SUCCESS")
    status_data = await mock_storage.get_file_status(file_id, mock_executor)
    assert status_data["status"] == "COMPLETED"
    
    # Test with failed beat detection task
    beat_task.set_state("FAILURE")
    status_data = await mock_storage.get_file_status(file_id, mock_executor)
    assert status_data["status"] == "FAILED"
    
    # Test with failed video generation task
    beat_task.set_state("SUCCESS")  # Reset beat task to success
    video_task.set_state("FAILURE")
    status_data = await mock_storage.get_file_status(file_id, mock_executor)
    assert status_data["status"] == "FAILED" 