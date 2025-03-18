"""Mock implementations of storage and task executor for testing."""

import pytest
from typing import Any, Dict, Optional
from web_app.storage import MetadataStorage
from web_app.task_executor import ANALYZING, ANALYZED, ANALYZING_FAILURE, GENERATING_VIDEO, COMPLETED, VIDEO_ERROR, ERROR
import pathlib
from datetime import datetime

class MockTask:
    """Mock implementation of a task."""
    
    def __init__(self):
        self.id = str(id(self))
        self.state = "STARTED"
        self.result = None
    
    def set_state(self, state: str, result: Any = None) -> None:
        """Set the task state and result."""
        self.state = state
        self.result = result
        # Make sure the task is updated in the MOCK_TASKS dictionary
        from web_app.test_app import MOCK_TASKS
        # Always update the task in MOCK_TASKS, regardless of whether it's already there
        MOCK_TASKS[self.id] = self

class MockMetadataStorage(MetadataStorage):
    """In-memory implementation of metadata storage for testing."""
    
    def __init__(self):
        self.storage = {}
        self.get_file_status_response = None
        self.base_upload_dir = pathlib.Path("web_app/uploads")
    
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
    
    # Path management methods
    def get_job_directory(self, file_id: str) -> pathlib.Path:
        """Get the standardized job directory for a file ID."""
        return self.base_upload_dir / file_id
    
    def get_audio_file_path(self, file_id: str, file_extension: str = None) -> pathlib.Path:
        """Get the standardized path for the audio file."""
        job_dir = self.get_job_directory(file_id)
        
        # If extension not provided, try to get from metadata
        if file_extension is None:
            metadata = self.storage.get(file_id, {})
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

    async def get_file_status(self, file_id: str) -> Dict[str, Any]:
        """Get the processing status for a file."""
        # If a custom response is set, return that instead
        if self.get_file_status_response is not None:
            return self.get_file_status_response
            
        # Import here to avoid circular imports
        from web_app.test_app import mock_get_task_status, MOCK_TASKS
        
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
            beat_task_status = mock_get_task_status(beat_task_id)
            status_data["beat_detection_task"] = beat_task_status

            if beat_task_status["state"] == "SUCCESS":
                overall_status = ANALYZED
            elif beat_task_status["state"] == "FAILURE":
                overall_status = ANALYZING_FAILURE
            else:
                overall_status = ANALYZING

        if video_task_id:
            video_task_status = mock_get_task_status(video_task_id)
            status_data["video_generation_task"] = video_task_status

            if video_task_status["state"] == "SUCCESS":
                overall_status = COMPLETED
            elif video_task_status["state"] == "FAILURE":
                overall_status = VIDEO_ERROR
            else:
                overall_status = GENERATING_VIDEO

        # Add overall status to response
        status_data["status"] = overall_status

        return status_data

class MockTaskExecutor:
    """Mock implementation of task executor for testing."""
    
    def __init__(self):
        self.tasks = {}
    
    def _create_task(self) -> MockTask:
        """Create a new mock task."""
        task = MockTask()
        self.tasks[task.id] = task
        return task
    
    def execute_beat_detection(self, file_id: str) -> MockTask:
        """Execute beat detection task."""
        task = self._create_task()
        task.state = "STARTED"
        return task
    
    def execute_video_generation(self, file_id: str) -> MockTask:
        """Execute video generation task."""
        task = self._create_task()
        task.state = "STARTED"
        return task
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task."""
        task = self.tasks.get(task_id)
        if not task:
            return {"id": task_id, "state": ERROR, "error": "Task not found"}
        
        result_dict = {"id": task_id, "state": task.state}
        
        # Add result for success or error for failure
        if task.state == "SUCCESS" and task.result is not None:
            result_dict["result"] = task.result
        elif task.state == "FAILURE" and task.result is not None:
            result_dict["error"] = task.result
            
        return result_dict
    
    async def get_beat_detection_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a beat detection task."""
        task = self.tasks.get(task_id)
        if not task:
            return {
                "id": task_id,
                "type": "beat_detection",
                "state": ERROR,
                "error": "Task not found"
            }
        
        # Create basic task data
        task_data = {
            "id": task_id,
            "type": "beat_detection",
            "state": task.state
        }
        
        # Add result or error based on state
        if task.state == "SUCCESS" and task.result is not None:
            task_data["result"] = task.result
        elif task.state == "FAILURE" and task.result is not None:
            task_data["error"] = task.result
        
        return task_data
    
    async def get_video_generation_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a video generation task."""
        task = self.tasks.get(task_id)
        if not task:
            return {
                "id": task_id,
                "type": "video_generation",
                "state": ERROR,
                "error": "Task not found"
            }
        
        # Create basic task data
        task_data = {
            "id": task_id,
            "type": "video_generation",
            "state": task.state
        }
        
        # Add result or error based on state
        if task.state == "SUCCESS" and task.result is not None:
            task_data["result"] = task.result
        elif task.state == "FAILURE" and task.result is not None:
            task_data["error"] = task.result
        
        return task_data
    
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
    assert task.state == "STARTED"
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
    assert status["state"] == "STARTED"
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
    assert status["state"] == "ERROR"
    assert "result" in status
    assert status["result"] is None

@pytest.mark.asyncio
async def test_get_file_status(mock_storage: MockMetadataStorage, mock_executor: MockTaskExecutor):
    """Test getting file status from mock storage."""
    # Create file metadata
    file_id = "test-file-status"
    
    # Get standardized file paths
    audio_file_path = mock_storage.get_audio_file_path(file_id, ".mp3")
    beats_file = mock_storage.get_beats_file_path(file_id)
    stats_file = mock_storage.get_beat_stats_file_path(file_id)
    video_file = mock_storage.get_video_file_path(file_id)
    
    metadata = {
        "original_filename": "test.mp3",
        "audio_file_path": str(audio_file_path),
        "file_extension": ".mp3",
        "upload_time": datetime.now().isoformat()
    }
    mock_storage.update_metadata(file_id, metadata)

    # Test with only beat detection task running
    beat_task = mock_executor.execute_beat_detection(file_id)
    
    # Update metadata with task ID
    mock_storage.update_metadata(file_id, {"beat_detection": beat_task.id})
    
    # Get file status
    status = await mock_storage.get_file_status(file_id)
    
    # Verify status structure
    assert status["file_id"] == file_id
    assert status["filename"] == "test.mp3"
    assert status["status"] == ANALYZING
    
    # Complete beat detection task
    mock_executor.complete_task(beat_task.id, {
        "file_id": file_id,
        "beats_file": str(beats_file),
        "stats_file": str(stats_file)
    })
    
    # Get updated status
    status = await mock_storage.get_file_status(file_id)
    assert status["status"] == ANALYZED
    
    # Test with video generation task
    video_task = mock_executor.execute_video_generation(file_id)
    
    # Update metadata with task ID
    mock_storage.update_metadata(file_id, {"video_generation": video_task.id})
    
    # Get updated status
    status = await mock_storage.get_file_status(file_id)
    assert status["status"] == GENERATING_VIDEO
    
    # Complete video generation task
    mock_executor.complete_task(video_task.id, {
        "file_id": file_id,
        "video_file": str(video_file)
    })
    
    # Get final status
    status = await mock_storage.get_file_status(file_id)
    assert status["status"] == COMPLETED

def test_get_task_status_not_found(mock_executor: MockTaskExecutor):
    """Test getting status of non-existent task."""
    status = mock_executor.get_task_status("nonexistent")
    assert status["state"] == "ERROR" 