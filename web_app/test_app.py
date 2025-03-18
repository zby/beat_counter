"""Tests for the FastAPI application."""

import os
import pathlib
import tempfile
from typing import Dict, Generator, List, Optional, Tuple, Any
import pytest
from fastapi.testclient import TestClient
from web_app.app import create_app, TaskServiceProvider
from web_app.app import ANALYZING, ANALYZED, ANALYZING_FAILURE, GENERATING_VIDEO, COMPLETED, VIDEO_ERROR, ERROR
from beat_detection.utils.constants import AUDIO_EXTENSIONS
from web_app.storage import MetadataStorage, FileMetadataStorage
from web_app.test_storage import MockMetadataStorage
import uuid
from datetime import datetime
import json

# Mock functions for task operations
MOCK_TASKS = {}

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
        # Always update the task in MOCK_TASKS, regardless of whether it's already there
        MOCK_TASKS[self.id] = self

def mock_get_task_status(task_id: str) -> Dict[str, Any]:
    """Mock implementation of get_task_status for testing."""
    task = MOCK_TASKS.get(task_id)
    if not task:
        return {"id": task_id, "state": ERROR, "error": "Task not found"}
    
    result_dict = {"id": task_id, "state": task.state}
    
    # Add result for success or error for failure
    if task.state == "SUCCESS" and task.result is not None:
        result_dict["result"] = task.result
    elif task.state == "FAILURE" and task.result is not None:
        result_dict["error"] = task.result
        
    return result_dict

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
            return {"id": task_id, "state": ERROR, "error": "Task not found", "result": None}
        
        result_dict = {"id": task_id, "state": task.state, "result": None}
        
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
                "error": "Task not found",
                "result": None
            }
        
        # Create basic task data
        task_data = {
            "id": task_id,
            "type": "beat_detection",
            "state": task.state,
            "result": None
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
                "error": "Task not found",
                "result": None
            }
        
        # Create basic task data
        task_data = {
            "id": task_id,
            "type": "video_generation",
            "state": task.state,
            "result": None
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
            # Update the task in MOCK_TASKS to reflect the changes
            MOCK_TASKS[task_id] = task
    
    def fail_task(self, task_id: str, error: str = "Task failed") -> None:
        """Fail a task with error state."""
        task = self.tasks.get(task_id)
        if task:
            task.set_state("FAILURE", error)

def create_mock_task(state: str = "STARTED", result: Any = None) -> MockTask:
    """Create a mock task for testing."""
    task = MockTask()
    task.state = state
    task.result = result
    MOCK_TASKS[task.id] = task
    return task

def complete_mock_task(task_id: str, result: Any = None) -> None:
    """Complete a mock task with success state."""
    task = MOCK_TASKS.get(task_id)
    if task:
        task.set_state("SUCCESS", result)

def fail_mock_task(task_id: str, error: str = "Task failed") -> None:
    """Fail a mock task with error state."""
    task = MOCK_TASKS.get(task_id)
    if task:
        task.set_state("FAILURE", error)

# Mock task service provider for testing
class MockTaskServiceProvider(TaskServiceProvider):
    """Mock implementation of TaskServiceProvider for testing."""
    
    def __init__(self):
        """Initialize with mock implementations."""
        super().__init__(
            get_task_status_fn=mock_get_task_status,
            detect_beats_fn=self.mock_detect_beats,
            generate_video_fn=self.mock_generate_video
        )
    
    def mock_detect_beats(self, file_id: str) -> MockTask:
        """Mock implementation for beat detection."""
        task = create_mock_task()
        task.task_type = "beat_detection"
        task.file_id = file_id
        return task
    
    def mock_generate_video(self, file_id: str) -> MockTask:
        """Mock implementation for video generation."""
        task = create_mock_task()
        task.task_type = "video_generation"
        task.file_id = file_id
        return task

# Test data
TEST_FILES = [
    ("test.mp3", "audio/mpeg", True, 303),  # Valid audio file - expect redirect
    ("test.wav", "audio/wav", True, 303),   # Valid audio file - expect redirect
    ("test.txt", "text/plain", False, 400), # Invalid file type - expect error
]

@pytest.fixture(scope="session")
def test_dir() -> Generator[pathlib.Path, None, None]:
    """Create and manage a test directory for file uploads."""
    test_dir = pathlib.Path(__file__).parent / "test_files"
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Cleanup after all tests
    import shutil
    shutil.rmtree(test_dir)

@pytest.fixture
def mock_storage() -> MockMetadataStorage:
    """Create a fresh mock storage instance for each test."""
    return MockMetadataStorage()

@pytest.fixture
def mock_task_service() -> MockTaskServiceProvider:
    """Create a mock task service provider for testing."""
    return MockTaskServiceProvider()

@pytest.fixture
def mock_executor() -> MockTaskExecutor:
    """Create a mock task executor for testing."""
    return MockTaskExecutor()

@pytest.fixture
def test_client(mock_storage: MockMetadataStorage, mock_task_service: MockTaskServiceProvider) -> TestClient:
    """Create a test client with mock dependencies."""
    # Create app with dependency injection
    app = create_app(
        metadata_storage=mock_storage,
        task_provider=mock_task_service
    )
    
    return TestClient(app)

@pytest.fixture
def test_file(test_dir: pathlib.Path, request: pytest.FixtureRequest) -> pathlib.Path:
    """Create a test file with the specified name."""
    filename = getattr(request, "param", "test.mp3")
    file_path = test_dir / filename
    file_path.touch()
    return file_path

@pytest.fixture
def sample_file_with_task(mock_storage: MockMetadataStorage) -> Tuple[str, str, MockTask]:
    """Create a sample file entry with an associated task."""
    file_id = "test_file"
    
    # Get standardized file path using the storage API
    audio_file_path = mock_storage.get_audio_file_path(file_id, ".mp3")
    
    # Create metadata with standardized structure
    metadata = {
        "original_filename": "test.mp3",
        "audio_file_path": str(audio_file_path),
        "file_extension": ".mp3",
        "upload_time": datetime.now().isoformat()
    }
    
    # Create mock task and update metadata
    task = create_mock_task()
    metadata["beat_detection"] = task.id
    
    # Store metadata
    mock_storage.update_metadata(file_id, metadata)
    
    return file_id, task.id, task

@pytest.mark.parametrize("filename,content_type,is_valid,expected_status", TEST_FILES)
def test_upload_audio(
    test_client: TestClient,
    mock_storage: MockMetadataStorage,
    test_file: pathlib.Path,
    filename: str,
    content_type: str,
    is_valid: bool,
    expected_status: int
):
    """Test uploading audio files with various formats."""
    with open(test_file, "rb") as f:
        response = test_client.post(
            "/upload",
            files={"file": (filename, f, content_type)},
            data={"analyze": "true"},
            follow_redirects=False  # Don't follow redirects to check status code
        )
    
    assert response.status_code == expected_status
    
    if is_valid:
        # Get file ID from redirect URL
        file_id = response.headers["location"].split("/")[-1]
        
        # Verify metadata storage
        metadata = mock_storage.storage.get(file_id)
        assert metadata is not None
        assert metadata["original_filename"] == filename
        
        # Verify file_extension is correctly stored
        file_extension = pathlib.Path(filename).suffix.lower()
        assert metadata["file_extension"] == file_extension
        
        # Verify the standardized file path structure
        audio_file_path = metadata["audio_file_path"]
        
        # Verify that beat detection task was started
        assert "beat_detection" in metadata
        task_id = metadata["beat_detection"]
        assert task_id in MOCK_TASKS
    else:
        assert "Unsupported file format" in response.json()["detail"]

@pytest.mark.asyncio
async def test_get_file_status_success(
    test_client: TestClient,
    sample_file_with_task: Tuple[str, str, MockTask]
):
    """Test getting file status for a successful task."""
    file_id, task_id, task = sample_file_with_task
    
    # Set task as successful
    task.set_state("SUCCESS", {"beats": [1.0, 2.0, 3.0]})
    
    # Get status
    response = test_client.get(f"/status/{file_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["beat_detection_task"]["state"] == "SUCCESS"
    assert data["beat_detection_task"]["result"]["beats"] == [1.0, 2.0, 3.0]

@pytest.mark.asyncio
async def test_get_file_status_failure(
    test_client: TestClient,
    sample_file_with_task: Tuple[str, str, MockTask]
):
    """Test getting file status for a failed task."""
    file_id, task_id, task = sample_file_with_task
    
    # Set task as failed
    task.set_state("FAILURE", "Processing error")
    
    # Get status
    response = test_client.get(f"/status/{file_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["beat_detection_task"]["state"] == "FAILURE"
    assert data["beat_detection_task"]["error"] == "Processing error"

def test_file_not_found(test_client: TestClient):
    """Test handling of non-existent files."""
    response = test_client.get("/status/nonexistent")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_confirm_analysis_success(
    test_client: TestClient,
    mock_storage: MockMetadataStorage
):
    """Test confirming analysis and generating a visualization video.
    
    This test checks that:
    1. The confirmation endpoint returns 200 when a file has completed beat analysis
    2. A video generation task is created and its ID is stored in metadata
    3. The status is correctly updated to show video generation in progress
    """
    # Setup file with completed beat detection
    file_id = "confirm-test-file"
    
    # Create a mock successful beat detection task
    beat_task = create_mock_task(state="SUCCESS", result={"beats_file": "beats.txt"})
    
    # Create the actual beats file
    beats_file = mock_storage.get_beats_file_path(file_id)
    beats_file.parent.mkdir(exist_ok=True, parents=True)
    beats_file.touch()
    
    # Setup metadata - minimum required for the endpoint to work
    mock_storage.update_metadata(file_id, {
        "original_filename": "test.mp3",
        "beat_detection": beat_task.id,
        "beats_file": str(beats_file)
    })
    
    # Print the metadata to debug
    print(f"Metadata before confirm: {mock_storage.storage.get(file_id)}")
    
    # Make sure the beats file exists
    print(f"Beats file exists: {beats_file.exists()}")
    
    # Call the confirm endpoint to start video generation
    response = test_client.post(f"/confirm/{file_id}")
    
    # Debug response
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.json()}")
    
    # Verify the API response
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["status"] == "ok"
    assert "task_id" in response_data
    assert "message" in response_data
    
    # Verify the metadata was updated with video generation task
    file_metadata = await mock_storage.get_metadata(file_id)
    assert "video_generation" in file_metadata
    
    # Verify the file status endpoint shows correct state
    status_response = test_client.get(f"/status/{file_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == GENERATING_VIDEO
    
    # Cleanup
    beats_file.unlink(missing_ok=True)

@pytest.mark.asyncio
async def test_confirm_analysis_not_ready(
    test_client: TestClient,
    mock_storage: MockMetadataStorage
):
    """Test confirmation when beat analysis is not complete."""
    # Setup with pending beat detection
    file_id = "test_file_not_ready"
    beat_task = create_mock_task()
    
    # Get standardized file path using the storage API
    audio_file_path = mock_storage.get_audio_file_path(file_id, ".mp3")
    
    metadata = {
        "original_filename": "test.mp3",
        "audio_file_path": str(audio_file_path),
        "file_extension": ".mp3",
        "beat_detection": beat_task.id
    }
    mock_storage.update_metadata(file_id, metadata)
    
    # Set the task state to STARTED (not yet complete)
    beat_task.set_state("STARTED")
    
    # Confirm analysis (should fail since analysis is not complete)
    response = test_client.post(f"/confirm/{file_id}")
    
    assert response.status_code == 400
    assert "not ready for confirmation" in response.json()["detail"]

def test_download_video_success(
    test_client: TestClient,
    mock_storage: MockMetadataStorage,
    test_dir: pathlib.Path
):
    """Test successfully downloading a generated video."""
    file_id = "test-download"
    
    # Get standardized file paths using the storage API
    audio_file_path = mock_storage.get_audio_file_path(file_id, ".mp3")
    video_file = mock_storage.get_video_file_path(file_id)
    
    # Create metadata
    metadata = {
        "original_filename": "test.mp3",
        "audio_file_path": str(audio_file_path),
        "file_extension": ".mp3",
        "upload_time": datetime.now().isoformat()
    }
    mock_storage.update_metadata(file_id, metadata)
    
    # Create beat detection task
    beat_task = create_mock_task()
    beat_task.set_state("SUCCESS")
    metadata["beat_detection"] = beat_task.id
    
    # Create the directory and dummy video file
    job_dir = mock_storage.ensure_job_directory(file_id)
    video_file.parent.mkdir(exist_ok=True, parents=True)
    video_file.touch()
    
    # Create video task
    video_task = create_mock_task()
    video_task.set_state("SUCCESS", {
        "video_file": str(video_file)
    })
    metadata["video_generation"] = video_task.id
    metadata["video_file"] = str(video_file)
    
    mock_storage.update_metadata(file_id, metadata)
    
    try:
        # Attempt to download the video
        response = test_client.get(f"/download/{file_id}")
        assert response.status_code == 200
    finally:
        # Cleanup test files
        video_file.unlink(missing_ok=True)
        job_dir = mock_storage.get_job_directory(file_id)
        job_dir.rmdir()

def test_download_video_not_ready(
    test_client: TestClient,
    mock_storage: MockMetadataStorage
):
    """Test downloading a video that's not ready yet."""
    file_id = "test-download-not-ready"
    
    # Get standardized file paths using the storage API
    audio_file_path = mock_storage.get_audio_file_path(file_id, ".mp3")
    
    # Create metadata
    metadata = {
        "original_filename": "test.mp3",
        "audio_file_path": str(audio_file_path),
        "file_extension": ".mp3",
        "upload_time": datetime.now().isoformat()
    }
    
    # Create beat detection task that's still processing
    beat_task = create_mock_task()
    beat_task.set_state("STARTED")
    metadata["beat_detection"] = beat_task.id
    
    mock_storage.update_metadata(file_id, metadata)
    
    # Attempt to download the video (should fail)
    response = test_client.get(f"/download/{file_id}")
    assert response.status_code == 404
    assert "Video file not found" in response.json()["detail"]

def test_download_video_not_found(
    test_client: TestClient,
    mock_storage: MockMetadataStorage
):
    """Test downloading a video for a file that doesn't exist."""
    file_id = "nonexistent-file"
    
    # For a non-existent file, get_metadata should return None
    # No patching needed as this is the default behavior
    
    # Attempt to download the video (should fail)
    response = test_client.get(f"/download/{file_id}")
    assert response.status_code == 404
    assert file_id in response.json()["detail"]

def test_download_video_analyzed_no_video(
    test_client: TestClient,
    mock_storage: MockMetadataStorage
):
    """Test downloading a video when beat detection is complete but video generation hasn't started."""
    file_id = "test-download-analyzed"
    
    # Get standardized file paths using the storage API
    audio_file_path = mock_storage.get_audio_file_path(file_id, ".mp3")
    beats_file = mock_storage.get_beats_file_path(file_id)
    
    # Create metadata
    metadata = {
        "original_filename": "test.mp3",
        "audio_file_path": str(audio_file_path),
        "file_extension": ".mp3",
        "upload_time": datetime.now().isoformat()
    }
    
    # Create beat detection task that's completed
    beat_task = create_mock_task()
    beat_task.set_state("SUCCESS", {
        "beats_file": str(beats_file)
    })
    metadata["beat_detection"] = beat_task.id
    metadata["beats_file"] = str(beats_file)
    
    # Create the beats file so it exists
    beats_file.parent.mkdir(exist_ok=True, parents=True)
    beats_file.touch()
    
    mock_storage.update_metadata(file_id, metadata)
    
    # Attempt to download the video (should fail)
    response = test_client.get(f"/download/{file_id}")
    assert response.status_code == 404
    assert "Video file not found" in response.json()["detail"]

def test_download_video_failed_generation(
    test_client: TestClient,
    mock_storage: MockMetadataStorage
):
    """Test downloading a video when video generation has failed."""
    file_id = "test-download-failed-video"
    
    # Get standardized file paths using the storage API
    audio_file_path = mock_storage.get_audio_file_path(file_id, ".mp3")
    beats_file = mock_storage.get_beats_file_path(file_id)
    
    # Create metadata
    metadata = {
        "original_filename": "test.mp3",
        "audio_file_path": str(audio_file_path),
        "file_extension": ".mp3",
        "upload_time": datetime.now().isoformat()
    }
    
    # Create beat detection task that's completed
    beat_task = create_mock_task()
    beat_task.set_state("SUCCESS", {
        "beats_file": str(beats_file)
    })
    metadata["beat_detection"] = beat_task.id
    metadata["beats_file"] = str(beats_file)
    
    # Create the beats file so it exists
    beats_file.parent.mkdir(exist_ok=True, parents=True)
    beats_file.touch()
    
    # Create video task that failed
    video_task = create_mock_task()
    video_task.set_state("FAILURE", "Video generation failed with error")
    metadata["video_generation"] = video_task.id
    
    mock_storage.update_metadata(file_id, metadata)
    
    # Attempt to download the video (should fail)
    response = test_client.get(f"/download/{file_id}")
    assert response.status_code == 404
    assert "Video file not found" in response.json()["detail"]

@pytest.mark.asyncio
@pytest.mark.parametrize("beat_state,video_state,expected_status", [
    (None, None, ERROR),  # No tasks
    ("STARTED", None, ANALYZING),  # Beat detection in progress
    ("SUCCESS", None, ANALYZED),  # Beat detection completed
    ("SUCCESS", "STARTED", GENERATING_VIDEO),  # Video generation in progress
    ("SUCCESS", "SUCCESS", COMPLETED),  # All completed
    ("FAILURE", None, ANALYZING_FAILURE),  # Beat detection failed
    ("SUCCESS", "FAILURE", VIDEO_ERROR),  # Video generation failed
])
async def test_get_file_status_overall_status(
    test_client: TestClient,
    mock_storage: MockMetadataStorage,
    beat_state: Optional[str],
    video_state: Optional[str],
    expected_status: str
):
    """Test that get_file_status returns the correct overall status."""
    # Setup
    file_id = f"test_file_status_{beat_state}_{video_state}"
    metadata = {
        "original_filename": "test.mp3",
        "audio_file_path": f"/path/to/{file_id}.mp3"
    }
    
    # Add tasks to metadata
    if beat_state:
        beat_task = create_mock_task()
        beat_task.set_state(beat_state)
        metadata["beat_detection"] = beat_task.id
        
        # If beat detection is successful, create beats file
        if beat_state == "SUCCESS":
            beats_file = mock_storage.get_beats_file_path(file_id)
            beats_file.parent.mkdir(exist_ok=True, parents=True)
            beats_file.touch()
            
            # Also create stats file with dummy content
            stats_file = mock_storage.get_beat_stats_file_path(file_id)
            with open(stats_file, 'w') as f:
                json.dump({"bpm": 120.0, "beats": 100}, f)
                
            metadata["beats_file"] = str(beats_file)
            metadata["stats_file"] = str(stats_file)
    
    if video_state:
        video_task = create_mock_task()
        video_task.set_state(video_state)
        metadata["video_generation"] = video_task.id
        
        # If video generation is successful, create video file
        if video_state == "SUCCESS":
            video_file = mock_storage.get_video_file_path(file_id)
            video_file.parent.mkdir(exist_ok=True, parents=True)
            video_file.touch()
            metadata["video_file"] = str(video_file)
    
    mock_storage.update_metadata(file_id, metadata)
    
    # Get status via API endpoint
    response = test_client.get(f"/status/{file_id}")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == expected_status
    
    # Verify task states are included
    if beat_state:
        assert "beat_detection_task" in data
        assert data["beat_detection_task"]["state"] == beat_state
    
    if video_state:
        assert "video_generation_task" in data
        assert data["video_generation_task"]["state"] == video_state
    
    # Cleanup created files
    if beat_state == "SUCCESS":
        beats_file.unlink(missing_ok=True)
        stats_file.unlink(missing_ok=True)
    
    if video_state == "SUCCESS":
        video_file.unlink(missing_ok=True)

@pytest.mark.parametrize("beat_state,video_state,expected_status", [
    ("SUCCESS", None, 200),  # Only beat detection completed
    ("PENDING", None, 200),  # Beat detection pending
    ("SUCCESS", "SUCCESS", 200),  # Both tasks completed
    ("SUCCESS", "PENDING", 200),  # Video generation in progress
    (None, None, 404),  # File not found
])
async def test_file_page_status(
    test_client: TestClient,
    mock_storage: MockMetadataStorage,
    beat_state: str,
    video_state: str,
    expected_status: int
):
    """Test the file page endpoint with various task states."""
    file_id = f"test_file_page_{beat_state}_{video_state}"
    
    if beat_state is not None:
        # Create metadata with tasks
        metadata = {
            "original_filename": "test.mp3",
            "audio_file_path": f"/path/to/{file_id}.mp3"
        }
        
        # Add tasks to metadata
        if beat_state:
            beat_task = create_mock_task()
            beat_task.set_state(beat_state)
            metadata["beat_detection"] = beat_task.id
            
            # If beat detection is successful, create beats file
            if beat_state == "SUCCESS":
                beats_file = mock_storage.get_beats_file_path(file_id)
                beats_file.parent.mkdir(exist_ok=True, parents=True)
                beats_file.touch()
                
                # Also create stats file with dummy content
                stats_file = mock_storage.get_beat_stats_file_path(file_id)
                with open(stats_file, 'w') as f:
                    json.dump({"bpm": 120.0, "beats": 100}, f)
                    
                metadata["beats_file"] = str(beats_file)
                metadata["stats_file"] = str(stats_file)
        
        if video_state:
            video_task = create_mock_task()
            video_task.set_state(video_state)
            metadata["video_generation"] = video_task.id
            
            # If video generation is successful, create video file
            if video_state == "SUCCESS":
                video_file = mock_storage.get_video_file_path(file_id)
                video_file.parent.mkdir(exist_ok=True, parents=True)
                video_file.touch()
                metadata["video_file"] = str(video_file)
        
        mock_storage.update_metadata(file_id, metadata)
        
        try:
            # Get file page
            response = test_client.get(f"/file/{file_id}")
            
            # Verify response code
            assert response.status_code == expected_status
            
            # If successful, verify response contains the right data
            if expected_status == 200:
                # Check that the response is HTML and includes the file ID
                assert response.headers["content-type"] == "text/html; charset=utf-8"
                assert file_id in response.text
                
                # Verify status-specific content
                if video_state == "SUCCESS":
                    assert 'data-status="COMPLETED"' in response.text
                elif video_state == "PENDING":
                    assert 'data-status="GENERATING_VIDEO"' in response.text
                elif beat_state == "PENDING":
                    assert 'data-status="ANALYZING"' in response.text
                elif beat_state == "SUCCESS":
                    assert 'data-status="ANALYZED"' in response.text
        finally:
            # Cleanup created files
            if beat_state == "SUCCESS":
                beats_file.unlink(missing_ok=True)
                stats_file.unlink(missing_ok=True)
            
            if video_state == "SUCCESS":
                video_file.unlink(missing_ok=True)
    else:
        # For files that don't exist, just verify the 404 response
        response = test_client.get(f"/file/{file_id}")
        assert response.status_code == expected_status

# Task-related tests
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

def test_mock_task_executor_methods(mock_executor: MockTaskExecutor):
    """Test the helper methods of MockTaskExecutor."""
    # Test task creation
    task = mock_executor._create_task()
    assert task.id in mock_executor.tasks
    assert mock_executor.tasks[task.id] is task
    
    # Test task completion helper
    mock_executor.complete_task(task.id, {"result": "success"})
    assert task.state == "SUCCESS"
    assert task.result == {"result": "success"}
    
    # Test task failure helper
    task2 = mock_executor._create_task()
    mock_executor.fail_task(task2.id, "test error")
    assert task2.state == "FAILURE"
    assert task2.result == "test error"
    
    # Test with non-existent task
    mock_executor.complete_task("nonexistent")  # Should not raise error
    mock_executor.fail_task("nonexistent")  # Should not raise error

def test_task_status_structure(mock_executor: MockTaskExecutor):
    """Test that get_task_status returns a properly structured response."""
    # Create a task and test different states
    task = mock_executor._create_task()
    
    # Test PENDING state (initial state)
    status = mock_executor.get_task_status(task.id)
    assert isinstance(status, dict)
    assert "state" in status
    assert status["state"] == "STARTED"
    assert "result" in status
    assert status["result"] is None
    
    # Test STARTED state
    task.set_state("STARTED")
    status = mock_executor.get_task_status(task.id)
    assert status["state"] == "STARTED"
    assert status["result"] is None
    
    # Test SUCCESS state with result
    result_data = {"beats_file": "beats.txt"}
    task.set_state("SUCCESS", result_data)
    status = mock_executor.get_task_status(task.id)
    assert status["state"] == "SUCCESS"
    assert status["result"] == result_data
    
    # Test FAILURE state with error message
    error_msg = "Task failed with error"
    task.set_state("FAILURE", error_msg)
    status = mock_executor.get_task_status(task.id)
    assert status["state"] == "FAILURE"
    
    # Check if error is in the "error" field or in the "result" field
    assert (("error" in status and status["error"] == error_msg) or 
            ("result" in status and status["result"] == error_msg))
    
    # Test non-existent task
    status = mock_executor.get_task_status("nonexistent")
    assert isinstance(status, dict)
    assert "state" in status
    assert status["state"] == ERROR
    assert "result" in status
    assert status["result"] is None

@pytest.mark.asyncio
async def test_file_metadata_with_tasks(mock_storage: MockMetadataStorage, mock_executor: MockTaskExecutor):
    """Test getting file metadata with different task states using a temporary directory."""
    # Create a temporary directory for file operations
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create file metadata
        file_id = "test-file-metadata"
        
        # Get standardized file paths
        audio_file_path = pathlib.Path(temp_dir) / "audio.mp3"
        beats_file = pathlib.Path(temp_dir) / "beats.txt"
        stats_file = pathlib.Path(temp_dir) / "stats.json"
        video_file = pathlib.Path(temp_dir) / "visualization.mp4"
        
        # Create the audio file to simulate upload
        audio_file_path.touch()
        
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
        
        # Get file metadata
        meta_data = await mock_storage.get_file_metadata(file_id)
        
        # Verify metadata structure
        assert meta_data["filename"] == "test.mp3"
        assert meta_data["beat_detection"] == beat_task.id
        assert "beat_detection_task" not in meta_data  # Task status is not included in metadata
        assert "beat_stats" not in meta_data  # No beat stats yet
        
        # Complete beat detection task - this should make the beats file "exist" in the mock
        beats_file.touch()  # Simulate the creation of the beats file
        stats_file.touch()  # Simulate the creation of the stats file
        # Write some dummy content to the stats file
        with open(stats_file, 'w') as f:
            json.dump({"total_beats": 120, "bpm": 100}, f)
            
        mock_executor.complete_task(beat_task.id, {
            "file_id": file_id,
            "beats_file": str(beats_file),
            "stats_file": str(stats_file)
        })
        
        # Tell the mock storage where to find the beats and stats files
        mock_storage.update_metadata(file_id, {
            "beats_file": str(beats_file),
            "stats_file": str(stats_file)
        })
        
        # Get updated metadata
        meta_data = await mock_storage.get_file_metadata(file_id)
        assert "beat_stats" in meta_data  # Beat stats should now be included
        
        # Test with video generation task
        video_task = mock_executor.execute_video_generation(file_id)
        
        # Update metadata with task ID
        mock_storage.update_metadata(file_id, {"video_generation": video_task.id})
        
        # Get updated metadata
        meta_data = await mock_storage.get_file_metadata(file_id)
        assert "video_generation" in meta_data
        assert meta_data["video_generation"] == video_task.id
        
        # Complete video generation task
        video_file.touch()  # Simulate the creation of the video file
        mock_executor.complete_task(video_task.id, {
            "file_id": file_id,
            "video_file": str(video_file)
        })
        
        # Update metadata with video file path
        mock_storage.update_metadata(file_id, {
            "video_file": str(video_file)
        })

def test_get_task_status_not_found(mock_executor: MockTaskExecutor):
    """Test getting status of non-existent task."""
    status = mock_executor.get_task_status("nonexistent")
    assert status["state"] == ERROR

class MockMetadataStorage(MetadataStorage):
    """Mock implementation of MetadataStorage for testing."""
    
    def __init__(self):
        self.storage = {}
        # Use a temporary directory for all file operations
        self.temp_dir = tempfile.TemporaryDirectory()

    async def get_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific file."""
        return self.storage.get(file_id)
    
    def update_metadata(self, file_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a specific file."""
        if file_id not in self.storage:
            self.storage[file_id] = {}
        
        # Deep update the metadata
        self._deep_update(self.storage[file_id], metadata)
    
    def _deep_update(self, target: Dict, source: Dict) -> None:
        """Helper function for deep updating dictionaries."""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    async def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all files."""
        return self.storage.copy()
    
    def delete_metadata(self, file_id: str) -> bool:
        """Delete metadata for a specific file."""
        if file_id in self.storage:
            del self.storage[file_id]
            return True
        return False

    async def get_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file metadata, simulating file existence checks."""
        metadata = self.storage.get(file_id)
        
        if metadata:
            # Create response with basic file information
            status_data = {
                "file_id": file_id,
                "filename": metadata.get("original_filename"),
                "audio_file_path": metadata.get("audio_file_path")
            }
            
            # Add task IDs to the status data
            beat_task_id = metadata.get("beat_detection")
            video_task_id = metadata.get("video_generation")
            
            if beat_task_id:
                status_data["beat_detection"] = beat_task_id
                
            if video_task_id:
                status_data["video_generation"] = video_task_id
            
            # Include beats_file if it exists in the original metadata
            if "beats_file" in metadata:
                status_data["beats_file"] = metadata["beats_file"]
                
            # Include video_file if it exists in the original metadata
            if "video_file" in metadata:
                status_data["video_file"] = metadata["video_file"]
            
            # Add beat stats if beats file exists
            beats_file = metadata.get("beats_file")
            stats_file = metadata.get("stats_file")
            if beats_file and stats_file and pathlib.Path(stats_file).exists():
                try:
                    with open(stats_file, 'r') as f:
                        beat_stats = json.load(f)
                        status_data["beat_stats"] = beat_stats
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    pass  # Silently ignore errors in tests
            
            return status_data
            
        return metadata
    
    def get_job_directory(self, file_id: str) -> pathlib.Path:
        """Get the standardized job directory for a file ID."""
        job_dir = pathlib.Path(self.temp_dir.name) / file_id
        job_dir.mkdir(exist_ok=True, parents=True)
        return job_dir
    
    def get_audio_file_path(self, file_id: str, file_extension: str = None) -> pathlib.Path:
        """Get the standardized path for the audio file."""
        if file_extension:
            return pathlib.Path(self.temp_dir.name) / f"{file_id}_audio{file_extension}"
        metadata = self.storage.get(file_id, {})
        ext = metadata.get("file_extension", "")
        return pathlib.Path(self.temp_dir.name) / f"{file_id}_audio{ext}"
    
    def get_beats_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the beats file."""
        return pathlib.Path(self.temp_dir.name) / f"{file_id}_beats.txt"
    
    def get_beat_stats_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the beat statistics file."""
        return pathlib.Path(self.temp_dir.name) / f"{file_id}_beat_stats.json"
    
    def get_video_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the visualization video."""
        return pathlib.Path(self.temp_dir.name) / f"{file_id}_visualization.mp4"
    
    def ensure_job_directory(self, file_id: str) -> pathlib.Path:
        """Ensure the job directory exists and return its path."""
        job_dir = self.get_job_directory(file_id)
        job_dir.mkdir(exist_ok=True, parents=True)
        return job_dir
        
    def save_audio_file(self, file_id: str, file_extension: str, file_obj, filename: str = None) -> pathlib.Path:
        """Save an uploaded audio file to the storage and return its path."""
        # Ensure job directory exists
        self.ensure_job_directory(file_id)
        
        # Get standardized path for the audio file
        audio_file_path = self.get_audio_file_path(file_id, file_extension)
        
        # Save the file (or just create it for tests)
        if hasattr(file_obj, 'read'):
            with open(audio_file_path, "wb") as f:
                import shutil
                shutil.copyfileobj(file_obj, f)
        else:
            # For tests, just create an empty file
            audio_file_path.touch()
        
        # Create metadata
        metadata = {
            "original_filename": filename or f"test{file_extension}",
            "audio_file_path": str(audio_file_path),
            "file_extension": file_extension,
            "upload_time": datetime.now().isoformat()
        }
        
        self.update_metadata(file_id, metadata)
        
        return audio_file_path 