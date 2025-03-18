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
import uuid
from datetime import datetime
import json

# Simplified mock infrastructure for tasks
class MockTask:
    """Mock implementation of a task."""
    
    def __init__(self, task_type: Optional[str] = None, file_id: Optional[str] = None, state: str = "STARTED", result: Any = None):
        self.id = str(uuid.uuid4())
        self.state = state
        self.result = result
        self.task_type = task_type
        self.file_id = file_id
    
    def set_state(self, state: str, result: Any = None) -> None:
        """Set the task state and result."""
        self.state = state
        self.result = result

# Simplified task service provider for testing
class MockTaskServiceProvider(TaskServiceProvider):
    """Mock implementation of TaskServiceProvider for testing."""
    
    def __init__(self):
        """Initialize with internal task tracking."""
        # Store tasks by ID
        self.tasks = {}
        # Initialize with mock implementations
        super().__init__(
            get_task_status_fn=self.get_task_status,
            detect_beats_fn=self.mock_detect_beats,
            generate_video_fn=self.mock_generate_video
        )
    
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
    
    def mock_detect_beats(self, file_id: str) -> MockTask:
        """Mock implementation for beat detection."""
        task = MockTask(task_type="beat_detection", file_id=file_id)
        self.tasks[task.id] = task
        return task
    
    def mock_generate_video(self, file_id: str) -> MockTask:
        """Mock implementation for video generation."""
        task = MockTask(task_type="video_generation", file_id=file_id)
        self.tasks[task.id] = task
        return task
    
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

# Mock metadata storage for testing
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
def sample_file_with_task(mock_storage: MockMetadataStorage, mock_task_service: MockTaskServiceProvider) -> Tuple[str, str, MockTask]:
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
    task = mock_task_service.mock_detect_beats(file_id)
    metadata["beat_detection"] = task.id
    
    # Store metadata
    mock_storage.update_metadata(file_id, metadata)
    
    return file_id, task.id, task

@pytest.mark.parametrize("filename,content_type,is_valid,expected_status", TEST_FILES)
def test_upload_audio(
    test_client: TestClient,
    mock_storage: MockMetadataStorage,
    mock_task_service: MockTaskServiceProvider,
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
        
        # Verify that beat detection task was started
        assert "beat_detection" in metadata
        task_id = metadata["beat_detection"]
        assert task_id in mock_task_service.tasks
    else:
        assert "Unsupported file format" in response.json()["detail"]

@pytest.mark.asyncio
async def test_get_file_status(
    test_client: TestClient,
    sample_file_with_task: Tuple[str, str, MockTask]
):
    """Test getting file status with different task states."""
    file_id, task_id, task = sample_file_with_task
    
    # Test success state
    task.set_state("SUCCESS", {"beats": [1.0, 2.0, 3.0]})
    response = test_client.get(f"/status/{file_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["beat_detection_task"]["state"] == "SUCCESS"
    assert data["beat_detection_task"]["result"]["beats"] == [1.0, 2.0, 3.0]
    
    # Test failure state
    task.set_state("FAILURE", "Processing error")
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
async def test_confirm_analysis(
    test_client: TestClient,
    mock_storage: MockMetadataStorage,
    mock_task_service: MockTaskServiceProvider
):
    """Test confirming analysis and generating a visualization video."""
    # 1. Setup file with completed beat detection
    file_id = "confirm-test-file"
    beat_task = MockTask(state="SUCCESS", result={"beats_file": "beats.txt"})
    mock_task_service.tasks[beat_task.id] = beat_task
    
    # Create the actual beats file
    beats_file = mock_storage.get_beats_file_path(file_id)
    beats_file.parent.mkdir(exist_ok=True, parents=True)
    beats_file.touch()
    
    # Setup metadata
    mock_storage.update_metadata(file_id, {
        "original_filename": "test.mp3",
        "beat_detection": beat_task.id,
        "beats_file": str(beats_file)
    })
    
    # 2. Successfully start video generation
    response = test_client.post(f"/confirm/{file_id}")
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["status"] == "ok"
    assert "task_id" in response_data
    
    # Verify metadata was updated with video generation task
    file_metadata = await mock_storage.get_metadata(file_id)
    assert "video_generation" in file_metadata
    
    # Verify status shows correct state
    status_response = test_client.get(f"/status/{file_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == GENERATING_VIDEO
    
    # 3. Test not-ready state
    not_ready_file_id = "test_file_not_ready"
    not_ready_task = MockTask(state="STARTED")
    mock_task_service.tasks[not_ready_task.id] = not_ready_task
    
    mock_storage.update_metadata(not_ready_file_id, {
        "original_filename": "test.mp3",
        "audio_file_path": str(mock_storage.get_audio_file_path(not_ready_file_id, ".mp3")),
        "file_extension": ".mp3",
        "beat_detection": not_ready_task.id
    })
    
    not_ready_response = test_client.post(f"/confirm/{not_ready_file_id}")
    assert not_ready_response.status_code == 400
    assert "not ready for confirmation" in not_ready_response.json()["detail"]
    
    # Cleanup
    beats_file.unlink(missing_ok=True)

@pytest.mark.asyncio
async def test_download_video(
    test_client: TestClient,
    mock_storage: MockMetadataStorage,
    mock_task_service: MockTaskServiceProvider,
    test_dir: pathlib.Path
):
    """Test downloading videos with different states."""
    # Setup for successful download
    success_file_id = "test-download-success"
    audio_file_path = mock_storage.get_audio_file_path(success_file_id, ".mp3")
    video_file = mock_storage.get_video_file_path(success_file_id)
    
    # Create basic metadata
    success_metadata = {
        "original_filename": "test.mp3",
        "audio_file_path": str(audio_file_path),
        "file_extension": ".mp3",
        "upload_time": datetime.now().isoformat()
    }
    
    # Create successful tasks
    beat_task = MockTask(state="SUCCESS")
    mock_task_service.tasks[beat_task.id] = beat_task
    success_metadata["beat_detection"] = beat_task.id
    
    # Create directory and video file
    job_dir = mock_storage.ensure_job_directory(success_file_id)
    video_file.parent.mkdir(exist_ok=True, parents=True)
    video_file.touch()
    
    # Add video task
    video_task = MockTask(state="SUCCESS", result={"video_file": str(video_file)})
    mock_task_service.tasks[video_task.id] = video_task
    success_metadata["video_generation"] = video_task.id
    success_metadata["video_file"] = str(video_file)
    
    mock_storage.update_metadata(success_file_id, success_metadata)
    
    # Test successful download
    success_response = test_client.get(f"/download/{success_file_id}")
    assert success_response.status_code == 200
    
    # Test missing video file
    no_video_file_id = "test-download-no-video"
    mock_storage.update_metadata(no_video_file_id, {
        "original_filename": "test.mp3",
        "audio_file_path": str(mock_storage.get_audio_file_path(no_video_file_id, ".mp3")),
        "file_extension": ".mp3"
    })
    
    no_video_response = test_client.get(f"/download/{no_video_file_id}")
    assert no_video_response.status_code == 404
    assert "Video file not found" in no_video_response.json()["detail"]
    
    # Test non-existent file
    nonexistent_response = test_client.get(f"/download/nonexistent-file")
    assert nonexistent_response.status_code == 404
    
    # Cleanup
    video_file.unlink(missing_ok=True)
    job_dir.rmdir()

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
    mock_task_service: MockTaskServiceProvider,
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
        beat_task = MockTask(state=beat_state)
        mock_task_service.tasks[beat_task.id] = beat_task
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
        video_task = MockTask(state=video_state)
        mock_task_service.tasks[video_task.id] = video_task
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
    ("STARTED", None, 200),  # Beat detection pending
    ("SUCCESS", "SUCCESS", 200),  # Both tasks completed
    ("SUCCESS", "STARTED", 200),  # Video generation in progress
    (None, None, 404),  # File not found
])
async def test_file_page(
    test_client: TestClient,
    mock_storage: MockMetadataStorage,
    mock_task_service: MockTaskServiceProvider,
    beat_state: Optional[str],
    video_state: Optional[str],
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
            beat_task = MockTask(state=beat_state)
            mock_task_service.tasks[beat_task.id] = beat_task
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
            video_task = MockTask(state=video_state)
            mock_task_service.tasks[video_task.id] = video_task
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
                
                # Verify status-specific content (simplified checking)
                assert 'data-status=' in response.text
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