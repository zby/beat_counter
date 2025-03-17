"""Tests for the FastAPI application."""

import os
import pathlib
import tempfile
from typing import Dict, Generator, List, Optional, Tuple
import pytest
from fastapi.testclient import TestClient
from web_app.app import create_app, RedisMetadataStorage, CeleryTaskExecutor
from beat_detection.utils.constants import AUDIO_EXTENSIONS
from web_app.storage import ANALYZING, ANALYZED, ANALYZING_FAILURE, GENERATING_VIDEO, COMPLETED, VIDEO_ERROR, ERROR, MetadataStorage, TaskExecutor
from web_app.test_storage import MockMetadataStorage, MockTask, MockTaskExecutor
import uuid

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
def mock_executor() -> MockTaskExecutor:
    """Create a fresh mock task executor instance for each test."""
    return MockTaskExecutor()

@pytest.fixture
def test_client(mock_storage: MockMetadataStorage, mock_executor: MockTaskExecutor) -> TestClient:
    """Create a test client with mock dependencies."""
    app = create_app(metadata_storage=mock_storage, task_executor=mock_executor)
    return TestClient(app)

@pytest.fixture
def test_file(test_dir: pathlib.Path, request: pytest.FixtureRequest) -> pathlib.Path:
    """Create a test file with the specified name."""
    filename = getattr(request, "param", "test.mp3")
    file_path = test_dir / filename
    file_path.touch()
    return file_path

@pytest.fixture
def sample_file_with_task(
    mock_storage: MockMetadataStorage,
    mock_executor: MockTaskExecutor
) -> Tuple[str, str, MockTask]:
    """Create a sample file entry with an associated task."""
    file_id = "test_file"
    task = mock_executor._create_task()
    mock_storage.update_metadata(file_id, {
        "filename": "test.mp3",
        "file_path": "/path/to/test.mp3",
        "beat_detection": task.id
    })
    return file_id, task.id, task

@pytest.mark.parametrize("filename,content_type,is_valid,expected_status", TEST_FILES)
def test_upload_audio(
    test_client: TestClient,
    mock_storage: MockMetadataStorage,
    mock_executor: MockTaskExecutor,
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
        assert metadata["filename"] == filename
        
        # Verify task creation
        assert "beat_detection" in metadata
        task_id = metadata["beat_detection"]
        task = mock_executor.tasks.get(task_id)
        assert task is not None
        assert task.state == "STARTED"
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
    assert data["filename"] == "test.mp3"
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
    assert data["beat_detection_task"]["result"] == "Processing error"

def test_file_not_found(test_client: TestClient):
    """Test handling of non-existent files."""
    response = test_client.get("/status/nonexistent")
    assert response.status_code == 404
    # Check if the file ID is in the error message
    assert "nonexistent" in response.json()["detail"]

@pytest.mark.asyncio
async def test_confirm_analysis_success(
    test_client: TestClient,
    mock_storage: MockMetadataStorage,
    mock_executor: MockTaskExecutor
):
    """Test successful confirmation of beat analysis."""
    # Setup
    file_id = "test_file"
    beat_task = mock_executor._create_task()
    mock_storage.update_metadata(file_id, {
        "filename": "test.mp3",
        "file_path": "/path/to/test.mp3",
        "beat_detection": beat_task.id
    })
    
    # Set beat detection as successful
    beat_task.set_state("SUCCESS", {
        "beats_file": "/path/to/beats.json"
    })
    
    # Confirm analysis
    response = test_client.post(f"/confirm/{file_id}", follow_redirects=False)
    
    assert response.status_code == 303
    
    # Verify video generation task
    metadata = mock_storage.storage.get(file_id)
    assert "video_generation" in metadata
    assert metadata["status"] == "GENERATING_VIDEO"
    
    video_task_id = metadata["video_generation"]
    video_task = mock_executor.tasks.get(video_task_id)
    assert video_task is not None
    assert video_task.state == "STARTED"

@pytest.mark.asyncio
async def test_confirm_analysis_not_ready(
    test_client: TestClient,
    mock_storage: MockMetadataStorage,
    mock_executor: MockTaskExecutor
):
    """Test confirmation when beat analysis is not complete."""
    # Setup with pending beat detection
    file_id = "test_file"
    beat_task = mock_executor._create_task()
    mock_storage.update_metadata(file_id, {
        "filename": "test.mp3",
        "file_path": "/path/to/test.mp3",
        "beat_detection": beat_task.id
    })
    
    # Confirm analysis
    response = test_client.post(f"/confirm/{file_id}")
    
    assert response.status_code == 400
    assert "Beat detection not completed" in response.json()["detail"]

@pytest.mark.parametrize("status,expected_status", [
    (COMPLETED, 200),  # File exists and video is ready
    (ANALYZING, 400),  # Video not ready
    (None, 404),       # File doesn't exist
])
def test_download_video(
    test_client: TestClient,
    mock_storage: MockMetadataStorage,
    mock_executor: MockTaskExecutor,
    test_dir: pathlib.Path,
    status: str,
    expected_status: int
):
    """Test downloading a video file."""
    file_id = "test-download-video"
    
    # Set up mock file with the specified status
    if status:
        # Create test files in the test directory
        audio_file = test_dir / "test.mp3"
        audio_file.touch()
        
        metadata = {
            "filename": "test.mp3",
            "file_path": str(audio_file)
        }
        mock_storage.update_metadata(file_id, metadata)
        
        if status == COMPLETED:
            # Create a test video file
            video_file = test_dir / "test.mp4"
            video_file.touch()
            
            # Simulate completed video
            beat_task = mock_executor.execute_beat_detection(file_id, str(audio_file))
            beat_task.set_state("SUCCESS")
            metadata["beat_detection"] = beat_task.id
            
            video_task = mock_executor.execute_video_generation(file_id, str(audio_file), str(test_dir / "beats.txt"))
            video_task.set_state("SUCCESS")
            metadata["video_generation"] = video_task.id
            metadata["video_file"] = str(video_file)  # Add video file path
            mock_storage.update_metadata(file_id, metadata)
        elif status == ANALYZING:
            beat_task = mock_executor.execute_beat_detection(file_id, str(audio_file))
            beat_task.set_state("STARTED")
            metadata["beat_detection"] = beat_task.id
            mock_storage.update_metadata(file_id, metadata)
    
    # Try to download the video
    response = test_client.get(f"/download/{file_id}")
    assert response.status_code == expected_status

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
    mock_executor: MockTaskExecutor,
    beat_state: Optional[str],
    video_state: Optional[str],
    expected_status: str
):
    """Test that get_file_status returns the correct overall status."""
    # Setup
    file_id = "test_file"
    metadata = {
        "filename": "test.mp3",
        "file_path": "/path/to/test.mp3"
    }
    
    if beat_state:
        beat_task = mock_executor._create_task()
        beat_task.set_state(beat_state)
        metadata["beat_detection"] = beat_task.id
    
    if video_state:
        video_task = mock_executor._create_task()
        video_task.set_state(video_state)
        metadata["video_generation"] = video_task.id
    
    mock_storage.update_metadata(file_id, metadata)
    
    # Get status
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
    mock_executor: MockTaskExecutor,
    beat_state: str,
    video_state: str,
    expected_status: int
):
    """Test the file page endpoint with various task states."""
    file_id = "test_file_id"
    
    if beat_state is not None:
        # Create metadata with beat detection task
        beat_task = mock_executor._create_task()
        metadata = {
            "filename": "test.mp3",
            "file_path": "/path/to/test.mp3",
            "beat_detection": beat_task.id
        }
        
        # Set up beat detection task state
        beat_task.set_state(beat_state, {"beats_file": "beats.json"} if beat_state == "SUCCESS" else None)
        
        if video_state is not None:
            # Add video generation task
            video_task = mock_executor._create_task()
            metadata["video_generation"] = video_task.id
            video_task.set_state(video_state, {"video_file": "video.mp4"} if video_state == "SUCCESS" else None)
        
        # Store metadata
        mock_storage.update_metadata(file_id, metadata)
    
    # Make request to file page
    response = test_client.get(f"/file/{file_id}")
    
    if expected_status == 404:
        # For 404 cases, we should get a JSON error response
        assert response.status_code == 404
        # Check if the file ID is in the error message
        assert "test_file_id" in response.json()["detail"]
    else:
        assert response.status_code == expected_status
        
        # Get response content
        response_html = response.text
        
        # Verify basic file information is present in the HTML
        assert file_id in response_html
        assert "test.mp3" in response_html
        
        # Verify task status information is present
        if beat_state is not None:
            assert beat_state in response_html
            if beat_state == "SUCCESS":
                assert "beats.json" in response_html
        
        if video_state is not None:
            assert video_state in response_html
            if video_state == "SUCCESS":
                assert "video.mp4" in response_html 