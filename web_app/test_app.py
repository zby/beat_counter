"""Tests for the FastAPI application."""

import os
import pathlib
from typing import Generator, Tuple
import pytest
from fastapi.testclient import TestClient
from web_app.app import create_app
from web_app.test_storage import MockMetadataStorage, MockTaskExecutor, MockTask

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
    assert "File not found" in response.json()["detail"]

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
    assert "File has not been analyzed yet" in response.json()["detail"]

@pytest.mark.parametrize("status,expected_status", [
    ("COMPLETED", 404),  # File exists but video doesn't
    ("PROCESSING", 400), # Video not ready
    (None, 404),         # File doesn't exist
])
def test_download_video(
    test_client: TestClient,
    mock_storage: MockMetadataStorage,
    mock_executor: MockTaskExecutor,
    status: str,
    expected_status: int
):
    """Test video download with various states."""
    file_id = "test_file"
    if status is not None:
        # Create a mock task
        task = mock_executor._create_task()
        if status == "PROCESSING":
            task.set_state("STARTED")
        else:
            # For COMPLETED status, set a success state but without a video file
            task.set_state("SUCCESS", {"result": "success"})
        
        mock_storage.update_metadata(file_id, {
            "filename": "test.mp3",
            "status": status,
            "video_generation": task.id
        })
    
    response = test_client.get(f"/download/{file_id}")
    assert response.status_code == expected_status 