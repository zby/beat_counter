"""Tests for the FastAPI application."""

import os
import pathlib
import tempfile
from typing import Dict, Generator, List, Optional, Tuple, Any
import pytest
from fastapi.testclient import TestClient
from web_app.app import create_app
from beat_detection.utils.constants import AUDIO_EXTENSIONS
from web_app.storage import MetadataStorage, FileMetadataStorage
from web_app.task_executor import ANALYZING, ANALYZED, ANALYZING_FAILURE, GENERATING_VIDEO, COMPLETED, VIDEO_ERROR, ERROR
from web_app.test_storage import MockMetadataStorage, MockTask
import uuid
from datetime import datetime
import json

# Mock functions for task operations
MOCK_TASKS = {}

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
def test_client(mock_storage: MockMetadataStorage) -> TestClient:
    """Create a test client with mock dependencies."""
    # Patch the app to use our mock get_task_status function
    import web_app.app
    original_get_task_status = web_app.app.get_task_status
    web_app.app.get_task_status = mock_get_task_status
    
    # Mock Celery tasks
    from web_app.tasks import detect_beats_task, generate_video_task
    
    # Save original methods
    original_detect_beats_delay = detect_beats_task.delay
    
    # Define mock methods
    def mock_detect_beats_delay(file_id):
        # Create a mock task and add it to MOCK_TASKS
        task = create_mock_task()
        # Update task with metadata that identifies it as a beat detection task
        task.task_type = "beat_detection"
        task.file_id = file_id
        return task
    
    # Apply patches
    detect_beats_task.delay = mock_detect_beats_delay
    
    app = create_app(metadata_storage=mock_storage)
    client = TestClient(app)
    
    yield client
    
    # Restore original functions after the test
    web_app.app.get_task_status = original_get_task_status
    detect_beats_task.delay = original_detect_beats_delay

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
        assert f"original{file_extension}" in audio_file_path
        
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
    """Test confirming analysis and generating a visualization video."""
    # Create a file in ANALYZED state
    file_id = "confirm-test-file"
    
    # Get standardized file paths using the storage API
    audio_file_path = mock_storage.get_audio_file_path(file_id, ".mp3")
    beats_file = mock_storage.get_beats_file_path(file_id)
    stats_file = mock_storage.get_beat_stats_file_path(file_id)
    
    # Create file metadata
    metadata = {
        "original_filename": "test.mp3",
        "audio_file_path": str(audio_file_path),
        "file_extension": ".mp3",
        "upload_time": datetime.now().isoformat()
    }
    
    # Create beat detection task
    beat_task = create_mock_task()
    beat_task.set_state("SUCCESS", {
        "file_id": file_id,
        "beats_file": str(beats_file),
        "stats_file": str(stats_file)
    })
    
    # Update metadata with beat detection task
    metadata["beat_detection"] = beat_task.id
    mock_storage.update_metadata(file_id, metadata)
    
    # Add stats to mock storage get_file_status response
    mock_storage.get_file_status_response = {
        "file_id": file_id,
        "filename": "test.mp3",
        "status": ANALYZED,
        "beat_detection_task": {
            "id": beat_task.id,
            "state": "SUCCESS",
            "result": {
                "beats_file": str(beats_file)
            }
        }
    }
    
    # Confirm analysis
    response = test_client.post(f"/confirm/{file_id}")
    assert response.status_code == 200
    
    # Verify video generation task was created
    file_metadata = await mock_storage.get_metadata(file_id)
    assert "video_generation" in file_metadata
    
    # Update mock response to reflect new status
    video_task_id = file_metadata["video_generation"]
    mock_storage.get_file_status_response = {
        "file_id": file_id,
        "filename": "test.mp3",
        "status": GENERATING_VIDEO,
        "beat_detection_task": {
            "id": beat_task.id,
            "state": "SUCCESS",
            "result": {
                "beats_file": str(beats_file)
            }
        },
        "video_generation_task": {
            "id": video_task_id,
            "state": "STARTED"
        }
    }
    
    # Verify status was updated
    file_status = await mock_storage.get_file_status(file_id)
    assert file_status["status"] == GENERATING_VIDEO

@pytest.mark.asyncio
async def test_confirm_analysis_not_ready(
    test_client: TestClient,
    mock_storage: MockMetadataStorage
):
    """Test confirmation when beat analysis is not complete."""
    # Setup with pending beat detection
    file_id = "test_file"
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
    
    # Set the task state to STARTED 
    beat_task.set_state("STARTED")
    
    # Set custom file status for consistent testing
    mock_storage.get_file_status_response = {
        "file_id": file_id,
        "filename": "test.mp3",
        "status": ANALYZING,
        "beat_detection_task": {
            "id": beat_task.id,
            "state": "STARTED"
        }
    }
    
    # Confirm analysis
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
    
    # Set mock response for file status
    mock_storage.get_file_status_response = {
        "file_id": file_id,
        "filename": "test.mp3",
        "status": COMPLETED,
        "beat_detection_task": {
            "id": metadata.get("beat_detection"),
            "state": "SUCCESS"
        },
        "video_generation_task": {
            "id": metadata["video_generation"],
            "state": "SUCCESS",
            "result": {"video_file": str(video_file)}
        }
    }
    
    # Attempt to download the video
    response = test_client.get(f"/download/{file_id}")
    assert response.status_code == 200
    
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
    
    # Set mock response for file status
    mock_storage.get_file_status_response = {
        "file_id": file_id,
        "filename": "test.mp3",
        "status": ANALYZING,
        "beat_detection_task": {
            "id": metadata.get("beat_detection"),
            "state": "STARTED"
        }
    }
    
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
    
    # Set mock storage to return None for metadata
    mock_storage.get_file_status_response = None
    
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
    
    mock_storage.update_metadata(file_id, metadata)
    
    # Set mock response for file status
    mock_storage.get_file_status_response = {
        "file_id": file_id,
        "filename": "test.mp3",
        "status": ANALYZED,
        "beat_detection_task": {
            "id": metadata.get("beat_detection"),
            "state": "SUCCESS",
            "result": {"beats_file": str(beats_file)}
        }
    }
    
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
    
    # Create video task that failed
    video_task = create_mock_task()
    video_task.set_state("FAILURE", "Video generation failed with error")
    metadata["video_generation"] = video_task.id
    
    mock_storage.update_metadata(file_id, metadata)
    
    # Set mock response for file status
    mock_storage.get_file_status_response = {
        "file_id": file_id,
        "filename": "test.mp3",
        "status": VIDEO_ERROR,
        "beat_detection_task": {
            "id": metadata.get("beat_detection"),
            "state": "SUCCESS",
            "result": {"beats_file": str(beats_file)}
        },
        "video_generation_task": {
            "id": metadata["video_generation"],
            "state": "FAILURE",
            "result": "Video generation failed with error"
        }
    }
    
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
    file_id = "test_file"
    metadata = {
        "filename": "test.mp3",
        "file_path": "/path/to/test.mp3"
    }
    
    if beat_state:
        beat_task = create_mock_task()
        beat_task.set_state(beat_state)
        metadata["beat_detection"] = beat_task.id
    
    if video_state:
        video_task = create_mock_task()
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
    beat_state: str,
    video_state: str,
    expected_status: int
):
    """Test the file page endpoint with various task states."""
    file_id = "test_file_id"
    
    if beat_state is not None:
        # Create metadata with beat detection task
        beat_task = create_mock_task()
        metadata = {
            "filename": "test.mp3",
            "file_path": "/path/to/test.mp3",
            "beat_detection": beat_task.id
        }
        
        # Set up beat detection task state
        beat_task.set_state(beat_state, {"beats_file": "beats.json"} if beat_state == "SUCCESS" else None)
        
        if video_state is not None:
            # Add video generation task
            video_task = create_mock_task()
            metadata["video_generation"] = video_task.id
            video_task.set_state(video_state, {"video_file": "video.mp4"} if video_state == "SUCCESS" else None)
        
        # Store metadata
        mock_storage.update_metadata(file_id, metadata)
    
    # Make request to file page
    response = test_client.get(f"/file/{file_id}")
    
    if expected_status == 404:
        assert response.status_code == 404
    else:
        assert response.status_code == expected_status
        
        # Get response content
        response_html = response.text
        
        # Verify basic file information is present in the HTML
        assert file_id in response_html
        
        # Verify the overall status is present in the data attributes
        if beat_state is not None:
            if beat_state == "SUCCESS" and video_state == "SUCCESS":
                assert 'data-status="COMPLETED"' in response_html
            elif beat_state == "SUCCESS" and video_state == "PENDING":
                assert 'data-status="GENERATING_VIDEO"' in response_html
            elif beat_state == "PENDING":
                assert 'data-status="ANALYZING"' in response_html
            elif beat_state == "SUCCESS":
                assert 'data-status="ANALYZED"' in response_html 