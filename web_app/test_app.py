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
from datetime import datetime

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
    
    # Get standardized file path using the storage API
    audio_file_path = mock_storage.get_audio_file_path(file_id, ".mp3")
    
    # Create metadata with standardized structure
    metadata = {
        "original_filename": "test.mp3",
        "audio_file_path": str(audio_file_path),
        "file_extension": ".mp3",
        "upload_time": datetime.now().isoformat()
    }
    
    # Create task and update metadata
    task = mock_executor._create_task()
    metadata["beat_detection"] = task.id
    
    # Store metadata
    mock_storage.update_metadata(file_id, metadata)
    
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
        assert task_id in mock_executor.tasks
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

@pytest.mark.asyncio
async def test_confirm_analysis_success(
    test_client: TestClient,
    mock_storage: MockMetadataStorage,
    mock_executor: MockTaskExecutor
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
    beat_task = mock_executor._create_task()
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
    file_status = await mock_storage.get_file_status(file_id, mock_executor)
    assert file_status["status"] == GENERATING_VIDEO

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
    mock_executor: MockTaskExecutor,
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
    beat_task = mock_executor._create_task()
    beat_task.set_state("SUCCESS")
    metadata["beat_detection"] = beat_task.id
    
    # Create the directory and dummy video file
    job_dir = mock_storage.ensure_job_directory(file_id)
    video_file.parent.mkdir(exist_ok=True, parents=True)
    video_file.touch()
    
    # Create video task
    video_task = mock_executor._create_task()
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
    mock_storage: MockMetadataStorage,
    mock_executor: MockTaskExecutor
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
    beat_task = mock_executor._create_task()
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
    mock_storage: MockMetadataStorage,
    mock_executor: MockTaskExecutor
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
    beat_task = mock_executor._create_task()
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
    mock_storage: MockMetadataStorage,
    mock_executor: MockTaskExecutor
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
    beat_task = mock_executor._create_task()
    beat_task.set_state("SUCCESS", {
        "beats_file": str(beats_file)
    })
    metadata["beat_detection"] = beat_task.id
    
    # Create video task that failed
    video_task = mock_executor._create_task()
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
        assert response.status_code == 404
    else:
        assert response.status_code == expected_status
        
        # Get response content
        response_html = response.text
        
        # Verify basic file information is present in the HTML
        assert file_id in response_html
        
        # Verify task status information is present
        if beat_state is not None:
            assert beat_state in response_html
            if beat_state == "SUCCESS":
                assert "beats.json" in response_html
        
        if video_state is not None:
            assert video_state in response_html
            if video_state == "SUCCESS":
                assert "video.mp4" in response_html 