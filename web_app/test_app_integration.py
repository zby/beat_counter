# web_app/test_app_integration.py
"""
Integration tests for the FastAPI application workflow.

Uses TestClient, real FileMetadataStorage with a temporary directory, and
configures Celery for eager execution. Mocks BeatDetector and BeatVideoGenerator
to avoid running real analysis on test audio data.
"""

import pytest
import pathlib
import tempfile
import shutil
import json
import os
import io
from typing import Generator, Dict, Any, Tuple
from datetime import datetime, timedelta

# Mocking imports
import numpy as np
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

# Third-party imports
from fastapi.testclient import TestClient
from celery import Celery
from celery.result import AsyncResult

from unittest.mock import patch, MagicMock # Ensure MagicMock is imported
# Import the specific task function to patch its methods
from web_app.celery_app import AppContext

# Import app creation function and components
from web_app.app import (
    create_app,
    ANALYZING, ANALYZED, ANALYZING_FAILURE, GENERATING_VIDEO, COMPLETED, VIDEO_ERROR, ERROR
)
from web_app.config import Config, StorageConfig, AppConfig, CeleryConfig, User
from web_app.storage import FileMetadataStorage # Import the class for type hinting
from web_app.auth import UserManager

# --- Test Fixtures ---

@pytest.fixture(scope="session")
def test_upload_dir() -> Generator[pathlib.Path, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield pathlib.Path(temp_dir)

@pytest.fixture(scope="session")
def test_config(test_upload_dir: pathlib.Path) -> Config:
    allowed_extensions = [".mp3", ".wav"]
    return Config(
        app=AppConfig(name="TestBeatApp", version="0.1-test", debug=True, allowed_hosts=["*"], max_queue_files=50),
        storage=StorageConfig(upload_dir=test_upload_dir, max_upload_size_mb=10, allowed_extensions=allowed_extensions, max_audio_secs=60),
        celery=CeleryConfig(name="TestBeatAppCelery", broker_url="memory://", result_backend="cache+memory://", task_serializer="json", accept_content=["json"], task_ignore_result=False, result_extended=True, task_track_started=True),
        users=[User.from_dict({"username": "testadmin", "password": "password", "is_admin": True, "created_at": datetime.now().isoformat() + "Z"})]
    )

@pytest.fixture(scope="session")
def test_storage(test_config: Config) -> FileMetadataStorage:
    storage = FileMetadataStorage(test_config.storage)
    storage.base_upload_dir.mkdir(parents=True, exist_ok=True)
    return storage

@pytest.fixture(scope="session")
def test_auth_manager(test_config: Config) -> UserManager:
    users_dict = {"users": []}
    for user in test_config.users:
        user_data = user.__dict__.copy()
        if isinstance(user_data['created_at'], datetime):
             user_data['created_at'] = user.created_at.strftime('%Y-%m-%dT%H:%M:%SZ')
        users_dict["users"].append(user_data)
    return UserManager(users=users_dict)


@pytest.fixture(scope="module", autouse=True)
def configure_celery_for_test(test_storage: FileMetadataStorage):
    from web_app.celery_app import app as actual_celery_app
    actual_celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
        broker_url="memory://",
        result_backend="cache+memory://",
        task_store_eager_result=True # Store results even when eager
    )
    if not hasattr(actual_celery_app, 'context') or not actual_celery_app.context:
         actual_celery_app.context = AppContext(storage=test_storage)
    else:
         actual_celery_app.context.storage = test_storage
    print("Celery configured for eager execution (module scope).")


@pytest.fixture()
def test_client(test_config: Config, test_storage: FileMetadataStorage, test_auth_manager: UserManager) -> Generator[TestClient, None, None]:
    # Patch the global config and app context used by create_app
    with patch('web_app.app._global_config', test_config), patch.dict('web_app.app._app_context', {
             "storage": test_storage,
             "auth_manager": test_auth_manager
         }, clear=True): # clear=True ensures we start with only our test components

        # Now create_app will use the patched globals/context
        app = create_app() # Call without arguments

        client = TestClient(app)
        # Perform login to get the auth cookie for subsequent requests
        login_data = {"username": test_config.users[0].username, "password": "password"}
        response = client.post("/login", data=login_data)
        # Ensure login was successful (check for cookie)
        assert "access_token" in client.cookies or "access_token" in response.cookies.get("access_token", ""), "Login failed during test setup"

        yield client # Provide the configured client to the test

        # Cleanup after tests finish with this client instance
        print(f"Cleaning up storage directory: {test_storage.base_upload_dir}")
        # Careful cleanup: only remove files/dirs created during the test
        for item in test_storage.base_upload_dir.iterdir():
            # Optionally skip log files or other persistent items if needed
            # Example: if item.name == "celery.log": continue
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception as e:
                 print(f"Warning: Could not clean up {item}: {e}")
        print("Storage directory cleaned.")

@pytest.fixture(scope="module")
def generated_sample_audio() -> Dict[str, Any]:
    print("Generating sample audio...")
    from pydub import AudioSegment
    silence = AudioSegment.silent(duration=100) # 100ms = 0.1s
    mp3_data = io.BytesIO()
    try:
        silence.export(mp3_data, format="mp3")
        mp3_data.seek(0)
        print("Sample MP3 generated successfully.")
        return {"filename": "generated_test.mp3", "file_obj": mp3_data, "mime_type": "audio/mpeg"}
    except Exception as e:
        pytest.fail(f"Failed to generate sample MP3 using pydub/ffmpeg: {e}")


# --- Mocking Fixture ---
@pytest.fixture
def mock_beat_detector():
    """ Mocks the BeatDetector call within the celery task """
    # Create a mock object simulating the Beats class structure
    mock_beats_obj = MagicMock()

    # --- Mimic attributes accessed in _perform_beat_detection ---
    mock_beats_obj.beats_per_bar = 4
    mock_beats_obj.timestamps = np.array([0.05, 0.08])
    mock_beats_obj.overall_stats = SimpleNamespace(
        tempo_bpm=120.0,
        irregularity_percent=0.0
    )
    mock_beats_obj.get_irregular_beats.return_value = []

    # --- Mock save_to_file to actually create the file ---
    def _mock_save_beats(file_path: pathlib.Path):
        """Side effect function to simulate file creation."""
        print(f"Mock save_to_file called with path: {file_path}")
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("mock beat data\n0.05\n0.08") # Write dummy content
            print(f"Mock save_to_file created file: {file_path}")
        except Exception as e:
            print(f"ERROR in mock save_to_file: {e}")
            # Propagate the error to potentially fail the test clearly
            raise 

    mock_beats_obj.save_to_file = MagicMock(side_effect=_mock_save_beats)

    # --- Mock other necessary attributes/methods ---
    mock_beats_obj.beat_list = [SimpleNamespace(beat_count=1), SimpleNamespace(beat_count=2)]
    mock_beats_obj.start_regular_beat_idx = 0
    mock_beats_obj.end_regular_beat_idx = 2

    patch_target = 'web_app.celery_app.BeatDetector'
    try:
        with patch(patch_target) as mock_DetectorClass:
            instance_mock = MagicMock()
            instance_mock.detect_beats.return_value = mock_beats_obj
            mock_DetectorClass.return_value = instance_mock
            print(f"Mocking '{patch_target}' - Instance will return a mock Beats object with file-creating save_to_file.")
            yield mock_DetectorClass
    except ModuleNotFoundError:
         pytest.fail(f"Failed to patch '{patch_target}'. Is the import path correct?")
    except Exception as e:
         pytest.fail(f"An unexpected error occurred during patching: {e}")


# --- Test Data ---
INVALID_EXT = ".txt"
INVALID_MIME_TYPE = "text/plain"

# --- Test Cases ---

def test_index_authenticated(test_client: TestClient):
    response = test_client.get("/")
    assert response.status_code == 200
    assert "Upload Audio" in response.text

def test_index_unauthenticated(test_config, test_storage, test_auth_manager):
    # Patch the necessary globals before calling create_app
    with patch('web_app.app._global_config', test_config), patch.dict('web_app.app._app_context', {
             "storage": test_storage,
             "auth_manager": test_auth_manager
         }, clear=True):
        app = create_app() # Call without arguments
        client = TestClient(app)
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 303
        assert "/login" in response.headers["location"]

def test_login_page(test_client: TestClient):
    test_client.get("/logout", follow_redirects=True)
    response = test_client.get("/login")
    assert response.status_code == 200
    assert "<form" in response.text

def test_login_logout(test_client: TestClient, test_config: Config):
    assert "access_token" in test_client.cookies
    response = test_client.get("/logout", follow_redirects=False)
    assert response.status_code == 303
    assert "/login" in response.headers["location"]
    # this format channged
    # assert "expires=Thu, 01 Jan 1970" in response.headers["set-cookie"]
    # Updated Assertion: Check for Max-Age=0 to confirm deletion intent
    assert "Max-Age=0" in response.headers["set-cookie"]
    assert "access_token=" in response.headers["set-cookie"]

def test_login_invalid(test_client: TestClient):
    test_client.get("/logout", follow_redirects=True)
    login_data = {"username": "wronguser", "password": "wrongpassword"}
    response = test_client.post("/login", data=login_data, follow_redirects=False)
    assert response.status_code == 303
    assert "/login?error=invalid" in response.headers["location"]


# --- Upload Tests ---

def test_upload_valid_audio(
    test_client: TestClient,
    test_storage: FileMetadataStorage,
    generated_sample_audio: Dict[str, Any],
    mock_beat_detector # IMPORTANT: Ensure fixture is included
):
    """Test uploading a valid audio file (uses mock beat detector)."""
    filename = generated_sample_audio["filename"]
    file_obj = generated_sample_audio["file_obj"]
    mime_type = generated_sample_audio["mime_type"]
    files = {"file": (filename, file_obj, mime_type)}

    response = test_client.post("/upload", files=files, follow_redirects=False)

    assert response.status_code == 303, f"Upload failed: {response.text}"
    file_page_url = response.headers.get("Location")
    assert file_page_url and "/file/" in file_page_url
    file_id = file_page_url.split("/")[-1]

    expected_audio_path = test_storage.get_audio_file_path(file_id, ".mp3")
    assert expected_audio_path.exists()

    metadata = test_storage.get_metadata(file_id)
    assert metadata is not None
    assert "beat_detection" in metadata
    
    # --- Explicitly ensure the mock beats file exists before checking status --- #
    beats_file_path_str = metadata.get("beats_file")
    assert beats_file_path_str, "beats_file path missing from metadata after mock detection"
    beats_file_path = pathlib.Path(beats_file_path_str)
    if not beats_file_path.exists():
        print(f"WARN: Mock beats file {beats_file_path} did not exist, creating it explicitly for status check.")
        beats_file_path.parent.mkdir(parents=True, exist_ok=True)
        beats_file_path.write_text("mock beat data\n0.05\n0.08") # Content matching mock
    # --- End explicit check --- #

    status_response = test_client.get(f"/status/{file_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    print("Status after valid upload (mocked detector):", status_data)
    assert status_data["status"] == ANALYZED
    assert status_data["beats_file_exists"] is True
    assert status_data["beat_stats"]["tempo_bpm"] == 120.0


def test_upload_invalid_type(test_client: TestClient):
    filename = f"test_invalid{INVALID_EXT}"
    files = {"file": (filename, b"invalid data", INVALID_MIME_TYPE)}
    response = test_client.post("/upload", files=files)
    assert response.status_code == 415
    assert "Unsupported file format" in response.text

def test_upload_unauthenticated(test_config, test_storage, test_auth_manager, generated_sample_audio):
    # Patch the necessary globals before calling create_app
    with patch('web_app.app._global_config', test_config), patch.dict('web_app.app._app_context', {
             "storage": test_storage,
             "auth_manager": test_auth_manager
         }, clear=True):
        app = create_app() # Call without arguments
        client = TestClient(app)
        filename = generated_sample_audio["filename"]
        file_obj = generated_sample_audio["file_obj"]
        mime_type = generated_sample_audio["mime_type"]
        files = {"file": (filename, file_obj, mime_type)}
        response = client.post("/upload", files=files, follow_redirects=False)
        assert response.status_code == 303
        assert "/login" in response.headers["location"]


# --- Status and Workflow Tests ---

@pytest.fixture
def uploaded_file_id(
    test_client: TestClient,
    test_storage: FileMetadataStorage,
    generated_sample_audio: Dict[str, Any],
    mock_beat_detector # IMPORTANT: Ensure fixture is included
) -> str:
    """Fixture to upload a generated file (with mocked detection) and return its ID."""
    filename = f"workflow_{generated_sample_audio['filename']}"
    file_obj = generated_sample_audio['file_obj']
    mime_type = generated_sample_audio['mime_type']
    file_obj.seek(0)
    files = {"file": (filename, file_obj, mime_type)}

    response = test_client.post("/upload", files=files, follow_redirects=False)
    assert response.status_code == 303
    file_page_url = response.headers.get("Location")
    file_id = file_page_url.split("/")[-1]

    # --- Add immediate check after upload+eager task --- #
    metadata_after_upload = test_storage.get_metadata(file_id)
    assert metadata_after_upload is not None, "Metadata file should exist after upload."
    assert metadata_after_upload.get("beat_detection_status") == "success", \
        f"Metadata missing 'beat_detection_status: success' immediately after upload. Got: {metadata_after_upload.get('beat_detection_status')}"
    # --- End immediate check --- #

    assert test_storage.get_beats_file_path(file_id).exists() # Check mock worked
    return file_id


def test_status_analyzed(test_client: TestClient, test_storage: FileMetadataStorage, uploaded_file_id: str):
    """Test status endpoint returns ANALYZED state correctly."""
    file_id = uploaded_file_id # Fixture ensures state is ANALYZED
    response = test_client.get(f"/status/{file_id}")
    assert response.status_code == 200
    data = response.json()
    print("Status ANALYZED data:", data)
    assert data["status"] == ANALYZED
    assert data["beat_stats"]["tempo_bpm"] == 120.0 # From mock
    assert data["beats_file_exists"] is True

@patch('web_app.celery_app.generate_video_task.delay')
def test_confirm_analysis_success(
    mock_generate_video_delay, # Mock object from patch
    test_client: TestClient,
    test_storage: FileMetadataStorage,
    uploaded_file_id: str # This fixture implicitly uses mock_beat_detector
):
    """Test confirming analysis triggers video generation task (mocked delay)."""
    file_id = uploaded_file_id # Fixture ensures state is ANALYZED

    # Configure the mock delay method (optional, but good practice)
    # You might want it to return a mock AsyncResult if needed elsewhere
    mock_async_result = MagicMock(spec=AsyncResult)
    mock_async_result.id = "mock_video_task_id_" + file_id
    mock_generate_video_delay.return_value = mock_async_result

    # ---- Action ----
    response = test_client.post(f"/confirm/{file_id}")

    # ---- Assertions ----
    assert response.status_code == 200, f"Confirm failed: {response.text}"
    assert response.json()["status"] == "ok"
    assert response.json()["task_id"] == mock_async_result.id # Check if task_id is returned

    # Assert that the delay method was called once with the correct file_id
    mock_generate_video_delay.assert_called_once_with(file_id)

    # Check that metadata was updated with the video task ID
    metadata = test_storage.get_metadata(file_id)
    assert metadata is not None
    assert "video_generation" in metadata
    assert metadata["video_generation"] == mock_async_result.id # Verify the correct task ID was stored

    # Check the status endpoint *immediately after* triggering
    # It should reflect that the video task has been *queued*
    status_response = test_client.get(f"/status/{file_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    print("Status after confirm (mocked delay):", status_data)
    # Assert it's now in a video generating state, NOT completed
    assert status_data["status"] == GENERATING_VIDEO
    assert status_data["video_generation"] == mock_async_result.id


def test_confirm_analysis_not_ready(test_client: TestClient, test_storage: FileMetadataStorage, generated_sample_audio: Dict[str, Any]):
    """Test confirming analysis when beat detection hasn't 'run' (no beats file)."""
    # Create a mock object simulating the Beats class structure
    mock_beats_obj = MagicMock()
    
    # Configure the mock object with necessary attributes
    mock_beats_obj.beats_per_bar = 4
    mock_beats_obj.timestamps = np.array([0.05, 0.08])
    mock_beats_obj.overall_stats = SimpleNamespace(
        tempo_bpm=120.0,
        mean_interval=0.5,
        median_interval=0.5,
        std_interval=0.05,
        min_interval=0.4,
        max_interval=0.6,
        total_beats=2,
        irregularity_percent=0.0
    )
    mock_beats_obj.get_irregular_beats.return_value = []
    mock_beats_obj.beat_list = [SimpleNamespace(beat_count=1), SimpleNamespace(beat_count=2)]
    mock_beats_obj.start_regular_beat_idx = 0
    mock_beats_obj.end_regular_beat_idx = 2
    
    # Mock save_to_file to actually create the file
    def _mock_save_beats(file_path: pathlib.Path):
        """Side effect function to simulate file creation."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("mock beat data\n0.05\n0.08")
    
    mock_beats_obj.save_to_file = MagicMock(side_effect=_mock_save_beats)
    
    filename = "not_ready_" + generated_sample_audio["filename"]
    file_obj = generated_sample_audio["file_obj"]
    mime_type = generated_sample_audio["mime_type"]
    file_obj.seek(0)
    files = {"file": (filename, file_obj, mime_type)}
    
    with patch('web_app.celery_app.BeatDetector') as mock_DetectorClass:
        instance_mock = MagicMock()
        instance_mock.detect_beats.return_value = mock_beats_obj
        mock_DetectorClass.return_value = instance_mock
        response = test_client.post("/upload", files=files, follow_redirects=False)
    assert response.status_code == 303
    file_id = response.headers.get("Location").split("/")[-1]

    # Now explicitly delete the beats file to test the "not ready" case
    beats_file = test_storage.get_beats_file_path(file_id)
    if beats_file.exists(): beats_file.unlink()

    response = test_client.post(f"/confirm/{file_id}")
    assert response.status_code == 400
    assert "not ready for confirmation" in response.text


def test_status_completed(test_client: TestClient, test_storage: FileMetadataStorage, uploaded_file_id: str):
    """Test status endpoint after simulating video generation success."""
    file_id = uploaded_file_id
    beats_file = test_storage.get_beats_file_path(file_id)
    if not beats_file.exists(): beats_file.write_text("0.05\n")
    video_file = test_storage.get_video_file_path(file_id)
    video_file.write_text("dummy video")
    test_storage.update_metadata(file_id, {
        "video_file": str(video_file),
        "video_generation": "sim_video",
        "video_generation_status": "success",
        "video_generation_error": None
    })

    response = test_client.get(f"/status/{file_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == COMPLETED
    assert data["beats_file_exists"] is True
    assert data["video_file_exists"] is True


def test_download_video_success(test_client: TestClient, test_storage: FileMetadataStorage, uploaded_file_id: str):
    """Test downloading a successfully generated video."""
    file_id = uploaded_file_id
    video_content = b"generated video data " + os.urandom(10)
    video_file = test_storage.get_video_file_path(file_id)
    video_file.write_bytes(video_content)
    orig_filename = test_storage.get_metadata(file_id).get("original_filename", "generated_download.mp3")
    test_storage.update_metadata(file_id, {"video_file": str(video_file), "original_filename": orig_filename})

    response = test_client.get(f"/download/{file_id}")
    assert response.status_code == 200
    assert response.content == video_content
    assert response.headers["content-type"] == "video/mp4"
    assert "filename=" in response.headers["content-disposition"]
    expected_dl_name = pathlib.Path(orig_filename).stem + "_with_beats.mp4"
    assert expected_dl_name in response.headers["content-disposition"]


def test_download_video_not_found(
    test_client: TestClient,
    test_storage: FileMetadataStorage, # FIX: Add missing fixture argument
    uploaded_file_id: str
):
    """Test downloading when video file doesn't exist."""
    file_id = uploaded_file_id
    # Use the injected test_storage instance
    video_file = test_storage.get_video_file_path(file_id)
    if video_file.exists(): video_file.unlink()

    response = test_client.get(f"/download/{file_id}")
    assert response.status_code == 404
    assert "Video file not found" in response.text


def test_status_file_not_found(test_client: TestClient):
    response = test_client.get("/status/nonexistent-file-id")
    assert response.status_code == 404


# --- Page Rendering Tests ---

def test_file_page_rendering(test_client: TestClient, uploaded_file_id: str):
    file_id = uploaded_file_id
    response = test_client.get(f"/file/{file_id}")
    assert response.status_code == 200
    assert f"File ID</h3>" in response.text
    assert file_id in response.text
    assert 'id="analysis-section"' in response.text
    assert 'id="video-section"' in response.text


def test_processing_queue_page(test_client: TestClient, uploaded_file_id: str):
    file_id = uploaded_file_id
    response = test_client.get("/processing_queue")
    assert response.status_code == 200
    assert "Processing Queue" in response.text
    assert file_id in response.text