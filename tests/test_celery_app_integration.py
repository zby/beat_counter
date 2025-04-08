import pytest
import os
import json
from pathlib import Path
import shutil
import numpy as np # Needed for potential BeatDetector interactions
from typing import Dict, Any
import logging # Import logging

from web_app.config import StorageConfig
from web_app.storage import FileMetadataStorage
from web_app.celery_app import _perform_beat_detection, _perform_video_generation # Import tasks and the new helper function
from beat_detection.core.detector import BeatDetector # Import real detector
from beat_detection.core.beats import Beats

# --- Fixtures ---

@pytest.fixture(scope="module") # Scope module cleans once before all tests in this file
def temp_storage_integration(request): # Add request fixture to get workspace path
    """Uses a fixed directory tests/data/celery_test_storage, cleans it before tests, leaves it after."""
    # Define the fixed path relative to the workspace root
    # Assuming the test runs from the workspace root where 'tests/' directory exists
    base_test_dir = Path("tests/data/celery_test_storage")
    print(f"\nUsing fixed test storage directory: {base_test_dir.resolve()}")

    # Clean up before tests
    if base_test_dir.exists():
        print(f"\nCleaning up existing test storage directory: {base_test_dir}")
        shutil.rmtree(base_test_dir)
    base_test_dir.mkdir(parents=True)

    storage_config = StorageConfig(
        # Ensure the path stored is absolute to avoid CWD issues
        upload_dir=base_test_dir.resolve() / "uploads", 
        max_upload_size_mb=10,
        max_audio_secs=60, # Keep it short for tests,
        allowed_extensions=["mp3", "wav", "m4a", "ogg", "flac"]
    )
    
    storage_config.upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the original storage class
    storage = FileMetadataStorage(storage_config)
    
    yield storage  # Provide the storage instance to the test
    
    # No cleanup after test execution - directory remains
    print(f"\nTest execution finished. Leaving storage directory intact: {base_test_dir.resolve()}")

@pytest.fixture
def sample_audio_file(temp_storage_integration):
    """Copies a real sample audio file into the test storage."""
    file_id = "sample_audio_123"
    original_filename = "Besito_a_Besito_10sec.mp3"
    source_audio_path = Path(f"tests/data/{original_filename}")

    # --- Pre-check: Ensure the source sample file exists --- 
    if not source_audio_path.exists():
        pytest.fail(f"Source sample audio file not found: {source_audio_path.resolve()}")
    # --------------------------------------------------------

    # Get the target path within the test storage
    storage = temp_storage_integration
    target_audio_path = storage.get_audio_file_path(file_id)
    target_audio_path.parent.mkdir(parents=True, exist_ok=True) # Ensure upload dir exists

    # Register the file in metadata first (as if uploaded)
    storage.update_metadata(file_id, {'status': 'uploaded', 'original_filename': original_filename})

    # Copy the real audio file to the target location
    shutil.copyfile(source_audio_path, target_audio_path)
    print(f"\nCopied sample audio file from {source_audio_path} to {target_audio_path} for file_id: {file_id}")

    return file_id, target_audio_path

@pytest.fixture
def sample_beats_file(sample_audio_file, temp_storage_integration):
    """Copies a sample beats JSON file into the test storage for the audio file."""
    file_id, audio_path = sample_audio_file # Get info from the dependent fixture
    storage = temp_storage_integration
    source_beats_filename = "Besito_a_Besito_10sec.beats.json"
    source_beats_path = Path(f"tests/data/{source_beats_filename}")

    # --- Pre-check: Ensure the source sample file exists --- 
    if not source_beats_path.exists():
        pytest.fail(f"Source sample beats file not found: {source_beats_path.resolve()}")
    # --------------------------------------------------------

    # Get the target path within the test storage
    target_beats_path = storage.get_beats_file_path(file_id)
    # Ensure job directory exists (where beats file should go)
    storage.ensure_job_directory(file_id)
    
    # Copy the sample beats file
    shutil.copyfile(source_beats_path, target_beats_path)
    print(f"\nCopied sample beats file from {source_beats_path} to {target_beats_path} for file_id: {file_id}")
    
    # Optionally, update metadata to reflect beat detection success (more realistic state)
    storage.update_metadata(file_id, {'beat_detection_status': 'success', 'beat_file': str(target_beats_path)})

    # Return all necessary info for the video test
    return file_id, audio_path, target_beats_path 

# --- Test Functions ---

# No need to patch create_progress_updater if we pass a dummy one directly
def test_detect_beats_task_integration_success(sample_audio_file, temp_storage_integration):
    """
    Integration test for the core beat detection logic (_perform_beat_detection).
    Focuses on ensuring the logic runs, interacts with storage correctly, 
    and produces an output file using a real BeatDetector instance.
    Doesn't validate the *accuracy* of beat detection itself here.
    """
    # --- Arrange ---
    file_id, audio_path = sample_audio_file
    storage = temp_storage_integration # Get the real storage instance
    
    # Parameters for the function (can be adjusted)
    min_bpm = 90
    max_bpm = 180
    tolerance_percent = 10.0 # Use default or specify
    min_measures = 1         # Use default or specify
    beats_per_bar = None     # Use default or specify

    # Simple dummy progress callback for the test
    def dummy_progress_callback(stage: str, value: float):
        print(f"[Test Progress] Stage: {stage}, Value: {value:.2f}")

    # Ensure the audio file actually exists where storage expects it
    assert audio_path.exists(), f"Test setup failed: Audio file {audio_path} not found."
    metadata_before = storage.get_metadata(file_id)
    assert metadata_before['status'] == 'uploaded'
    
    # Ensure the job directory exists (task usually does this, mimic here)
    storage.ensure_job_directory(file_id)

    # --- Act ---
    # Call the core logic function directly
    result = _perform_beat_detection(
        storage=storage, 
        file_id=file_id, 
        min_bpm=min_bpm, 
        max_bpm=max_bpm,
        tolerance_percent=tolerance_percent,
        min_measures=min_measures,
        beats_per_bar=beats_per_bar,
        update_progress=dummy_progress_callback
    )

    # --- Assert ---
    print(f"\nFunction result: {result}") # Debug print

    # 1. Check task status
    assert result is not None, "Task returned None"
    assert result.get('status') == 'success', f"Task failed with error: {result.get('error')}"

    # 2. Check that the output beat file was created
    expected_beats_path = storage.get_beats_file_path(file_id)
    assert 'beat_file' in result
    assert Path(result['beat_file']) == expected_beats_path, f"Result path {result['beat_file']} doesn't match expected {expected_beats_path}"
    assert expected_beats_path.exists(), f"Expected beats file not found at {expected_beats_path}"

    # 3. Check if the beat file has content (basic check)
    assert expected_beats_path.stat().st_size > 0, f"Beats file {expected_beats_path} is empty."

    # 4. Load the beat data (we need some info from it for metadata check)
    beat_data_json = None
    try:
        with open(expected_beats_path, 'r') as f:
            beat_data_json = json.load(f)
        # Verify expected top-level keys based on the actual output
        assert 'beat_list' in beat_data_json
        assert 'beats_per_bar' in beat_data_json
        assert 'overall_stats' in beat_data_json
        assert 'tempo_bpm' in beat_data_json['overall_stats']
        print(f"\nSuccessfully loaded beats JSON from {expected_beats_path}")
    except Exception as e:
        pytest.fail(f"Failed to load or validate beats JSON {expected_beats_path}: {e}")

    # 5. Check job directory creation 
    job_dir = storage.get_job_directory(file_id)
    assert job_dir.exists() and job_dir.is_dir(), f"Job directory {job_dir} not found."
    
    # 6. Verify metadata update
    metadata = storage.get_metadata(file_id)
    assert metadata is not None, f"Metadata not found for file_id {file_id}"
    print(f"\nMetadata after task: {metadata}") # Debug print
    
    assert metadata.get('beat_detection_status') == 'success'
    assert metadata.get('beat_file') == str(expected_beats_path)
    # Compare total beats count with the length of the beat_list
    assert metadata.get('total_beats') == len(beat_data_json['beat_list']) 
    assert metadata.get('detected_beats_per_bar') == beat_data_json.get('beats_per_bar')
    # Calculate irregular beats count from the loaded JSON for comparison
    json_irregular_count = sum(1 for beat in beat_data_json.get('beat_list', []) if beat.get('is_irregular_interval'))
    assert metadata.get('irregular_beats_count') == json_irregular_count
    # Compare BPM with a tolerance - assuming metadata uses overall_stats
    # Check which stats object is actually used by _perform_beat_detection if this fails.
    assert 'detected_tempo_bpm' in metadata
    assert abs(metadata['detected_tempo_bpm'] - beat_data_json['overall_stats']['tempo_bpm']) < 0.01
    print(f"\nSuccessfully verified metadata update for file_id {file_id}")

    # Test beat data JSON structure
    assert isinstance(beat_data_json, dict), "Beat data should be a dictionary"
    assert 'beats_per_bar' in beat_data_json, "Beat data should contain beats_per_bar"
    
    # Test metadata structure
    assert isinstance(metadata, dict), "Metadata should be a dictionary"
    assert metadata.get('detected_beats_per_bar') == beat_data_json.get('beats_per_bar'), "Metadata beats_per_bar mismatch"

    # Check the stats directly in the result dictionary
    assert result['total_beats'] == 22 # Check result directly (Updated expected value)
    assert result['beats_per_bar'] == 4 # Check result directly


def test_generate_video_integration_success(sample_beats_file, temp_storage_integration):
    """
    Integration test for the core video generation logic (_perform_video_generation).
    Focuses on ensuring the logic runs, interacts with storage correctly, 
    loads beats, generates a video file, and updates metadata.
    Requires a pre-existing audio file and beats file (handled by fixtures).
    """
    # --- Arrange ---
    # Get file_id, audio path, and beats path from the fixture
    file_id, audio_path, beats_path = sample_beats_file 
    storage = temp_storage_integration # Get the real storage instance
    workspace_root = Path.cwd() # Capture CWD at the start of the test

    # Simple dummy progress callback for the test
    def dummy_progress_callback(stage: str, value: float):
        print(f"[Test Progress - Video] Stage: {stage}, Value: {value:.2f}")

    # --- Pre-conditions Check ---
    assert audio_path.exists(), f"Test setup failed: Audio file {audio_path} not found."
    assert beats_path.exists(), f"Test setup failed: Beats file {beats_path} not found."
    metadata_before = storage.get_metadata(file_id)
    assert metadata_before.get('beat_detection_status') == 'success', "Metadata should reflect beat success before video gen."
    assert metadata_before.get('beat_file') == str(beats_path), "Metadata beat file path mismatch."

    # --- Act ---
    # Call the core logic function directly
    try:
        result = _perform_video_generation(
            storage=storage, 
            file_id=file_id, 
            update_progress=dummy_progress_callback
        )
        # Add a small delay if video generation might be async in future
        # import time; time.sleep(1) 
    except Exception as e:
        # Capture CWD if error occurs, as _perform_video_generation might not restore it
        cwd_after_error = Path.cwd()
        pytest.fail(f"_perform_video_generation raised an exception: {e}\nCWD at error: {cwd_after_error}", pytrace=True)

    # --- Assert ---
    print(f"\nFunction result (Video): {result}") # Debug print

    # 1. Check function status
    assert result is not None, "Function returned None"
    assert result.get('status') == 'success', f"Function failed with error: {result.get('error')}"

    # 2. Check that the output video file was created
    expected_video_path = storage.get_video_file_path(file_id)
    assert 'video_file' in result
    # Compare absolute paths to avoid relative vs absolute mismatch
    assert Path(result['video_file']).resolve() == expected_video_path.resolve(), \
           f"Result path {result['video_file']} doesn't match expected {expected_video_path}"
    assert expected_video_path.exists(), f"Expected video file not found at {expected_video_path}"
    # Basic check for non-empty file (adjust size threshold if needed)
    assert expected_video_path.stat().st_size > 1000, f"Video file {expected_video_path} seems too small."

    # 3. Verify metadata update for video generation
    metadata_after = storage.get_metadata(file_id)
    assert metadata_after is not None, f"Metadata not found for file_id {file_id} after video gen."
    print(f"\nMetadata after video task: {metadata_after}") # Debug print
    
    assert metadata_after.get('video_generation_status') == 'success', "Video generation status not updated to success in metadata."
    assert metadata_after.get('video_file') == str(expected_video_path), "Video file path not updated in metadata."
    
    # 4. Check CWD restoration (important!)
    # Ensure CWD is back to what it was when the test started.
    assert Path.cwd() == workspace_root, f"Current working directory not restored! Expected {workspace_root}, but got {Path.cwd()}"
    print(f"\nSuccessfully verified video generation and metadata update for file_id {file_id}")


if __name__ == "__main__":
    # Run specific tests by uncommenting/commenting their node IDs
    # This makes it easy to run only one test when debugging directly.
    tests_to_run = [
        f"{__file__}::test_detect_beats_task_integration_success",
        f"{__file__}::test_generate_video_integration_success",
        # Add other test functions from this file here if needed
    ]
    
    # Filter out any commented-out lines (which might result in just __file__::)
    # Though typically one would comment the whole line.
    # A more direct way is just to comment the lines above.
    
    print(f"Running tests: {tests_to_run}")
    pytest.main(tests_to_run)