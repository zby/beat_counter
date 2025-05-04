import pytest
import os
import json
from pathlib import Path
import shutil
import numpy as np  # Needed for potential BeatDetector interactions
from typing import Dict, Any
import logging  # Import logging

from web_app.config import StorageConfig
from web_app.storage import FileMetadataStorage
from web_app.celery_app import (
    _perform_beat_detection,
    _perform_video_generation,
)  # Import tasks and the new helper function
from beat_detection.core.detector import BeatDetector # Import real detector
from beat_detection.core.factory import get_beat_detector # Import factory function
from beat_detection.core.beats import Beats, RawBeats # Import RawBeats

# --- Fixtures ---


@pytest.fixture(
    scope="module"
)  # Scope module cleans once before all tests in this file
def temp_storage_integration(request):  # Add request fixture to get workspace path
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
        max_audio_secs=60,  # Keep it short for tests,
        allowed_extensions=["mp3", "wav", "m4a", "ogg", "flac"],
    )

    storage_config.upload_dir.mkdir(parents=True, exist_ok=True)

    # Use the original storage class
    storage = FileMetadataStorage(storage_config)

    yield storage  # Provide the storage instance to the test

    # No cleanup after test execution - directory remains
    print(
        f"\nTest execution finished. Leaving storage directory intact: {base_test_dir.resolve()}"
    )


@pytest.fixture
def sample_audio_file(temp_storage_integration):
    """Copies a real sample audio file from tests/fixtures into the test storage."""
    file_id = "sample_audio_123"
    original_filename = "Besito_a_Besito_10sec.mp3"
    # Use tests/fixtures as the source directory
    source_audio_path = Path(f"tests/fixtures/{original_filename}")

    # --- Pre-check: Ensure the source sample file exists ---
    if not source_audio_path.exists():
        pytest.fail(
            f"Source sample audio file not found: {source_audio_path.resolve()}"
        )
    # --------------------------------------------------------

    # Get the target path within the test storage
    storage = temp_storage_integration
    target_audio_path = storage.get_audio_file_path(file_id)
    target_audio_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure upload dir exists

    # Register the file in metadata first (as if uploaded)
    storage.update_metadata(
        file_id, {"status": "uploaded", "original_filename": original_filename}
    )

    # Copy the real audio file to the target location
    shutil.copyfile(source_audio_path, target_audio_path)
    print(
        f"\nCopied sample audio file from {source_audio_path} to {target_audio_path} for file_id: {file_id}"
    )

    return file_id, target_audio_path


@pytest.fixture
def sample_beats_file(sample_audio_file, temp_storage_integration):
    """Generates a RawBeats JSON file using a detector for the sample audio file."""
    file_id, audio_path = sample_audio_file  # Get info from the dependent fixture
    storage = temp_storage_integration

    # --- 1. Perform Beat Detection ---
    print(f"\nGenerating beats for {audio_path} using 'madmom' detector...")
    try:
        detector: BeatDetector = get_beat_detector("madmom") # Use madmom for consistency or choose another
        raw_beats: RawBeats = detector.detect(str(audio_path))
        assert raw_beats is not None, "Beat detection did not return a RawBeats object."
        assert len(raw_beats.timestamps) > 0, "Beat detection returned no timestamps."
        print(f"\nDetected {len(raw_beats.timestamps)} beats.")
    except Exception as e:
        pytest.fail(f"Beat detection failed during fixture setup: {e}")

    # --- 2. Save Beats File ---
    target_beats_path = storage.get_beats_file_path(file_id)
    storage.ensure_job_directory(file_id) # Ensure job directory exists

    try:
        raw_beats.save_to_file(target_beats_path)
        print(f"\nSaved generated RawBeats file to: {target_beats_path}")
        assert target_beats_path.exists(), "Generated beats file was not saved."
        assert target_beats_path.stat().st_size > 0, "Generated beats file is empty."
    except Exception as e:
        pytest.fail(f"Failed to save generated RawBeats file: {e}")


    # --- 3. Update Metadata (mimicking _perform_beat_detection success) ---
    # We need to store the detected bpb and the parameters used for *potential*
    # reconstruction later (even though this fixture doesn't reconstruct).
    # For the video test, we'll use some default/test reconstruction params.
    # NOTE: The video test logic might need to be adjusted if it relies on specific
    # reconstruction params being present from a *prior* detection step.
    # Let's assume the video test will use its own defaults or fixed values if needed.
    test_tolerance = 10.0 # Define a typical tolerance for metadata
    test_min_measures = 5 # Define a typical min_measures for metadata

    storage.update_metadata(
        file_id,
        {
            "beat_detection_status": "success",
            "beat_file": str(target_beats_path),
            "total_beats": len(raw_beats.timestamps),
            # Store placeholder reconstruction params needed by video generation
            "reconstruction_params": {
                "tolerance_percent": test_tolerance,
                "min_measures": test_min_measures
            }
        },
    )
    print(f"\nUpdated metadata for {file_id} after generating beats.")

    # Return all necessary info for the video test
    return file_id, audio_path, target_beats_path


# --- Test Functions ---


# No need to patch create_progress_updater if we pass a dummy one directly
def test_detect_beats_task_integration_success(
    sample_audio_file, temp_storage_integration
):
    """
    Integration test for the core beat detection logic (_perform_beat_detection).
    Focuses on ensuring the logic runs, interacts with storage correctly,
    and produces an output file using a real BeatDetector instance.
    Doesn't validate the *accuracy* of beat detection itself here.
    """
    # --- Arrange ---
    file_id, audio_path = sample_audio_file
    storage = temp_storage_integration  # Get the real storage instance

    # Define explicit params for detection and reconstruction
    # Restore min/max bpm definitions
    min_bpm = 90
    max_bpm = 180
    beats_per_bar = 4
    tolerance_percent = 15.0
    min_measures = 2

    # Simple dummy progress callback for the test
    def dummy_progress_callback(stage: str, value: float):
        print(f"[Test Progress] Stage: {stage}, Value: {value:.2f}")

    # Ensure the audio file actually exists where storage expects it
    assert audio_path.exists(), f"Test setup failed: Audio file {audio_path} not found."
    metadata_before = storage.get_metadata(file_id)
    assert metadata_before["status"] == "uploaded"

    # Ensure the job directory exists (task usually does this, mimic here)
    storage.ensure_job_directory(file_id)

    # --- Act ---
    # Call the core logic function directly
    result = _perform_beat_detection(
        storage=storage,
        file_id=file_id,
        algorithm="madmom",
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        beats_per_bar=beats_per_bar,
        tolerance_percent=tolerance_percent,
        min_measures=min_measures,
        update_progress=dummy_progress_callback,
    )

    # --- Assert ---
    print(f"\nFunction result: {result}")  # Debug print

    # 1. Check task status
    # The function returns None, the status is checked via metadata below

    # 2. Check that the output beat file was created
    expected_beats_path = storage.get_beats_file_path(file_id)
    assert (
        expected_beats_path.exists()
    ), f"Expected beats file not found at {expected_beats_path}"

    # 3. Check if the beat file has content (basic check)
    assert (
        expected_beats_path.stat().st_size > 0
    ), f"Beats file {expected_beats_path} is empty."

    # 4. Load the RAW beat data and verify structure
    raw_beat_data: RawBeats = None
    try:
        raw_beat_data = RawBeats.load_from_file(expected_beats_path)
        assert isinstance(raw_beat_data, RawBeats)
        assert isinstance(raw_beat_data.timestamps, np.ndarray)
        assert isinstance(raw_beat_data.beat_counts, np.ndarray)
        assert raw_beat_data.timestamps.ndim == 1
        assert raw_beat_data.timestamps.shape == raw_beat_data.beat_counts.shape
        assert len(raw_beat_data.timestamps) > 0 # Check that some beats were detected
        print(f"\nSuccessfully loaded RawBeats JSON from {expected_beats_path}")
    except Exception as e:
        pytest.fail(f"Failed to load or validate RawBeats JSON {expected_beats_path}: {e}")

    # 5. Check job directory creation
    job_dir = storage.get_job_directory(file_id)
    assert job_dir.exists() and job_dir.is_dir(), f"Job directory {job_dir} not found."

    # 6. Verify metadata update
    metadata_after = storage.get_metadata(file_id)
    assert metadata_after is not None, f"Metadata not found for file_id {file_id}"
    print(f"\nMetadata after task: {metadata_after}")  # Debug print

    assert metadata_after.get("beat_detection_status") == "success"
    assert metadata_after.get("beats_file") == str(expected_beats_path)
    assert metadata_after.get("total_beats") == len(raw_beat_data.timestamps)

    # Check analysis params are stored
    analysis_params = metadata_after.get("analysis_params")
    assert analysis_params is not None, "'analysis_params' not found in metadata"
    assert analysis_params.get("beats_per_bar_override") == beats_per_bar
    assert analysis_params.get("tolerance_percent") == tolerance_percent
    assert analysis_params.get("min_measures") == min_measures
    # Assertions for other fields if needed

    # 7. Reconstruct Beats from RawBeats and Metadata Params to check derived values
    reconstructed_beats: Beats = None
    try:
        # Get analysis params from metadata (already asserted they exist)
        bpb_override = analysis_params.get("beats_per_bar_override") # Could be None
        tol = float(analysis_params["tolerance_percent"])
        meas = int(analysis_params["min_measures"])

        # Use the standard Beats constructor with the loaded RawBeats and analysis params
        reconstructed_beats = Beats(
            raw_beats=raw_beat_data,
            beats_per_bar=bpb_override, # Pass the override (constructor handles None)
            tolerance_percent=tol,
            min_measures=meas
        )
    except Exception as recon_e:
        pytest.fail(f"Failed to reconstruct Beats from {expected_beats_path} using metadata params: {recon_e}")

    # Now check the metadata values that depend on the reconstructed Beats object
    assert reconstructed_beats is not None, "Reconstruction failed silently."
    assert metadata_after.get("detected_beats_per_bar") == reconstructed_beats.beats_per_bar # Check inferred/overridden bpb
    assert metadata_after.get("irregular_beats_count") == len(reconstructed_beats.irregular_beat_indices)
    assert "detected_tempo_bpm" in metadata_after
    assert abs(metadata_after["detected_tempo_bpm"] - reconstructed_beats.overall_stats.tempo_bpm) < 0.01

    print(f"\nSuccessfully verified metadata update for file_id {file_id}")


def test_generate_video_integration_success(
    sample_beats_file, temp_storage_integration
):
    """
    Integration test for the core video generation logic (_perform_video_generation).
    Focuses on ensuring the logic runs, interacts with storage correctly,
    loads beats, generates a video file, and updates metadata.
    Requires a pre-existing audio file and beats file (handled by fixtures).
    """
    # --- Arrange ---
    file_id, audio_path, beats_path = sample_beats_file
    # NOTE: sample_beats_file fixture copies a pre-existing .beats.json file.
    # This file MUST be updated manually or regenerated to contain the new RawBeats format
    # (including beats_per_bar, excluding tol/meas) for this test to pass.
    # We also need the tol/meas that correspond to that file for reconstruction.
    # Let's assume the copied file corresponds to these parameters:
    expected_tolerance = 10.0
    expected_min_measures = 5
    # We need to save these into metadata like the detection task would
    storage = temp_storage_integration
    # Store the analysis params that would have been saved by the detection task
    storage.update_metadata(file_id, {
        "analysis_params": {
             "beats_per_bar_override": 4, # Assume 4 for this test fixture
             "tolerance_percent": expected_tolerance,
             "min_measures": expected_min_measures
        }
    })

    # --- Load and Validate Sample Beats File Format ---
    # Ensure the sample file is in the expected RawBeats format before proceeding.
    raw_beat_data_video: RawBeats = None
    try:
        raw_beat_data_video = RawBeats.load_from_file(beats_path)
        assert isinstance(raw_beat_data_video, RawBeats), "Sample beats file is not a RawBeats instance."
        # assert raw_beat_data_video.beats_per_bar is not None, "Sample beats file missing 'beats_per_bar'." # REMOVED: RawBeats no longer has bpb
        assert isinstance(raw_beat_data_video.timestamps, np.ndarray), "Sample beats file 'timestamps' is not a numpy array."
        assert isinstance(raw_beat_data_video.beat_counts, np.ndarray), "Sample beats file 'beat_counts' is not a numpy array."
        assert len(raw_beat_data_video.timestamps) > 0, "Sample beats file has no timestamps."
        print(f"\nSuccessfully loaded and validated RawBeats structure from sample file: {beats_path}")
    except Exception as e:
        pytest.fail(f"Failed to load or validate sample RawBeats JSON {beats_path}: {e}")
    # ----------------------------------------------------

    workspace_root = Path.cwd()  # Capture CWD at the start of the test

    # Simple dummy progress callback for the test
    def dummy_progress_callback(stage: str, value: float):
        print(f"[Test Progress - Video] Stage: {stage}, Value: {value:.2f}")

    # --- Pre-conditions Check ---
    assert audio_path.exists(), f"Test setup failed: Audio file {audio_path} not found."
    assert beats_path.exists(), f"Test setup failed: Beats file {beats_path} not found."
    metadata_before = storage.get_metadata(file_id)
    assert (
        metadata_before.get("beat_detection_status") == "success"
    ), "Metadata should reflect beat success before video gen."
    assert metadata_before.get("beat_file") == str(
        beats_path
    ), "Metadata beat file path mismatch."

    # --- Act ---
    # Call the core logic function directly
    try:
        result = _perform_video_generation(
            storage=storage, file_id=file_id, update_progress=dummy_progress_callback
        )
        # Add a small delay if video generation might be async in future
        # import time; time.sleep(1)
    except Exception as e:
        # Capture CWD if error occurs, as _perform_video_generation might not restore it
        cwd_after_error = Path.cwd()
        pytest.fail(
            f"_perform_video_generation raised an exception: {e}\nCWD at error: {cwd_after_error}",
            pytrace=True,
        )

    # --- Assert ---
    print(f"\nFunction result (Video): {result}")  # Debug print

    # 1. Check function status
    # The function returns None, status checked via side effects below.

    # 2. Check that the output video file was created
    expected_video_path = storage.get_video_file_path(file_id)
    assert (
        expected_video_path.exists()
    ), f"Expected video file not found at {expected_video_path}"
    # Basic check for non-empty file (adjust size threshold if needed)
    assert (
        expected_video_path.stat().st_size > 1000
    ), f"Video file {expected_video_path} seems too small."

    # 3. Verify metadata update for video generation
    metadata_after = storage.get_metadata(file_id)
    assert (
        metadata_after is not None
    ), f"Metadata not found for file_id {file_id} after video gen."
    print(f"\nMetadata after video task: {metadata_after}")  # Debug print

    assert (
        metadata_after.get("video_generation_status") == "success"
    ), "Video generation status not updated to success in metadata."
    assert metadata_after.get("video_file") == str(
        expected_video_path
    ), "Video file path not updated in metadata."

    # 4. Check CWD restoration
    assert (
        Path.cwd() == workspace_root
    ), f"CWD not restored! Expected {workspace_root}, got {Path.cwd()}"
    print(
        f"\nSuccessfully verified video generation and metadata update for file_id {file_id}"
    )


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
