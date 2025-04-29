import pytest
import os  # Import os for file operations
from pathlib import Path
from typing import Tuple, Iterator
from beat_detection.core.detector import BeatDetector
from beat_detection.core.beats import BeatCalculationError, Beats, RawBeats
from beat_detection.core.video import BeatVideoGenerator  # Import video generator
from beat_detection.utils.beat_file import load_raw_beats # Import the new loading function
import numpy as np

# Define the path to the test audio file relative to the tests directory
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_AUDIO_FILE = TEST_DATA_DIR / "Besito_a_Besito_10sec.mp3"
OUTPUT_VIDEO_FILE = TEST_DATA_DIR / "Besito_a_Besito_10sec.mp4"  # Define output path

# Define parameters used across tests
TEST_TOLERANCE_PERCENT = 10.0
TEST_MIN_MEASURES = 5

@pytest.fixture(scope="module")
def detected_raw_beats_file(tmp_path_factory) -> Iterator[Tuple[Path, RawBeats, float, int]]:
    """
    Fixture to perform beat detection on the test audio file and save the
    RawBeats result to a temporary file. Runs once per module.

    Yields:
        Tuple containing:
        - Path to the saved RawBeats JSON file.
        - The original RawBeats object detected (for comparison).
        - Tolerance percentage used for reconstruction.
        - Minimum measures used for reconstruction.
    """
    # Ensure the test audio file exists
    if not TEST_AUDIO_FILE.is_file():
        pytest.fail(f"Test audio file not found: {TEST_AUDIO_FILE}")

    # Use a temporary directory for this fixture's output
    temp_dir = tmp_path_factory.mktemp("beat_detection_output")
    output_beats_file = temp_dir / f"{TEST_AUDIO_FILE.stem}.beats.json"

    raw_beats: RawBeats = None
    try:
        detector = BeatDetector(beats_per_bar=None) # Auto-detect bpb
        raw_beats = detector.detect_beats(str(TEST_AUDIO_FILE))
        assert raw_beats is not None, "RawBeats object was not created."
        assert raw_beats.timestamps.shape[0] > 0, "No raw beats were detected."
        assert raw_beats.beats_per_bar > 0, "Invalid beats_per_bar in RawBeats."
        print(f"[Fixture] Detected raw beats with bpb={raw_beats.beats_per_bar}")

        # Save Raw Beats to the temporary file
        raw_beats.save_to_file(output_beats_file)
        print(f"[Fixture] Saved raw beats to: {output_beats_file}")
        assert output_beats_file.is_file(), f"Raw beats file was not created at {output_beats_file}"
        assert output_beats_file.stat().st_size > 0, f"Raw beats file {output_beats_file} is empty."

        yield output_beats_file, raw_beats, TEST_TOLERANCE_PERCENT, TEST_MIN_MEASURES

    except BeatCalculationError as e:
        pytest.fail(f"Beat detection failed in fixture for {TEST_AUDIO_FILE.name}: {e}")
    except Exception as e:
        pytest.fail(
            f"An unexpected error occurred during beat detection/saving in fixture: {e}"
        )
    # tmp_path cleanup is handled automatically by pytest


def test_load_and_reconstruct_beats(detected_raw_beats_file):
    """
    Test loading RawBeats from a file and reconstructing the full Beats object.
    Relies on the detected_raw_beats_file fixture.
    """
    beats_file_path, original_raw_beats, tolerance_percent, min_measures = detected_raw_beats_file

    # --- Load Raw Beats from File ---
    loaded_raw_beats: RawBeats = None
    try:
        loaded_raw_beats = load_raw_beats(str(beats_file_path))
        print(f"[Fixture] Loaded raw beats data from: {beats_file_path}")
        assert loaded_raw_beats is not None, "Failed to load raw beats from file."
        # Compare loaded RawBeats with the one saved by the fixture
        np.testing.assert_array_equal(loaded_raw_beats.timestamps, original_raw_beats.timestamps)
        np.testing.assert_array_equal(loaded_raw_beats.beat_counts, original_raw_beats.beat_counts)
        assert loaded_raw_beats.beats_per_bar == original_raw_beats.beats_per_bar

    except FileNotFoundError:
        pytest.fail(f"Raw beats file not found for loading: {beats_file_path}")
    except Exception as e:
        pytest.fail(f"Failed to load RawBeats file {beats_file_path}: {e}")

    # --- Reconstruct Beats from *loaded* data ---
    try:
        reconstructed_beats = Beats.from_timestamps(
            timestamps=loaded_raw_beats.timestamps,
            beat_counts=loaded_raw_beats.beat_counts,
            beats_per_bar=loaded_raw_beats.beats_per_bar,
            tolerance_percent=tolerance_percent,
            min_measures=min_measures
        )
        assert reconstructed_beats is not None, "Failed to reconstruct Beats object from loaded data."
        assert reconstructed_beats.beats_per_bar == loaded_raw_beats.beats_per_bar
        assert reconstructed_beats.tolerance_percent == tolerance_percent
        assert reconstructed_beats.min_measures == min_measures
        print(f"Reconstructed Beats object from loaded data with bpb={reconstructed_beats.beats_per_bar}")

    except Exception as e:
        pytest.fail(f"Failed to reconstruct Beats object from loaded data: {e}")


def test_generate_video_from_loaded_beats(detected_raw_beats_file, tmp_path):
    """
    Test generating a video using a Beats object reconstructed from a saved file.
    Relies on the detected_raw_beats_file fixture and uses a test-specific temp dir for video output.
    """
    beats_file_path, _, tolerance_percent, min_measures = detected_raw_beats_file
    output_video_file = tmp_path / f"{TEST_AUDIO_FILE.stem}_test_output.mp4"

    # --- Load and Reconstruct (Steps required for video generation) ---
    try:
        loaded_raw_beats = load_raw_beats(str(beats_file_path))
        reconstructed_beats = Beats.from_timestamps(
            timestamps=loaded_raw_beats.timestamps,
            beat_counts=loaded_raw_beats.beat_counts,
            beats_per_bar=loaded_raw_beats.beats_per_bar,
            tolerance_percent=tolerance_percent,
            min_measures=min_measures
        )
        assert reconstructed_beats is not None
    except Exception as e:
        pytest.fail(f"Setup failed: Could not load/reconstruct Beats from {beats_file_path}: {e}")

    # --- Video Generation ---
    try:
        video_generator = BeatVideoGenerator()
        output_path_str = video_generator.generate_video(
            audio_path=TEST_AUDIO_FILE,
            beats=reconstructed_beats,
            output_path=output_video_file, # Save to test-specific temp dir
        )
        # Ensure the returned path is the same Path object we passed in
        assert Path(output_path_str) == output_video_file, "Output path mismatch"
    except Exception as e:
        pytest.fail(f"Video generation failed: {e}")

    # --- Post-check ---
    assert output_video_file.is_file(), f"Output video file was not created at {output_video_file}"
    assert output_video_file.stat().st_size > 0, f"Output video file {output_video_file} is empty."

    print(
        f"Successfully generated video: {output_video_file} (Size: {output_video_file.stat().st_size} bytes)"
    )


# Keep the main block for potentially running tests directly (though pytest is preferred)
if __name__ == "__main__":
     # Note: Running this directly won't use pytest fixtures correctly.
     # Use `pytest tests/test_process_audio_file.py` instead.
     print("Please run these tests using pytest.")
     # Attempting to run might fail due to fixture dependencies.
     # Consider removing this block or adding guards if direct execution is needed.
     pass
