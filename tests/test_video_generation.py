import pytest
import numpy as np
from pathlib import Path

from beat_counter.core.beats import Beats, RawBeats, BeatCalculationError
from beat_counter.core.video import BeatVideoGenerator, generate_single_video_from_files
from beat_counter.cli.generate_video import main as generate_video_main

# Define paths relative to the tests directory, using the fixtures subdir
TEST_FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_AUDIO_FILE = TEST_FIXTURES_DIR / "Besito_a_Besito_10sec.mp3"
# Use the beats file from the fixtures directory
INPUT_BEATS_FILE = TEST_FIXTURES_DIR / "Besito_a_Besito_10sec.beats"

# Define parameters used for Beats reconstruction
TEST_TOLERANCE_PERCENT = 10.0
TEST_MIN_MEASURES = 4

@pytest.fixture(scope="module")
def raw_beats_fixture() -> RawBeats:
    loaded_raw_beats = RawBeats.load_from_file(INPUT_BEATS_FILE)
    return loaded_raw_beats

def test_generate_video_from_saved_beats(tmp_path):
    """
    Test generating a video using a Beats object reconstructed from a pre-existing saved file.
    Uses a test-specific temp dir for video output.
    """
    # Define output path within the temporary directory provided by pytest
    output_video_file = tmp_path / f"{TEST_AUDIO_FILE.stem}_video_test_output.mp4"

    # Ensure input files exist
    if not TEST_AUDIO_FILE.is_file():
        pytest.fail(f"Test audio file not found: {TEST_AUDIO_FILE}")
    if not INPUT_BEATS_FILE.is_file():
        pytest.fail(f"Input beats file not found: {INPUT_BEATS_FILE}")

    # --- Video Generation ---
    try:
        # Use the high-level function that works directly with files
        output_video_path = generate_single_video_from_files(
            audio_file=TEST_AUDIO_FILE,
            beats_file=INPUT_BEATS_FILE,
            output_file=output_video_file,
            tolerance_percent=TEST_TOLERANCE_PERCENT,
            min_measures=TEST_MIN_MEASURES,
            verbose=True
        )
        # Ensure the returned path is the same Path object we passed in
        assert output_video_path == output_video_file, "Output path mismatch"

        # Print the path immediately after successful generation
        print(
            f"\n[Test Info] Successfully generated video: {output_video_file} (Size: {output_video_file.stat().st_size} bytes)"
        )

    except Exception as e:
        pytest.fail(f"Video generation failed: {e}")

    # --- Post-check ---
    assert (
        output_video_file.is_file()
    ), f"Output video file was not created at {output_video_file}"
    assert (
        output_video_file.stat().st_size > 0
    ), f"Output video file {output_video_file} is empty."

# Keep the main block for potentially running tests directly (though pytest is preferred)
if __name__ == "__main__":
     # Note: Running this directly won't use pytest fixtures correctly.
     # Use `pytest tests/test_video_generation.py` instead.
     print("Please run this test using pytest.")
     pass # No direct execution logic needed 