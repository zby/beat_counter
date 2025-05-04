import pytest
import numpy as np
from pathlib import Path

from beat_detection.core.beats import Beats, RawBeats, BeatCalculationError
from beat_detection.core.video import BeatVideoGenerator
from beat_detection.cli.generate_video import main as generate_video_main

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

    # --- Load and Reconstruct (Steps required for video generation) ---
    reconstructed_beats: Beats = None
    try:
        # Load the simplified RawBeats data
        loaded_raw_beats = RawBeats.load_from_file(INPUT_BEATS_FILE)
        assert loaded_raw_beats is not None, "Failed to load RawBeats."
        assert len(loaded_raw_beats.timestamps) > 0, "Loaded RawBeats has no timestamps."

        # Construct the full Beats object, passing None for beats_per_bar to trigger inference
        reconstructed_beats = Beats(
            raw_beats=loaded_raw_beats,
            beats_per_bar=None, # Let the constructor infer from raw_beats.beat_counts
            tolerance_percent=TEST_TOLERANCE_PERCENT,
            min_measures=TEST_MIN_MEASURES,
        )
        assert reconstructed_beats is not None, "Failed to reconstruct Beats."
        # Optional: Verify the inferred value if known/expected for the fixture file
        # assert reconstructed_beats.beats_per_bar == 4, "Inferred beats_per_bar mismatch"

    except BeatCalculationError as e:
         pytest.fail(f"Beats reconstruction failed: {e}")
    except FileNotFoundError:
         pytest.fail(f"Could not find input file: {INPUT_BEATS_FILE}")
    except Exception as e:
        pytest.fail(
            f"Setup failed: Could not load/reconstruct Beats from {INPUT_BEATS_FILE}: {e}"
        )

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