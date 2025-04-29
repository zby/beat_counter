import pytest
import os  # Import os for file operations
from pathlib import Path
from beat_detection.core.detector import BeatDetector
from beat_detection.core.beats import BeatCalculationError, Beats, RawBeats
from beat_detection.core.video import BeatVideoGenerator  # Import video generator
from beat_detection.utils.beat_file import load_raw_beats # Import the new loading function
import numpy as np

# Define the path to the test audio file relative to the tests directory
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_AUDIO_FILE = TEST_DATA_DIR / "Besito_a_Besito_10sec.mp3"
OUTPUT_VIDEO_FILE = TEST_DATA_DIR / "Besito_a_Besito_10sec.mp4"  # Define output path
OUTPUT_BEATS_FILE = (
    TEST_DATA_DIR / f"{TEST_AUDIO_FILE.stem}.beats.json"
)  # Define beats output path


def test_generate_video_from_beats():
    """
    Test generating a beat visualization video from detected beats,
    including saving/loading the Beats object.
    Ensures the output file is created and is not empty.
    """
    # --- Pre-check and Cleanup ---
    files_to_remove = [OUTPUT_VIDEO_FILE, OUTPUT_BEATS_FILE]
    for file_path in files_to_remove:
        if file_path.exists():
            try:
                os.remove(file_path)
                print(f"Removed existing output file: {file_path}")
            except OSError as e:
                pytest.fail(f"Failed to remove existing output file {file_path}: {e}")

    # Ensure the test audio file exists
    if not TEST_AUDIO_FILE.is_file():
        pytest.fail(f"Test audio file not found: {TEST_AUDIO_FILE}")

    # --- Beat Detection ---
    raw_beats: RawBeats = None
    # effective_beats_per_bar: int = None # No longer need this from detector
    # Define the parameters used for reconstruction in this test
    tolerance_percent: float = 10.0
    min_measures: int = 5
    try:
        # Remove tol/meas from detector instantiation
        detector = BeatDetector(beats_per_bar=None) # Pass None for auto-detect as before
        # Get the simplified RawBeats (contains bpb)
        raw_beats = detector.detect_beats(str(TEST_AUDIO_FILE))
        assert raw_beats is not None, "RawBeats object was not created."
        assert raw_beats.timestamps.shape[0] > 0, "No raw beats were detected."
        assert raw_beats.beats_per_bar > 0, "Invalid beats_per_bar in RawBeats."
        print(f"Detected raw beats with bpb={raw_beats.beats_per_bar}")

        # --- Save Raw Beats to File ---
        try:
            raw_beats.save_to_file(OUTPUT_BEATS_FILE)
            print(f"Saved raw beats to: {OUTPUT_BEATS_FILE}")
            assert OUTPUT_BEATS_FILE.is_file(), f"Raw beats file was not created at {OUTPUT_BEATS_FILE}"
            assert OUTPUT_BEATS_FILE.stat().st_size > 0, f"Raw beats file {OUTPUT_BEATS_FILE} is empty."
        except Exception as e:
            pytest.fail(f"Failed to save RawBeats object to {OUTPUT_BEATS_FILE}: {e}")

        # --- Reconstruct Beats object (using data from RawBeats and test params) --- 
        try:
            reconstructed_beats = Beats.from_timestamps(
                timestamps=raw_beats.timestamps,
                beat_counts=raw_beats.beat_counts,
                beats_per_bar=raw_beats.beats_per_bar, # From RawBeats
                tolerance_percent=tolerance_percent,   # Restore test param
                min_measures=min_measures          # Restore test param
            )
            assert reconstructed_beats is not None, "Failed to reconstruct Beats object."
            assert reconstructed_beats.beats_per_bar == raw_beats.beats_per_bar
            print(f"Reconstructed Beats object with bpb={reconstructed_beats.beats_per_bar}")
        except Exception as e:
            pytest.fail(f"Failed to reconstruct Beats object: {e}")

    except BeatCalculationError as e:
        pytest.fail(f"Beat detection failed for {TEST_AUDIO_FILE.name} with error: {e}")
    except Exception as e:
        pytest.fail(
            f"An unexpected error occurred during beat detection for {TEST_AUDIO_FILE.name}: {e}"
        )

    # --- Load Raw Beats from File (Refactoring Step 5) ---
    loaded_raw_beats: RawBeats = None
    try:
        loaded_raw_beats = load_raw_beats(str(OUTPUT_BEATS_FILE))
        print(f"Loaded raw beats data from: {OUTPUT_BEATS_FILE}")
        assert loaded_raw_beats is not None, "Failed to load raw beats from file."
        # Compare loaded RawBeats with the one saved earlier
        np.testing.assert_array_equal(loaded_raw_beats.timestamps, raw_beats.timestamps)
        np.testing.assert_array_equal(loaded_raw_beats.beat_counts, raw_beats.beat_counts)
        assert loaded_raw_beats.beats_per_bar == raw_beats.beats_per_bar

        # --- Reconstruct Beats from *loaded* data --- 
        reconstructed_beats_from_load = Beats.from_timestamps(
            timestamps=loaded_raw_beats.timestamps,
            beat_counts=loaded_raw_beats.beat_counts,
            beats_per_bar=loaded_raw_beats.beats_per_bar, # Use value from loaded RawBeats
            # Use the same tolerance/min_measures defined earlier in the test
            tolerance_percent=tolerance_percent, # Restore test param
            min_measures=min_measures          # Restore test param
        )
        assert reconstructed_beats_from_load is not None, "Failed to reconstruct Beats object from loaded data."
        print(f"Reconstructed Beats object from loaded data with bpb={reconstructed_beats_from_load.beats_per_bar}")

    except FileNotFoundError:
        pytest.fail(f"Raw beats file not found for loading: {OUTPUT_BEATS_FILE}")
    except Exception as e:
        pytest.fail(f"Failed to load/reconstruct from RawBeats file {OUTPUT_BEATS_FILE}: {e}")

    # --- Video Generation --- 
    # Video generator needs a full Beats object. Use the one reconstructed from loaded data.
    try:
        video_generator = BeatVideoGenerator()
        output_path_str = video_generator.generate_video(
            audio_path=TEST_AUDIO_FILE,
            beats=reconstructed_beats_from_load, # Use the object reconstructed from file
            output_path=OUTPUT_VIDEO_FILE,
        )
        assert Path(output_path_str) == OUTPUT_VIDEO_FILE, f"Output path mismatch"
    except Exception as e:
        pytest.fail(f"Video generation failed: {e}")

    # --- Post-check ---
    assert OUTPUT_VIDEO_FILE.is_file(), f"Output video file was not created at {OUTPUT_VIDEO_FILE}"
    assert OUTPUT_VIDEO_FILE.stat().st_size > 0, f"Output video file {OUTPUT_VIDEO_FILE} is empty."

    print(
        f"Successfully generated video: {OUTPUT_VIDEO_FILE} (Size: {OUTPUT_VIDEO_FILE.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    test_generate_video_from_beats()
