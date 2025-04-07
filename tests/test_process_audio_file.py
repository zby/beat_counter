import pytest
import os # Import os for file operations
from pathlib import Path
from beat_detection.core.detector import BeatDetector
from beat_detection.core.beats import BeatCalculationError
from beat_detection.core.video import BeatVideoGenerator # Import video generator
from beat_detection.core.beats import Beats # Import Beats for loading

# Define the path to the test audio file relative to the tests directory
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_AUDIO_FILE = TEST_DATA_DIR / "Besito_a_Besito_10sec.mp3"
OUTPUT_VIDEO_FILE = TEST_DATA_DIR / "Besito_a_Besito_10sec.mp4" # Define output path
OUTPUT_BEATS_FILE = TEST_DATA_DIR / f"{TEST_AUDIO_FILE.stem}.beats.json" # Define beats output path

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
    try:
        detector = BeatDetector()
        beats = detector.detect_beats(str(TEST_AUDIO_FILE))
        assert len(beats.timestamps) > 0, "No beats were detected."
        assert beats.meter is not None, "Meter was not determined."
        assert len(beats.downbeat_indices) > 0, \
            f"No downbeats were detected in {TEST_AUDIO_FILE.name}"

        # --- Save Beats to File ---
        try:
            beats.save_to_file(OUTPUT_BEATS_FILE)
            print(f"Saved beats to: {OUTPUT_BEATS_FILE}")
            assert OUTPUT_BEATS_FILE.is_file(), f"Beats file was not created at {OUTPUT_BEATS_FILE}"
            assert OUTPUT_BEATS_FILE.stat().st_size > 0, f"Beats file {OUTPUT_BEATS_FILE} is empty."
        except Exception as e:
            pytest.fail(f"Failed to save Beats object to {OUTPUT_BEATS_FILE}: {e}")

    except BeatCalculationError as e:
        pytest.fail(f"Beat detection failed for {TEST_AUDIO_FILE.name} with error: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during beat detection for {TEST_AUDIO_FILE.name}: {e}")

    # --- Load Beats from File ---
    try:
        loaded_beats = Beats.load_from_file(OUTPUT_BEATS_FILE)
        print(f"Loaded beats from: {OUTPUT_BEATS_FILE}")
        assert isinstance(loaded_beats, Beats), "Loaded object is not a Beats instance."
        # Optional: Add more specific assertions comparing loaded_beats to original beats if needed
        assert loaded_beats.meter == beats.meter, "Loaded beats meter mismatch."
        assert len(loaded_beats.timestamps) == len(beats.timestamps), "Loaded beats timestamp count mismatch."
    except FileNotFoundError:
        pytest.fail(f"Beats file not found for loading: {OUTPUT_BEATS_FILE}")
    except Exception as e:
        pytest.fail(f"Failed to load Beats object from {OUTPUT_BEATS_FILE}: {e}")

    # --- Video Generation ---
    try:
        video_generator = BeatVideoGenerator() # Use default settings
        
        # Call the method that takes a Beats object and audio path, using the loaded object
        output_path_str = video_generator.generate_video(
            audio_path=TEST_AUDIO_FILE,
            beats=loaded_beats, # Use the loaded beats object
            output_path=OUTPUT_VIDEO_FILE
        )
        
        # Verify output path matches expected (optional, but good practice)
        assert Path(output_path_str) == OUTPUT_VIDEO_FILE, \
               f"Generated video path '{output_path_str}' does not match expected '{OUTPUT_VIDEO_FILE}'"

    except Exception as e:
        pytest.fail(f"Video generation failed for {TEST_AUDIO_FILE.name} with error: {e}")

    # --- Post-check ---
    assert OUTPUT_VIDEO_FILE.is_file(), f"Output video file was not created at {OUTPUT_VIDEO_FILE}"
    assert OUTPUT_VIDEO_FILE.stat().st_size > 0, f"Output video file {OUTPUT_VIDEO_FILE} is empty."
    
    print(f"Successfully generated video: {OUTPUT_VIDEO_FILE} (Size: {OUTPUT_VIDEO_FILE.stat().st_size} bytes)")


if __name__ == "__main__":
    test_generate_video_from_beats()