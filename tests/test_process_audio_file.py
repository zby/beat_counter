import pytest
import os # Import os for file operations
from pathlib import Path
from beat_detection.core.detector import BeatDetector
from beat_detection.core.beats import BeatCalculationError
from beat_detection.core.video import BeatVideoGenerator # Import video generator

# Define the path to the test audio file relative to the tests directory
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_AUDIO_FILE = TEST_DATA_DIR / "Besito_a_Besito_10sec.mp3"
OUTPUT_VIDEO_FILE = TEST_DATA_DIR / "Besito_a_Besito_10sec.mp4" # Define output path

def test_generate_video_from_beats():
    """
    Test generating a beat visualization video from detected beats.
    Ensures the output file is created and is not empty.
    """
    # --- Pre-check and Cleanup ---
    if OUTPUT_VIDEO_FILE.exists():
        try:
            os.remove(OUTPUT_VIDEO_FILE)
            print(f"Removed existing output file: {OUTPUT_VIDEO_FILE}")
        except OSError as e:
            pytest.fail(f"Failed to remove existing output file {OUTPUT_VIDEO_FILE}: {e}")

    # Ensure the test audio file exists
    if not TEST_AUDIO_FILE.is_file():
        pytest.fail(f"Test audio file not found: {TEST_AUDIO_FILE}")

    # --- Beat Detection ---
    try:
        detector = BeatDetector()
        beats = detector.detect_beats(str(TEST_AUDIO_FILE))
        assert len(beats.timestamps) > 0, "No beats were detected."
        assert beats.meter is not None, "Meter was not determined."
        # Assert that the downbeat_indices list is not empty
        assert len(beats.downbeat_indices) > 0, \
            f"No downbeats were detected in {TEST_AUDIO_FILE.name}"


    except BeatCalculationError as e:
        pytest.fail(f"Beat detection failed for {TEST_AUDIO_FILE.name} with error: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during beat detection for {TEST_AUDIO_FILE.name}: {e}")

    # --- Video Generation ---
    try:
        video_generator = BeatVideoGenerator() # Use default settings
        
        # Call the method that takes a Beats object and audio path
        output_path_str = video_generator.generate_video(
            audio_path=TEST_AUDIO_FILE,
            beats=beats, 
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