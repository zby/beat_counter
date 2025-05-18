import subprocess
import sys
import pytest
import json
import pathlib
import shutil
from typing import List

# --- Test Configuration ---

# Define the path to the test audio file relative to the tests directory
TEST_DATA_DIR = pathlib.Path(__file__).parent / "fixtures"
SAMPLE_AUDIO_NAME = "Besito_a_Besito_10sec.mp3"
SAMPLE_AUDIO_PATH = TEST_DATA_DIR / SAMPLE_AUDIO_NAME

# Define the persistent output directory within tests/
CLI_TEST_OUTPUT_DIR = pathlib.Path(__file__).parent / "output" / "cli_tests"

# Check if the sample audio file exists before running tests
if not SAMPLE_AUDIO_PATH.is_file():
    pytest.fail(
        f"Sample audio file not found at expected location: {SAMPLE_AUDIO_PATH}\n"
        f"Please ensure the file exists or adjust SAMPLE_AUDIO_PATH in test_cli.py",
        pytrace=False,
    )


# --- Fixtures ---


@pytest.fixture(scope="module")
def cli_test_setup() -> pathlib.Path:
    """Set up the test environment for CLI tests. Returns path to the audio file in test dir."""
    # Set up test output dir
    CLI_TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy a test audio file to the output directory for testing
    audio_file_dest = CLI_TEST_OUTPUT_DIR / SAMPLE_AUDIO_NAME

    # Copy only if it doesn't exist or is different (prevents constant copying)
    if not audio_file_dest.is_file() or audio_file_dest.stat().st_size != SAMPLE_AUDIO_PATH.stat().st_size:
        try:
            # Use binary read/write for cross-platform safety
            with open(SAMPLE_AUDIO_PATH, 'rb') as src_file, open(audio_file_dest, 'wb') as dest_file:
                dest_file.write(src_file.read())
            print(f"Copied test audio file to {audio_file_dest}")
        except Exception as e:
            pytest.fail(f"Failed to copy test audio file: {e}")

    return audio_file_dest


# --- Helper Functions ---


def run_cli(command: List[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command-line interface command and capture output.
    
    Params:
    -------
    command : List[str]
        The command to run, e.g. ["detect-beats", "path/to/audio.mp3"]
    **kwargs : Any
        Additional keyword arguments to pass to subprocess.run
    
    Returns:
    --------
    subprocess.CompletedProcess
        The result of the command execution
    """
    # Capture stdout and stderr by default
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)  # Return strings not bytes
    
    print(f"\nRunning CLI: {' '.join(command)}")
    result = subprocess.run(command, **kwargs)
    print(f"Exit Code: {result.returncode}")
    
    if result.stdout and result.stdout.strip():
        print(f"Stdout:\n{result.stdout.strip()}")
    if result.stderr and result.stderr.strip():
        print(f"Stderr:\n{result.stderr.strip()}")
    
    return result


# --- Test Cases ---


# 1. detect-beats (Single File)

def test_detect_beats_default_output(cli_test_setup):
    """Test `detect-beats` with default output path (derived from input)."""
    audio_file = cli_test_setup
    expected_beats_file = audio_file.with_suffix(".beats")
    
    # Clean up first if file exists
    expected_beats_file.unlink(missing_ok=True)
    
    # Run command with defaults
    result = run_cli(["detect-beats", str(audio_file)])
    
    assert result.returncode == 0
    assert expected_beats_file.is_file()
    assert expected_beats_file.stat().st_size > 0
    
    # Verify the beats file format
    try:
        with expected_beats_file.open("r") as f:
            beats_data = json.load(f)
        assert isinstance(beats_data, dict)
        # Check for RawBeats format
        assert "timestamps" in beats_data
        assert "beat_counts" in beats_data
        assert isinstance(beats_data["timestamps"], list)
        assert isinstance(beats_data["beat_counts"], list)
        assert len(beats_data["timestamps"]) == len(beats_data["beat_counts"])
        assert len(beats_data["timestamps"]) > 0, "No beats detected"
    except json.JSONDecodeError:
        pytest.fail(f"Default output beats file was not valid JSON: {expected_beats_file}")

def test_detect_beats_audio_not_found(cli_test_setup):
    """Test `detect-beats` with a non-existent audio file."""
    # Use a path relative to the test output dir, but ensure it doesn't exist
    non_existent_file = cli_test_setup.parent / "non_existent_audio.mp3"
    non_existent_file.unlink(missing_ok=True)

    result = run_cli(["detect-beats", str(non_existent_file)])
    assert result.returncode != 0
    assert "Audio file not found" in result.stderr


# 2. detect-beats-batch

def test_detect_beats_batch_success(cli_test_setup):
    """Test `detect-beats-batch` processing the test directory (using madmom)."""
    audio_file = cli_test_setup
    input_dir = cli_test_setup.parent # The directory containing the fixture audio
    expected_beats_file = audio_file.with_suffix(".beats")

    # Ensure output file doesn't exist beforehand (it might from previous test)
    expected_beats_file.unlink(missing_ok=True)

    # Run command (use madmom)
    result = run_cli(["detect-beats-batch", str(input_dir), "--detector-name", "madmom", "--min-measures", "1"])

    assert result.returncode == 0
    # Allow stderr for logging, but check main message
    # assert not result.stderr # Might have logging info

    expected_beats_file = audio_file.with_suffix(".beats")
    assert expected_beats_file.is_file()
    assert expected_beats_file.stat().st_size > 0

    # Verify beats file content
    try:
        with expected_beats_file.open("r") as f:
            beats_data = json.load(f)
        assert isinstance(beats_data, dict)
        # Check for simplified RawBeats format
        assert "timestamps" in beats_data
        assert "beat_counts" in beats_data
        assert isinstance(beats_data["timestamps"], list)
        assert isinstance(beats_data["beat_counts"], list)
        assert len(beats_data["timestamps"]) == len(beats_data["beat_counts"])
        # beats_per_bar is no longer part of RawBeats
        assert len(beats_data["timestamps"]) > 0, "No beats detected"
    except json.JSONDecodeError:
        pytest.fail(f"Batch output beats file not valid JSON: {expected_beats_file}")


# 3. generate-video (Single File)

@pytest.fixture
def beats_file_for_video(cli_test_setup) -> pathlib.Path:
    """Fixture to ensure the necessary .beats file exists before video tests."""
    audio_file = cli_test_setup # Get path from main fixture
    beats_file = audio_file.with_suffix(".beats")

    # Run detect-beats if beats file doesn't exist from a previous test
    # or if we want to ensure it's fresh (though fixture scope might handle this)
    # For simplicity, let's ensure it's created here.
    beats_file.unlink(missing_ok=True)
    # Use min_measures=1 for testing with short audio samples
    result = run_cli(["detect-beats", str(audio_file), "--detector-name", "madmom", "--min-measures", "1", "-o", str(beats_file)])
    assert result.returncode == 0, f"Failed to create beats file for video test: {result.stderr}"
    assert beats_file.is_file()
    return beats_file

def test_generate_video_default_output(cli_test_setup, beats_file_for_video):
    """Test `generate-video` with default output path."""
    audio_file = cli_test_setup # Path from main fixture
    # beats_file_for_video fixture ensures .beats file exists

    expected_video_file = audio_file.with_name(f"{audio_file.stem}_counter.mp4")
    expected_video_file.unlink(missing_ok=True)

    # Add required arguments for reconstruction
    result = run_cli([
        "generate-video",
        str(audio_file),
        "--tolerance-percent", "10.0", # Provide default or specific value
        "--min-measures", "5"          # Provide default or specific value
    ])

    assert result.returncode == 0
    # Expect moviepy log messages on stderr
    # assert not result.stderr

    expected_video_file = audio_file.with_name(f"{audio_file.stem}_counter.mp4")
    assert expected_video_file.is_file()
    assert expected_video_file.stat().st_size > 1000 # Check it's not empty


def test_generate_video_missing_beats_file(cli_test_setup):
    """Test `generate-video` when the .beats file is missing."""
    audio_file = cli_test_setup
    beats_file = audio_file.with_suffix(".beats")
    # Ensure beats file does not exist
    beats_file.unlink(missing_ok=True)

    # Add required arguments even if beats file is missing (CLI parsing happens first)
    result = run_cli([
        "generate-video",
        str(audio_file),
        "--tolerance-percent", "10.0",
        "--min-measures", "5"
    ])
    assert result.returncode != 0
    # Error message might change slightly, ensure it indicates missing file
    assert "Beats file not found" in result.stderr


# 4. generate-video-batch

# Note: Batch tests now rely on the state within CLI_TEST_OUTPUT_DIR setup by cli_test_setup

def test_generate_video_batch_default_output(cli_test_setup):
    """Test `generate-video-batch` processing the test directory with default output."""
    input_dir = cli_test_setup.parent # Dir containing the fixture audio
    audio_file_path = cli_test_setup # The audio file path itself
    beats_file = audio_file_path.with_suffix(".beats")

    # Ensure beats file exists first for this test (run detect-beats)
    if not beats_file.is_file(): # pragma: no cover (should exist from previous tests/setup)
        # Run detect-beats if needed, although fixture should handle this? Re-run for safety.
        result_beats = run_cli(["detect-beats", str(audio_file_path), "--detector-name", "madmom", "--min-measures", "1", "-o", str(beats_file)])
        assert result_beats.returncode == 0, f"Setup failed: Could not generate beats file: {result_beats.stderr}"

    # Define and ensure expected output file doesn't exist beforehand
    expected_video_file = audio_file_path.with_name(f"{audio_file_path.stem}_counter.mp4")
    expected_video_file.unlink(missing_ok=True)

    result = run_cli([
        "generate-video-batch",
        str(input_dir),
        "--min-measures", "1",  # Use min_measures=1 for testing with short audio samples
    ])

    assert result.returncode == 0

    expected_video_file = audio_file_path.with_name(f"{audio_file_path.stem}_counter.mp4")
    assert expected_video_file.is_file()
    assert expected_video_file.stat().st_size > 1000

def test_generate_video_batch_explicit_output(cli_test_setup):
    """Test `generate-video-batch` with an explicit output directory."""
    input_dir = cli_test_setup.parent
    audio_file_path = input_dir / SAMPLE_AUDIO_NAME
    beats_file = audio_file_path.with_suffix(".beats")
    # Define output dir relative to the main test output dir
    output_dir = input_dir / "video_batch_output"
    # output_dir.mkdir(exist_ok=True) # Let the script create it

    # Ensure beats file exists first
    if not beats_file.is_file(): # pragma: no cover
        result_beats = run_cli(["detect-beats", str(audio_file_path), "--detector-name", "madmom", "--min-measures", "1", "-o", str(beats_file)])
        assert result_beats.returncode == 0, f"Setup failed: Could not generate beats file: {result_beats.stderr}"

    # Define and ensure expected output file doesn't exist beforehand
    expected_video_file = output_dir / f"{SAMPLE_AUDIO_PATH.stem}_counter.mp4"
    expected_video_file.unlink(missing_ok=True)

    result = run_cli([
        "generate-video-batch",
        str(input_dir),
        "--min-measures", "1",  # Use min_measures=1 for testing with short audio samples
        "-o",
        str(output_dir),
    ])

    assert result.returncode == 0

    expected_video_file = output_dir / f"{SAMPLE_AUDIO_PATH.stem}_counter.mp4"
    assert output_dir.is_dir()
    assert expected_video_file.is_file()
    assert expected_video_file.stat().st_size > 1000

def test_generate_video_batch_missing_beats(cli_test_setup):
    """Test `generate-video-batch` when a beats file is missing (should skip)."""
    input_dir = cli_test_setup.parent
    audio_file = cli_test_setup # Path to the audio file in the dir
    beats_file = audio_file.with_suffix(".beats")
    # Define expected video path (default location)
    video_file = audio_file.with_name(f"{audio_file.stem}_counter.mp4")

    # Ensure the corresponding beats file *does not* exist for this specific test
    beats_file.unlink(missing_ok=True)
    # Ensure video file doesn't exist from previous run
    video_file.unlink(missing_ok=True)

    result = run_cli(["generate-video-batch", str(input_dir)])
    # The batch script should skip the file and report success (exit 0), but log a warning
    assert result.returncode == 0
    # Some indication the file was skipped should be in stdout or stderr
    assert "skipped" in result.stdout.lower() or "skipped" in result.stderr.lower() or \
           "missing" in result.stdout.lower() or "missing" in result.stderr.lower()
    assert not video_file.is_file(), "Expected video file to not be created"

def test_generate_video_batch_input_dir_not_found(cli_test_setup):
    """Test `generate-video-batch` with a non-existent input directory."""
    # Use a path that doesn't exist
    non_existent_dir = cli_test_setup.parent / "non_existent_dir"
    try:
        # Ensure it doesn't exist
        if non_existent_dir.is_dir():
            non_existent_dir.rmdir()
    
        result = run_cli(["generate-video-batch", str(non_existent_dir)])
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "does not exist" in result.stderr.lower()
    finally:
        # Clean up in case test somehow creates it
        if non_existent_dir.is_dir():
            non_existent_dir.rmdir()

def test_generate_video_batch_no_audio_files_found(cli_test_setup):
    """Test `generate-video-batch` with a directory containing no audio files."""
    # Create a temporary directory for this test
    empty_dir = cli_test_setup.parent / "empty_dir_for_batch_test"
    empty_dir.mkdir(exist_ok=True)
    
    try:
        # Remove any audio files (just to be sure)
        for f in empty_dir.glob("*.mp3"):
            f.unlink()
        for f in empty_dir.glob("*.wav"):
            f.unlink()
        
        result = run_cli(["generate-video-batch", str(empty_dir)])
        # Should succeed with a warning
        assert result.returncode == 0
        assert "no audio files" in result.stdout.lower() or "no audio files" in result.stderr.lower()
    finally:
        # Clean up
        if empty_dir.is_dir():
            # Try to remove files first
            for f in empty_dir.glob("*"):
                try:
                    f.unlink()
                except:
                    pass
            try:
                empty_dir.rmdir()
            except:
                pass # Non-critical cleanup