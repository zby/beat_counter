import sys
import pytest
import json
import pathlib
import shutil
import subprocess
from typing import List

# --- Test Configuration ---

# Define the path to the test audio file relative to the tests directory
TEST_DATA_DIR = pathlib.Path(__file__).parent / "data"
SAMPLE_AUDIO_NAME = "Besito_a_Besito_10sec.mp3"
SAMPLE_AUDIO_PATH = TEST_DATA_DIR / SAMPLE_AUDIO_NAME
CLI_TEST_OUTPUT_DIR = TEST_DATA_DIR / "cli_tests_beat_this"  # Separate directory for beat_this tests

# Check if the sample audio file exists before running tests
if not SAMPLE_AUDIO_PATH.is_file():
    pytest.fail(
        f"Sample audio file not found at expected location: {SAMPLE_AUDIO_PATH}\n"
        f"Please ensure the file exists or adjust SAMPLE_AUDIO_PATH in test_cli_beat_this.py",
        pytrace=False,
    )


# --- Fixtures ---


@pytest.fixture(scope="module")
def cli_test_setup() -> pathlib.Path:
    """Sets up the tests/data/cli_tests_beat_this directory for CLI testing.

    - Cleans and recreates the directory.
    - Copies the sample audio into it.
    - Yields the path to the copied audio file.
    - Leaves the directory contents for inspection after tests.
    """
    # Clean up before test run
    if CLI_TEST_OUTPUT_DIR.exists():
        shutil.rmtree(CLI_TEST_OUTPUT_DIR)
    CLI_TEST_OUTPUT_DIR.mkdir(parents=True)

    # Copy sample audio
    dest_path = CLI_TEST_OUTPUT_DIR / SAMPLE_AUDIO_NAME
    shutil.copy(SAMPLE_AUDIO_PATH, dest_path)
    yield dest_path
    # No cleanup - leave files for inspection


@pytest.fixture
def beats_file_for_video(cli_test_setup) -> pathlib.Path:
    """Fixture to generate the necessary .beats file before video tests."""
    audio_file = cli_test_setup
    beats_file = audio_file.with_suffix(".beats")
    # Run detect-beats to create the file
    # Ensure it doesn't exist from a previous test in the module
    beats_file.unlink(missing_ok=True)
    # Use min_measures=1 for testing with short audio samples
    result = run_cli(["detect-beats", str(audio_file), "--algorithm", "beat_this", "--min-measures", "1"])
    assert result.returncode == 0, f"Failed to create beats file for video test: {result.stderr}"
    assert beats_file.is_file()
    return beats_file


# --- Helper Functions ---


def run_cli(command: List[str], **kwargs) -> subprocess.CompletedProcess:
    """Helper to run CLI commands using the current Python executable."""
    # Prepend sys.executable to ensure running with the correct environment
    # This avoids issues if the virtual env isn't activated in the shell running pytest
    # We directly call the registered script names, assuming `uv pip install .` was run.
    # If scripts aren't found, prepend [sys.executable, "-m", "beat_detection.cli.<module_name>"]
    # For now, assume scripts like 'detect-beats' are on PATH
    print(f"\nRunning CLI: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=False, **kwargs)
    print(f"Exit Code: {result.returncode}")
    
    if result.stdout:
        print("Stdout:")
        print(result.stdout)
    
    if result.stderr:
        print("Stderr:")
        print(result.stderr)
    
    return result


# --- Test Cases ---


# 1. detect-beats (Single File)


def test_detect_beats_default_output(cli_test_setup):
    """Test `detect-beats` writing beats file next to audio by default."""
    audio_file = cli_test_setup
    expected_beats_file = audio_file.with_suffix(".beats")

    # Ensure output file doesn't exist beforehand
    expected_beats_file.unlink(missing_ok=True)
    
    try:
        # Use min_measures=1 for testing with short audio samples
        result = run_cli(["detect-beats", str(audio_file), "--algorithm", "beat_this", "--min-measures", "1"]) # No -o specified

        assert result.returncode == 0
        assert expected_beats_file.is_file(), f"Expected beats file {expected_beats_file} was not created"
        assert expected_beats_file.stat().st_size > 0

        # Verify file content
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
            assert len(beats_data["timestamps"]) > 0, "No beats detected"
        except json.JSONDecodeError:
            pytest.fail(f"Default output beats file was not valid JSON: {expected_beats_file}")
    except subprocess.CalledProcessError as e:
        # Skip beat_this test if the algorithm is not available or has compatibility issues
        if "No module named 'beat_this'" in str(e.stderr) or "Weights only load failed" in str(e.stderr):
            pytest.skip("Skipping beat_this test due to compatibility issues")
        raise


def test_detect_beats_explicit_output(cli_test_setup):
    """Test `detect-beats` writing beats to a file."""
    audio_file = cli_test_setup
    # Test writing to a different filename
    output_beats_file = cli_test_setup.parent / "explicit_output_beat_this.beats"

    # Ensure output file doesn't exist beforehand
    output_beats_file.unlink(missing_ok=True)
    
    try:
        # Use min_measures=1 for testing with short audio samples
        result = run_cli([
            "detect-beats",
            str(audio_file),
            "--algorithm",
            "beat_this",
            "--min-measures", "1",
            "-o",
            str(output_beats_file),
        ])

        assert result.returncode == 0
        assert output_beats_file.is_file()
        assert output_beats_file.stat().st_size > 0

        # Verify file content
        try:
            with output_beats_file.open("r") as f:
                beats_data = json.load(f)
            assert isinstance(beats_data, dict)
            # Check for simplified RawBeats format
            assert "timestamps" in beats_data
            assert "beat_counts" in beats_data
            assert isinstance(beats_data["timestamps"], list)
            assert isinstance(beats_data["beat_counts"], list)
            assert len(beats_data["timestamps"]) == len(beats_data["beat_counts"])
            assert len(beats_data["timestamps"]) > 0, "No beats detected"
            assert len(beats_data["beat_counts"]) > 0
            # Verify all beat counts are in range 1-4
            for count in beats_data["beat_counts"]:
                assert 1 <= count <= 4, f"Beat count {count} is outside the expected range of 1-4"
            # Verify that all values 1, 2, 3, and 4 appear in the beat counts
            unique_counts = set(beats_data["beat_counts"])
            assert unique_counts.issubset({1, 2, 3, 4}), f"Found unexpected beat counts: {unique_counts}"
            assert len(unique_counts) == 4, f"Not all expected beat counts (1-4) were found: {unique_counts}"
        except json.JSONDecodeError:
            pytest.fail(f"Explicit output beats file not valid JSON: {output_beats_file}")
    except subprocess.CalledProcessError as e:
        # Skip beat_this test if the algorithm is not available or has compatibility issues
        if "No module named 'beat_this'" in str(e.stderr) or "Weights only load failed" in str(e.stderr):
            pytest.skip("Skipping beat_this test due to compatibility issues")
        raise


def test_detect_beats_audio_not_found():
    """Test `detect-beats` with a non-existent audio file."""
    non_existent_file = TEST_DATA_DIR / "does_not_exist.mp3"
    result = run_cli(["detect-beats", str(non_existent_file), "--algorithm", "beat_this"])
    assert result.returncode != 0
    assert "not found" in result.stderr.lower() or "no such file" in result.stderr.lower()


# 2. detect-beats-batch


def test_detect_beats_batch_success(cli_test_setup):
    """Test `detect-beats-batch` processing the test directory."""
    audio_file = cli_test_setup
    input_dir = cli_test_setup.parent # The directory containing the audio

    # Define and ensure expected output file doesn't exist beforehand
    expected_beats_file = audio_file.with_suffix(".beats")
    expected_beats_file.unlink(missing_ok=True)

    try:
        # Use min_measures=1 for testing with short audio samples
        result = run_cli(["detect-beats-batch", str(input_dir), "--algorithm", "beat_this", "--min-measures", "1"])

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
            assert len(beats_data["timestamps"]) > 0, "No beats detected"
        except json.JSONDecodeError:
            pytest.fail(f"Batch output beats file not valid JSON: {expected_beats_file}")
    except subprocess.CalledProcessError as e:
        # Skip beat_this test if the algorithm is not available or has compatibility issues
        if "No module named 'beat_this'" in str(e.stderr) or "Weights only load failed" in str(e.stderr):
            pytest.skip("Skipping beat_this test due to compatibility issues")
        raise


def test_detect_beats_batch_dir_not_found():
    """Test `detect-beats-batch` with a non-existent directory."""
    non_existent_dir = TEST_DATA_DIR / "does_not_exist"
    result = run_cli(["detect-beats-batch", str(non_existent_dir), "--algorithm", "beat_this"])
    assert result.returncode != 0
    assert "not found" in result.stderr.lower() or "no such" in result.stderr.lower()


def test_detect_beats_batch_empty_dir(cli_test_setup):
    """Test `detect-beats-batch` with an empty directory."""
    empty_dir = cli_test_setup.parent / "empty_dir_beat_this"
    if empty_dir.exists():
        shutil.rmtree(empty_dir)
    empty_dir.mkdir()

    result = run_cli(["detect-beats-batch", str(empty_dir), "--algorithm", "beat_this"])
    assert result.returncode != 0
    assert "no audio files" in result.stderr.lower()


# 3. generate-video (Single File)


def test_generate_video_default_output(cli_test_setup, beats_file_for_video):
    """Test `generate-video` with default output path."""
    audio_file = cli_test_setup
    beats_file = beats_file_for_video # Ensure .beats file exists

    # Define and ensure expected output file doesn't exist beforehand
    expected_video_file = audio_file.with_name(f"{audio_file.stem}_counter.mp4")
    expected_video_file.unlink(missing_ok=True)

    try:
        # Add required arguments for reconstruction
        result = run_cli([
            "generate-video",
            str(audio_file),
            "--min-measures", "1",  # Use min_measures=1 for testing with short audio samples
            "--tolerance-percent", "10.0", # Provide default or specific value
        ])

        assert result.returncode == 0
        # Expect moviepy log messages on stderr
        # assert not result.stderr

        expected_video_file = audio_file.with_name(f"{audio_file.stem}_counter.mp4")
        assert expected_video_file.is_file()
    except subprocess.CalledProcessError as e:
        # Skip if there are issues with the beat_this algorithm
        if "No module named 'beat_this'" in str(e.stderr) or "Weights only load failed" in str(e.stderr):
            pytest.skip("Skipping beat_this test due to compatibility issues")
        raise


def test_generate_video_explicit_output(cli_test_setup, beats_file_for_video):
    """Test `generate-video` with an explicit output path."""
    audio_file = cli_test_setup
    beats_file = beats_file_for_video
    
    # Test writing to a specific output file
    output_video_file = cli_test_setup.parent / "explicit_output_beat_this.mp4"
    output_video_file.unlink(missing_ok=True)
    
    try:
        result = run_cli([
            "generate-video",
            str(audio_file),
            "--min-measures", "1",  # Use min_measures=1 for testing with short audio samples
            "-o",
            str(output_video_file),
            "--tolerance-percent", "15.0", # Example: Use non-default
        ])

        assert result.returncode == 0
        assert output_video_file.is_file()
    except subprocess.CalledProcessError as e:
        # Skip if there are issues with the beat_this algorithm
        if "No module named 'beat_this'" in str(e.stderr) or "Weights only load failed" in str(e.stderr):
            pytest.skip("Skipping beat_this test due to compatibility issues")
        raise


def test_generate_video_missing_beats_file(cli_test_setup):
    """Test `generate-video` when the .beats file is missing."""
    audio_file = cli_test_setup
    beats_file = audio_file.with_suffix(".beats")
    
    # Ensure beats file doesn't exist
    beats_file.unlink(missing_ok=True)
    
    result = run_cli([
        "generate-video",
        str(audio_file),
        "--min-measures", "1",
    ])
    
    assert result.returncode != 0
    assert "not found" in result.stderr.lower() or "no such file" in result.stderr.lower()


# 4. generate-video-batch


def test_generate_video_batch_default_output(cli_test_setup):
    """Test `generate-video-batch` processing the test directory."""
    input_dir = cli_test_setup.parent
    audio_file_path = input_dir / SAMPLE_AUDIO_NAME
    beats_file = audio_file_path.with_suffix(".beats")

    # Ensure beats file exists first
    if not beats_file.is_file():
        try:
            # Use min_measures=1 for testing with short audio samples
            result_beats = run_cli(["detect-beats", str(audio_file_path), "--algorithm", "beat_this", "--min-measures", "1"])
            assert result_beats.returncode == 0, f"Setup failed: Could not generate beats file: {result_beats.stderr}"
        except subprocess.CalledProcessError as e:
            # Skip if there are issues with the beat_this algorithm
            if "No module named 'beat_this'" in str(e.stderr) or "Weights only load failed" in str(e.stderr):
                pytest.skip("Skipping beat_this test due to compatibility issues")
            raise

    # Define and ensure expected output file doesn't exist beforehand
    expected_video_file = audio_file_path.with_name(f"{audio_file_path.stem}_counter.mp4")
    expected_video_file.unlink(missing_ok=True)

    try:
        result = run_cli([
            "generate-video-batch",
            str(input_dir),
            "--min-measures", "1",  # Use min_measures=1 for testing with short audio samples
        ])

        assert result.returncode == 0

        expected_video_file = audio_file_path.with_name(f"{audio_file_path.stem}_counter.mp4")
        assert expected_video_file.is_file()
    except subprocess.CalledProcessError as e:
        # Skip if there are issues with the beat_this algorithm
        if "No module named 'beat_this'" in str(e.stderr) or "Weights only load failed" in str(e.stderr):
            pytest.skip("Skipping beat_this test due to compatibility issues")
        raise


def test_generate_video_batch_explicit_output(cli_test_setup):
    """Test `generate-video-batch` with an explicit output directory."""
    input_dir = cli_test_setup.parent
    audio_file_path = input_dir / SAMPLE_AUDIO_NAME
    beats_file = audio_file_path.with_suffix(".beats")
    output_dir = input_dir / "video_batch_output_beat_this"
    output_dir.mkdir(exist_ok=True) # Create the specific output subdir

    # Ensure beats file exists first
    if not beats_file.is_file():
        try:
            # Use min_measures=1 for testing with short audio samples
            result_beats = run_cli(["detect-beats", str(audio_file_path), "--algorithm", "beat_this", "--min-measures", "1"])
            assert result_beats.returncode == 0, f"Setup failed: Could not generate beats file: {result_beats.stderr}"
        except subprocess.CalledProcessError as e:
            # Skip if there are issues with the beat_this algorithm
            if "No module named 'beat_this'" in str(e.stderr) or "Weights only load failed" in str(e.stderr):
                pytest.skip("Skipping beat_this test due to compatibility issues")
            raise

    # Define and ensure expected output file doesn't exist beforehand
    expected_video_file = output_dir / f"{SAMPLE_AUDIO_PATH.stem}_counter.mp4"
    expected_video_file.unlink(missing_ok=True)

    try:
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
    except subprocess.CalledProcessError as e:
        # Skip if there are issues with the beat_this algorithm
        if "No module named 'beat_this'" in str(e.stderr) or "Weights only load failed" in str(e.stderr):
            pytest.skip("Skipping beat_this test due to compatibility issues")
        raise


def test_generate_video_batch_missing_beats(cli_test_setup):
    """Test `generate-video-batch` when a beats file is missing."""
    input_dir = cli_test_setup.parent
    audio_file_path = input_dir / SAMPLE_AUDIO_NAME
    beats_file = audio_file_path.with_suffix(".beats")
    
    # Ensure beats file doesn't exist
    beats_file.unlink(missing_ok=True)
    
    result = run_cli([
        "generate-video-batch",
        str(input_dir),
        "--min-measures", "1",
    ])
    
    # Should complete but report failures
    assert result.returncode == 0
    assert "failed" in result.stdout.lower() or "skipped" in result.stdout.lower()
