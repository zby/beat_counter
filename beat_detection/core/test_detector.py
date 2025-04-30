import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from beat_detection.core.detector import MadmomBeatDetector, BeatDetector # Import Protocol too for type checking
from beat_detection.core.beats import RawBeats, BeatCalculationError
from beat_detection.utils.constants import SUPPORTED_BEATS_PER_BAR # Import needed constant

# Sample data returned by mocked madmom
SAMPLE_MADMOM_OUTPUT = np.array([[0.5, 1], [1.0, 2], [1.5, 3], [2.0, 4]], dtype=float)
SAMPLE_TIMESTAMPS = np.array([0.5, 1.0, 1.5, 2.0])
SAMPLE_COUNTS = np.array([1, 2, 3, 4])

@pytest.fixture
def mock_madmom_processors():
    """Mocks the madmom processors used by MadmomBeatDetector."""
    # Use the correct string literal for patching
    with patch('beat_detection.core.detector.RNNDownBeatProcessor') as mock_rnn, \
         patch('beat_detection.core.detector.DBNDownBeatTrackingProcessor') as mock_dbn:

        # Mock the instances returned by the constructors
        mock_rnn_instance = MagicMock()
        mock_dbn_instance = MagicMock()

        # Mock the __call__ method of the instances
        # RNNDownBeatProcessor returns activations
        mock_rnn_instance.return_value = np.random.rand(100, 3) # Dummy activations
        # DBNDownBeatTrackingProcessor returns the beats array
        mock_dbn_instance.return_value = SAMPLE_MADMOM_OUTPUT

        mock_rnn.return_value = mock_rnn_instance
        mock_dbn.return_value = mock_dbn_instance

        yield mock_rnn, mock_dbn, mock_rnn_instance, mock_dbn_instance

@pytest.fixture
def audio_file_fixture(tmp_path) -> Path:
    """Creates a dummy audio file for testing existence checks."""
    p = tmp_path / "test.wav"
    p.touch() # Create the file
    return p

# --- Test __init__ ---

def test_madmom_detector_init(mock_madmom_processors):
    """Test MadmomBeatDetector initialization."""
    mock_rnn, mock_dbn, _, _ = mock_madmom_processors
    detector = MadmomBeatDetector(min_bpm=70, max_bpm=180, fps=50)
    assert detector.min_bpm == 70
    assert detector.max_bpm == 180
    assert detector.fps == 50
    assert detector.progress_callback is None
    mock_rnn.assert_called_once()
    mock_dbn.assert_called_once_with(
        beats_per_bar=SUPPORTED_BEATS_PER_BAR, # Use imported constant
        min_bpm=70.0,
        max_bpm=180.0,
        fps=50.0
    )
    # Check it conforms to protocol (runtime check)
    assert isinstance(detector, BeatDetector)

def test_madmom_detector_invalid_init_params():
    """Test MadmomBeatDetector __init__ raises error for invalid BPM/FPS settings."""
    # Test invalid min_bpm
    with pytest.raises(BeatCalculationError, match="Invalid min_bpm"):
        MadmomBeatDetector(min_bpm=0)
    with pytest.raises(BeatCalculationError, match="Invalid min_bpm"):
        MadmomBeatDetector(min_bpm=-10)
    
    # Test invalid max_bpm
    with pytest.raises(BeatCalculationError, match="Invalid max_bpm"):
        MadmomBeatDetector(min_bpm=120, max_bpm=100) # max <= min
    with pytest.raises(BeatCalculationError, match="Invalid max_bpm"):
        MadmomBeatDetector(min_bpm=120, max_bpm=120) # max <= min

    # Test invalid fps
    with pytest.raises(BeatCalculationError, match="Invalid fps"):
        MadmomBeatDetector(fps=0)
    with pytest.raises(BeatCalculationError, match="Invalid fps"):
        MadmomBeatDetector(fps=-100)

# --- Test _detect_downbeats ---

def test_madmom_detect_downbeats_success(mock_madmom_processors, audio_file_fixture):
    """Test _detect_downbeats returns correct array on success."""
    _, _, mock_rnn_instance, mock_dbn_instance = mock_madmom_processors
    detector = MadmomBeatDetector()
    result = detector._detect_downbeats(str(audio_file_fixture))

    mock_rnn_instance.assert_called_once_with(str(audio_file_fixture))
    # Check DBN tracker was called with the *output* of RNN
    mock_dbn_instance.assert_called_once_with(mock_rnn_instance.return_value)
    np.testing.assert_array_equal(result, SAMPLE_MADMOM_OUTPUT)

def test_madmom_detect_downbeats_no_beats(mock_madmom_processors, audio_file_fixture):
    """Test _detect_downbeats returns empty array when madmom finds no beats."""
    _, _, _, mock_dbn_instance = mock_madmom_processors
    mock_dbn_instance.return_value = np.empty((0, 2))
    detector = MadmomBeatDetector()
    result = detector._detect_downbeats(str(audio_file_fixture))
    np.testing.assert_array_equal(result, np.empty((0, 2)))

def test_madmom_detect_downbeats_madmom_error(mock_madmom_processors, audio_file_fixture):
    """Test _detect_downbeats raises BeatCalculationError if madmom fails."""
    _, _, mock_rnn_instance, _ = mock_madmom_processors
    mock_rnn_instance.side_effect = Exception("Madmom internal error")
    detector = MadmomBeatDetector()
    with pytest.raises(BeatCalculationError, match="Madmom processing failed: Madmom internal error"):
        detector._detect_downbeats(str(audio_file_fixture))

def test_madmom_detect_downbeats_unexpected_shape(mock_madmom_processors, audio_file_fixture):
    """Test _detect_downbeats raises error for unexpected madmom output shape."""
    _, _, _, mock_dbn_instance = mock_madmom_processors
    mock_dbn_instance.return_value = np.array([0.5, 1.0, 1.5]) # 1D array
    detector = MadmomBeatDetector()
    with pytest.raises(BeatCalculationError, match="unexpected shape"):
        detector._detect_downbeats(str(audio_file_fixture))

    mock_dbn_instance.return_value = np.array([[0.5], [1.0]]) # Shape (N, 1)
    detector = MadmomBeatDetector() # Re-instantiate if necessary, though not strictly needed here
    with pytest.raises(BeatCalculationError, match="unexpected shape"):
        detector._detect_downbeats(str(audio_file_fixture))


# --- Test detect ---

def test_madmom_detect_success(mock_madmom_processors, audio_file_fixture):
    """Test the main detect method successfully returns RawBeats."""
    detector = MadmomBeatDetector()

    # Mock the internal call
    with patch.object(detector, '_detect_downbeats', return_value=SAMPLE_MADMOM_OUTPUT) as mock_internal_detect:
        raw_beats = detector.detect(audio_file_fixture)

        mock_internal_detect.assert_called_once_with(str(audio_file_fixture))
        assert isinstance(raw_beats, RawBeats)
        np.testing.assert_array_equal(raw_beats.timestamps, SAMPLE_TIMESTAMPS)
        np.testing.assert_array_equal(raw_beats.beat_counts, SAMPLE_COUNTS)
        # Verify beats_per_bar is NOT present
        assert not hasattr(raw_beats, 'beats_per_bar')

def test_madmom_detect_file_not_found():
    """Test detect raises FileNotFoundError for non-existent file."""
    detector = MadmomBeatDetector()
    with pytest.raises(FileNotFoundError):
        detector.detect("non_existent_file.wav")

def test_madmom_detect_no_beats_error(mock_madmom_processors, audio_file_fixture):
    """Test detect raises BeatCalculationError if _detect_downbeats finds no beats."""
    detector = MadmomBeatDetector()

    # Mock the internal call to return empty array
    with patch.object(detector, '_detect_downbeats', return_value=np.empty((0, 2))) as mock_internal_detect:
        with pytest.raises(BeatCalculationError, match="No beats detected"):
            detector.detect(audio_file_fixture)
        mock_internal_detect.assert_called_once_with(str(audio_file_fixture))

def test_madmom_detect_calls_progress_callback(mock_madmom_processors, audio_file_fixture):
    """Test that the progress callback is called during detection."""
    mock_callback = MagicMock()
    detector = MadmomBeatDetector(progress_callback=mock_callback)

    # Mock the internal call
    with patch.object(detector, '_detect_downbeats', return_value=SAMPLE_MADMOM_OUTPUT):
        detector.detect(audio_file_fixture)

    # Check if callback was called (at least at start and end)
    assert mock_callback.call_count >= 2
    mock_callback.assert_any_call(0.0)
    mock_callback.assert_any_call(1.0)
