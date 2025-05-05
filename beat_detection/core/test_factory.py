"""
Tests for the beat detector factory.
"""
import pytest
from unittest.mock import patch, MagicMock
import os

from beat_detection.core.factory import get_beat_detector, DETECTOR_REGISTRY
from beat_detection.core.beat_this_detector import BeatThisDetector
from beat_detection.core.madmom_detector import MadmomBeatDetector
from beat_detection.core.beats import Beats
from beat_detection.core.beats import RawBeats


def test_get_beat_detector_default():
    """Test that the default detector is MadmomBeatDetector."""
    detector = get_beat_detector()
    assert isinstance(detector, MadmomBeatDetector)


def test_get_beat_detector_madmom():
    """Test getting a MadmomBeatDetector explicitly."""
    detector = get_beat_detector("madmom")
    assert isinstance(detector, MadmomBeatDetector)


# Create a patch at module level to mock the BeatThisDetector before it's imported
@patch('beat_detection.core.beat_this_detector.BeatThisDetector')
def test_get_beat_detector_beat_this(mock_detector):
    """Test getting a BeatThisDetector."""
    # Configure the mock to return a MagicMock instance when instantiated
    mock_instance = MagicMock()
    mock_detector.return_value = mock_instance
    
    # Reload the factory module to use our mock
    import importlib
    import beat_detection.core.factory
    importlib.reload(beat_detection.core.factory)
    
    # Call the factory function
    detector = beat_detection.core.factory.get_beat_detector("beat_this")
    
    # Verify the mock was called
    assert mock_detector.called
    assert detector is mock_instance


def test_get_beat_detector_invalid():
    """Test that an invalid algorithm name raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported beat detection algorithm"):
        get_beat_detector("invalid_algorithm")


def test_get_beat_detector_kwargs():
    """Test that kwargs are passed to the detector constructor."""
    # Create a detector with a custom parameter
    detector = get_beat_detector("madmom", fps=200)
    assert detector.fps == 200


def test_detector_registry():
    """Test that the detector registry contains the expected detectors."""
    assert "madmom" in DETECTOR_REGISTRY
    assert "beat_this" in DETECTOR_REGISTRY
    assert DETECTOR_REGISTRY["madmom"] == MadmomBeatDetector
    # We don't directly import BeatThisDetector to avoid ImportError
    assert DETECTOR_REGISTRY["beat_this"].__name__ == "BeatThisDetector"


# --- Tests for extract_beats --- 

@pytest.fixture
def mock_audio_file(tmp_path):
    """Create a dummy audio file."""
    audio_path = tmp_path / "test_audio.wav"
    audio_path.touch()
    return str(audio_path)

@patch('beat_detection.core.factory.get_beat_detector')
def test_extract_beats_default_output(mock_get_detector, tmp_path, mock_audio_file):
    """Test extract_beats with the default output path."""
    # Mock the detector and its method
    mock_detector_instance = MagicMock()
    # Make detect_beats return a mock RawBeats object
    mock_raw_beats = MagicMock(spec=RawBeats)
    mock_raw_beats.beats = [1.0, 2.0, 3.0]
    mock_detector_instance.detect_beats.return_value = mock_raw_beats
    mock_get_detector.return_value = mock_detector_instance

    # Mock the Beats class constructor and save_to_file
    with patch('beat_detection.core.factory.Beats', spec=Beats) as mock_beats_class:
        mock_beats_instance = MagicMock()
        # mock_beats_instance.beats = [1.0, 2.0, 3.0] # No longer needed for this assertion path
        # mock_beats_instance.save_to_file = MagicMock() # No longer needed
        mock_beats_class.return_value = mock_beats_instance

        # Call extract_beats
        from beat_detection.core.factory import extract_beats # Import locally
        # Pass empty beats_args for now
        result = extract_beats(mock_audio_file, algorithm="mock_alg", beats_args={})

        # Assertions
        mock_beats_class.assert_called_once_with(mock_raw_beats, **{})
        assert result is mock_beats_instance
        
        # Assert that save_to_file was called on the RAW beats object
        mock_raw_beats.save_to_file.assert_called_once()
        saved_path = mock_raw_beats.save_to_file.call_args[0][0]
        expected_output_path = tmp_path / "test_audio.beats"
        assert str(saved_path) == str(expected_output_path)

        # Check original Beats mock save was NOT called
        # mock_beats_instance.save_to_file.assert_not_called() # Not needed as we check raw_beats mock

        mock_get_detector.assert_called_once_with(algorithm="mock_alg")
        mock_detector_instance.detect_beats.assert_called_once_with(mock_audio_file)

        # Check file content (optional, as save_to_file is mocked)
        # assert expected_output_path.exists() 
        # with open(expected_output_path, 'r') as f:
        #     content = f.read()
        #     assert content == "1.000000\n2.000000\n3.000000\n"

@patch('beat_detection.core.factory.get_beat_detector')
def test_extract_beats_specified_output(mock_get_detector, tmp_path, mock_audio_file):
    """Test extract_beats with a specified output path."""
    # Mock the detector
    mock_detector_instance = MagicMock()
    mock_raw_beats = MagicMock(spec=RawBeats)
    mock_raw_beats.beats = [0.5, 1.5]
    mock_detector_instance.detect_beats.return_value = mock_raw_beats
    mock_get_detector.return_value = mock_detector_instance

    specified_output_path = tmp_path / "custom_output.txt"

    # Mock the Beats class
    with patch('beat_detection.core.factory.Beats', spec=Beats) as mock_beats_class:
        mock_beats_instance = MagicMock()
        # mock_beats_instance.beats = [0.5, 1.5]
        # mock_beats_instance.save_to_file = MagicMock()
        mock_beats_class.return_value = mock_beats_instance
        
        # Call extract_beats
        from beat_detection.core.factory import extract_beats # Import locally
        result = extract_beats(
            mock_audio_file,
            output_path=str(specified_output_path),
            algorithm="mock_alg",
            beats_args={"tolerance_percent": 5.0} # Example beats_args
        )

        # Assertions
        mock_beats_class.assert_called_once_with(mock_raw_beats, **{"tolerance_percent": 5.0})
        assert result is mock_beats_instance
        
        # Assert save_to_file called on raw_beats mock with specified path
        mock_raw_beats.save_to_file.assert_called_once_with(str(specified_output_path))
        
        # mock_beats_instance.save_to_file.assert_not_called()

        # Check that default path was NOT created (save_to_file is mocked, so file isn't actually written)
        # default_output_path = tmp_path / "test_audio.beats"
        # assert not default_output_path.exists()

@patch('beat_detection.core.factory.get_beat_detector')
def test_extract_beats_passes_kwargs_to_detector(mock_get_detector, tmp_path, mock_audio_file):
    """Test that extra kwargs are passed to get_beat_detector."""
    mock_detector_instance = MagicMock()
    mock_raw_beats = MagicMock(spec=RawBeats)
    mock_raw_beats.beats = [1.0]
    mock_detector_instance.detect_beats.return_value = mock_raw_beats
    mock_get_detector.return_value = mock_detector_instance

    with patch('beat_detection.core.factory.Beats', spec=Beats) as mock_beats_class:
        mock_beats_instance = MagicMock()
        mock_beats_class.return_value = mock_beats_instance

        from beat_detection.core.factory import extract_beats
        extract_beats(mock_audio_file, algorithm="mock_alg", beats_args={}, fps=100, extra_arg="test")

        # Assert save_to_file was called on raw_beats object
        mock_raw_beats.save_to_file.assert_called_once()

        # Assert that kwargs were passed to get_beat_detector
        mock_get_detector.assert_called_once_with(algorithm="mock_alg", fps=100, extra_arg="test")
