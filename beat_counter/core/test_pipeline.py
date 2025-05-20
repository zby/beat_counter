"""
Tests for the pipeline module.
"""
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from beat_counter.core.pipeline import process_batch, extract_beats
from beat_counter.core.beats import RawBeats, Beats


def test_extract_beats_calls_build():
    """Test that extract_beats uses the build function from registry."""
    # Create beats_per_bar that we'll use
    beats_per_bar = 3
    
    # Create enough data to satisfy the requirements (min_measures=5 with 3 beats per measure = 15 beats)
    timestamps = np.linspace(0, 7.0, 15)  # 15 evenly spaced beats
    # Create beat counts that cycle 1,2,3,1,2,3,... to simulate beats in 3/4 time
    beat_counts = np.array([(i % beats_per_bar) + 1 for i in range(15)])
    
    # Create a proper mock of RawBeats that passes isinstance check and has required attributes
    mock_raw_beats = MagicMock(spec=RawBeats)
    mock_raw_beats.timestamps = timestamps
    mock_raw_beats.beat_counts = beat_counts
    mock_raw_beats.clip_length = 8.0
    mock_raw_beats.save_to_file = MagicMock()
    
    # Create a mock detector that returns our mock_raw_beats
    mock_detector = MagicMock()
    mock_detector.detect_beats.return_value = mock_raw_beats
    
    # Create a mock Beats object that will be returned
    mock_beats = MagicMock()
    
    # Patch multiple functions to avoid validation
    with patch('beat_counter.core.pipeline.build', return_value=mock_detector) as mock_build, \
         patch('pathlib.Path.is_file', return_value=True), \
         patch('beat_counter.core.pipeline.get_output_path', return_value='test.beats'), \
         patch('beat_counter.core.pipeline.Beats', return_value=mock_beats) as mock_beats_class, \
         patch('builtins.open', MagicMock()), \
         patch('json.dump'):
        
        # Call the function we're testing
        result = extract_beats("test.wav", detector_name="madmom", min_bpm=90)
        
        # Verify that build was called with the correct arguments
        mock_build.assert_called_once_with("madmom", min_bpm=90)
        
        # Verify that the detector's detect_beats method was called
        mock_detector.detect_beats.assert_called_once_with("test.wav")
        
        # Verify that raw_beats.save_to_file was called
        mock_raw_beats.save_to_file.assert_called_once()
        
        # Verify that Beats constructor was called with raw_beats
        mock_beats_class.assert_called_once_with(mock_raw_beats, **{})
        
        # Verify that we got the expected result
        assert result is mock_beats


def test_extract_beats_validation_failure():
    """Test that raw beats are saved even when Beats validation fails."""
    # Create beats_per_bar that we'll use
    beats_per_bar = 3
    
    # Create enough data to satisfy the requirements (min_measures=5 with 3 beats per measure = 15 beats)
    timestamps = np.linspace(0, 7.0, 15)  # 15 evenly spaced beats
    # Create beat counts that cycle 1,2,3,1,2,3,... to simulate beats in 3/4 time
    beat_counts = np.array([(i % beats_per_bar) + 1 for i in range(15)])
    
    # Create a proper mock of RawBeats that passes isinstance check and has required attributes
    mock_raw_beats = MagicMock(spec=RawBeats)
    mock_raw_beats.timestamps = timestamps
    mock_raw_beats.beat_counts = beat_counts
    mock_raw_beats.clip_length = 8.0
    mock_raw_beats.save_to_file = MagicMock()
    
    # Create a mock detector that returns our mock_raw_beats
    mock_detector = MagicMock()
    mock_detector.detect_beats.return_value = mock_raw_beats
    
    # Patch multiple functions - but make Beats constructor raise an exception
    with patch('beat_counter.core.pipeline.build', return_value=mock_detector) as mock_build, \
         patch('pathlib.Path.is_file', return_value=True), \
         patch('beat_counter.core.pipeline.get_output_path', return_value='test.beats'), \
         patch('beat_counter.core.pipeline.Beats', side_effect=Exception("Validation failed")) as mock_beats_class, \
         patch('builtins.open', MagicMock()), \
         patch('json.dump'):
        
        # Call the function we're testing - should not raise exception
        result = extract_beats("test.wav", detector_name="madmom", min_bpm=90)
        
        # Verify that build was called with the correct arguments
        mock_build.assert_called_once_with("madmom", min_bpm=90)
        
        # Verify that the detector's detect_beats method was called
        mock_detector.detect_beats.assert_called_once_with("test.wav")
        
        # Verify that raw_beats.save_to_file was called
        mock_raw_beats.save_to_file.assert_called_once()
        
        # Verify that Beats constructor was called with raw_beats
        mock_beats_class.assert_called_once_with(mock_raw_beats, **{})
        
        # Verify that we got None as the result due to validation failure
        assert result is None


def test_process_batch_calls_extract_beats():
    """Test that process_batch calls extract_beats with the correct arguments."""
    mock_audio_files = [Path('/path/to/audio1.mp3'), Path('/path/to/audio2.mp3')]
    
    with patch('beat_counter.core.pipeline.Path.is_dir', return_value=True), \
         patch('beat_counter.core.pipeline.find_audio_files', return_value=mock_audio_files), \
         patch('beat_counter.core.pipeline.extract_beats') as mock_extract_beats, \
         patch('pathlib.Path.relative_to', side_effect=lambda x: Path(os.path.basename(x))):
        
        # Mock extract_beats to return a MagicMock
        mock_extract_beats.return_value = MagicMock()
        
        # Create detector kwargs dict
        detector_kwargs = {"min_bpm": 90}
        
        # Call process_batch with custom parameters
        process_batch("/path/to/dir", detector_name="beat_this", detector_kwargs=detector_kwargs, no_progress=True)
        
        # Verify extract_beats was called for each file with correct params
        assert mock_extract_beats.call_count == 2
        
        # Check first call
        args, kwargs = mock_extract_beats.call_args_list[0]
        assert kwargs['audio_file_path'] == str(mock_audio_files[0])
        assert kwargs['detector_name'] == "beat_this"
        assert kwargs['min_bpm'] == 90
        
        # Check second call
        args, kwargs = mock_extract_beats.call_args_list[1]
        assert kwargs['audio_file_path'] == str(mock_audio_files[1])
        assert kwargs['detector_name'] == "beat_this"
        assert kwargs['min_bpm'] == 90 