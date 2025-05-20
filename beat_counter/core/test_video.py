"""
Unit tests for video.py module.

These tests verify that the video generation functions work correctly,
with a focus on file handling, error conditions, and parameter passing.
"""
import pathlib
import pytest
from unittest.mock import patch, MagicMock, call

from beat_counter.core.video import (
    BeatVideoGenerator,
    generate_single_video_from_files,
    generate_batch_videos,
    prepare_beats_from_file,
    write_video_clip,
)
from beat_counter.core.beats import Beats, RawBeats


class TestBeatVideoGenerator:
    """Test the BeatVideoGenerator class."""
    
    @patch('beat_counter.core.video.VideoClip')
    def test_generate_video_returns_clip(self, mock_video_clip):
        """Test that generate_video returns a VideoClip object."""
        # Setup mocks
        mock_audio_clip = MagicMock()
        mock_audio_clip.duration = 10.0  # Required for function to work
        
        mock_beats = MagicMock()
        mock_beats.timestamps = [1.0, 2.0, 3.0, 4.0]
        mock_beats.get_info_at_time.return_value = (1, 0.5, 0)  # beat_count, time_since_beat, beat_idx
        mock_beats.beats_per_bar = 4
        
        mock_video = MagicMock()
        mock_video_clip.return_value = mock_video
        mock_video.with_audio.return_value = mock_video  # Return self from with_audio
        
        # Create generator and call generate_video
        generator = BeatVideoGenerator()
        
        # Patch the internal methods that would normally do frame generation
        with patch.object(generator, '_fill_frame_cache'), \
             patch.object(generator, 'create_counter_frame'):
            
            result = generator.generate_video(
                audio_clip=mock_audio_clip,
                beats=mock_beats
            )
        
        # Verify the result is a VideoClip
        assert result == mock_video
        
        # Verify internal interactions
        mock_video_clip.assert_called_once()  # VideoClip constructor was called
        mock_video.with_audio.assert_called_once_with(mock_audio_clip)  # Audio was attached
    
    @patch('beat_counter.core.video.VideoClip')
    def test_generate_video_with_sample_beats(self, mock_video_clip):
        """Test that generate_video handles sample_beats parameter correctly."""
        # Setup mocks
        mock_audio_clip = MagicMock()
        mock_audio_clip.duration = 10.0  # Set initial duration
        
        mock_beats = MagicMock()
        mock_beats.timestamps = [1.0, 2.0, 3.0, 4.0, 5.0]
        mock_beats.get_info_at_time.return_value = (1, 0.5, 0)
        mock_beats.beats_per_bar = 4
        
        mock_video = MagicMock()
        mock_video_clip.return_value = mock_video
        mock_video.with_audio.return_value = mock_video
        
        # Create generator and call generate_video with sample_beats=2
        generator = BeatVideoGenerator()
        
        with patch.object(generator, '_fill_frame_cache'), \
             patch.object(generator, 'create_counter_frame'):
            
            result = generator.generate_video(
                audio_clip=mock_audio_clip,
                beats=mock_beats,
                sample_beats=2  # Limit to first 2 beats
            )
        
        # Verify that audio duration was modified to end shortly after the second beat
        assert mock_audio_clip.duration == 3.0  # Second beat (2.0) + 1.0 second
        
        # Verify the result was returned correctly
        assert result == mock_video


class TestWriteVideoClip:
    """Test the write_video_clip function."""
    
    def test_write_video_clip_with_progress_callback(self):
        """Test that write_video_clip correctly handles progress callbacks."""
        # Create mocks
        mock_video_clip = MagicMock()
        mock_output_path = "output/test_video.mp4"
        mock_callback = MagicMock()
        
        # Call the function with a progress callback
        result = write_video_clip(
            video_clip=mock_video_clip,
            output_path=mock_output_path,
            progress_callback=mock_callback,
        )
        
        # Verify the results
        assert result == mock_output_path
        assert mock_callback.call_count == 2
        # First call with "Starting video encoding" and 0.8
        mock_callback.assert_any_call("Starting video encoding", 0.8)
        # Second call with "Video encoding complete" and 1.0
        mock_callback.assert_any_call("Video encoding complete", 1.0)


class TestPrepareBeatFromFile:
    """Test the prepare_beats_from_file helper function."""
    
    @pytest.fixture
    def mock_paths(self, tmp_path):
        """Create temporary paths for tests."""
        beats_file = tmp_path / "test_audio.beats"
        stats_file = tmp_path / "test_audio._beat_stats"
        # Create empty files
        beats_file.write_text('{"timestamps": [1.0, 2.0, 3.0], "audio_file": "test_audio.mp3"}')
        stats_file.write_text('{}')  # Just needs to exist
        return beats_file, stats_file
    
    @patch('beat_counter.core.video.RawBeats.load_from_file')
    @patch('beat_counter.core.video.Beats')
    def test_successful_preparation(self, mock_beats_class, mock_load, mock_paths):
        """Test successful preparation of Beats object."""
        beats_file, _ = mock_paths
        
        # Configure mocks
        mock_raw_beats = MagicMock()
        mock_raw_beats.timestamps = [1.0, 2.0, 3.0]
        mock_load.return_value = mock_raw_beats
        
        mock_beats = MagicMock()
        mock_beats_class.return_value = mock_beats
        
        # Call the function
        result = prepare_beats_from_file(beats_file=beats_file)
        
        # Verify result and mocks
        assert result == mock_beats
        mock_load.assert_called_once_with(beats_file)
        mock_beats_class.assert_called_once_with(
            raw_beats=mock_raw_beats,
            beats_per_bar=None,
            tolerance_percent=10.0,
            min_measures=5
        )
    
    @patch('beat_counter.core.video.RawBeats.load_from_file')
    @patch('beat_counter.core.video.Beats')
    def test_custom_parameters(self, mock_beats_class, mock_load, mock_paths):
        """Test preparation with custom parameters."""
        beats_file, _ = mock_paths
        
        # Configure mocks
        mock_raw_beats = MagicMock()
        mock_load.return_value = mock_raw_beats
        
        mock_beats = MagicMock()
        mock_beats_class.return_value = mock_beats
        
        # Custom parameters
        custom_tolerance = 15.0
        custom_min_measures = 10
        
        # Call with custom parameters
        result = prepare_beats_from_file(
            beats_file=beats_file,
            tolerance_percent=custom_tolerance,
            min_measures=custom_min_measures
        )
        
        # Verify parameter passing
        mock_beats_class.assert_called_once_with(
            raw_beats=mock_raw_beats,
            beats_per_bar=None,
            tolerance_percent=custom_tolerance,
            min_measures=custom_min_measures
        )
    
    def test_error_handling(self, tmp_path):
        """Test error handling for different error conditions."""
        # Test beats file not found
        nonexistent_file = tmp_path / "nonexistent.beats"
        with pytest.raises(FileNotFoundError, match="Beats file not found"):
            prepare_beats_from_file(beats_file=nonexistent_file)
        
        # Test stats file not found
        beats_file = tmp_path / "test_audio.beats"
        beats_file.write_text('{"timestamps": [1.0, 2.0, 3.0], "audio_file": "test_audio.mp3"}')
        
        with pytest.raises(FileNotFoundError, match="Beat statistics file not found"):
            prepare_beats_from_file(beats_file=beats_file)
        
        # Create stats file for further tests
        stats_file = tmp_path / "test_audio._beat_stats"
        stats_file.write_text('{}')
        
        # Test raw beats loading error
        with patch('beat_counter.core.video.RawBeats.load_from_file') as mock_load:
            mock_load.side_effect = Exception("Failed to load beats")
            with pytest.raises(RuntimeError, match="Failed to load raw beats"):
                prepare_beats_from_file(beats_file=beats_file)
        
        # Test beats reconstruction error
        with patch('beat_counter.core.video.RawBeats.load_from_file') as mock_load, \
             patch('beat_counter.core.video.Beats') as mock_beats_class:
            mock_load.return_value = MagicMock()
            mock_beats_class.side_effect = Exception("Failed to reconstruct")
            with pytest.raises(RuntimeError, match="Failed to reconstruct Beats object"):
                prepare_beats_from_file(beats_file=beats_file)


class TestGenerateSingleVideoFromFiles:
    """Test the generate_single_video_from_files function."""

    @pytest.fixture
    def mock_paths(self, tmp_path):
        """Create temporary paths for tests."""
        audio_file = tmp_path / "test_audio.mp3"
        beats_file = tmp_path / "test_audio.beats"
        stats_file = tmp_path / "test_audio._beat_stats"
        output_file = tmp_path / "output_video.mp4"
        # Create empty files
        audio_file.write_text("mock audio content")
        beats_file.write_text('{"timestamps": [1.0, 2.0, 3.0], "audio_file": "test_audio.mp3"}')
        stats_file.write_text('{}')  # Just needs to exist
        return audio_file, beats_file, stats_file, output_file

    @patch('beat_counter.core.video.AudioFileClip')
    @patch('beat_counter.core.video.prepare_beats_from_file')
    @patch('beat_counter.core.video.BeatVideoGenerator')
    @patch('pathlib.Path.is_file')
    def test_end_to_end_generation(self, mock_is_file, mock_generator_class, mock_prepare_beats, mock_audio_clip, mock_paths):
        """Test end-to-end video generation with different parameter combinations."""
        audio_file, beats_file, _, output_file = mock_paths
        
        # Mock that all files exist
        mock_is_file.return_value = True
        
        # Configure mocks
        mock_beats = MagicMock()
        mock_prepare_beats.return_value = mock_beats
        
        mock_audio = MagicMock()
        mock_audio_clip.return_value = mock_audio
        
        mock_video_clip = MagicMock()
        mock_generator = MagicMock()
        mock_generator.generate_video.return_value = mock_video_clip
        mock_generator_class.return_value = mock_generator
        
        # Test cases with different parameter combinations
        test_cases = [
            # Default parameters
            {
                "params": {
                    "audio_file": audio_file,
                    "beats_file": beats_file,
                    "output_file": output_file,
                    "verbose": True
                },
                "expected_generator_params": {"resolution": (720, 540), "fps": 100},
                "expected_video_params": {"sample_beats": None},
                "expected_beats_params": {"tolerance_percent": 10.0, "min_measures": 5}
            },
            # Custom parameters
            {
                "params": {
                    "audio_file": audio_file,
                    "beats_file": beats_file,
                    "output_file": output_file, 
                    "resolution": (1280, 720),
                    "fps": 120,
                    "tolerance_percent": 15.0,
                    "min_measures": 10,
                    "sample_beats": 5,
                    "verbose": False
                },
                "expected_generator_params": {"resolution": (1280, 720), "fps": 120},
                "expected_video_params": {"sample_beats": 5},
                "expected_beats_params": {"tolerance_percent": 15.0, "min_measures": 10}
            },
            # Default output path
            {
                "params": {
                    "audio_file": audio_file,
                    "beats_file": beats_file,
                    "output_file": None,
                    "verbose": False
                },
                "expected_generator_params": {"resolution": (720, 540), "fps": 100},
                "expected_video_params": {"sample_beats": None},
                "expected_beats_params": {"tolerance_percent": 10.0, "min_measures": 5},
                "expected_output": audio_file.with_name(f"{audio_file.stem}_counter.mp4")
            }
        ]
        
        # Run each test case
        for i, case in enumerate(test_cases):
            # Reset mocks
            mock_prepare_beats.reset_mock()
            mock_generator_class.reset_mock()
            mock_generator.reset_mock()
            
            # Mock write_video_clip to return expected output
            expected_output = case.get("expected_output", output_file)
            with patch('beat_counter.core.video.write_video_clip') as mock_write_video:
                mock_write_video.return_value = str(expected_output)
                
                # Call the function with test case parameters
                result = generate_single_video_from_files(**case["params"])
            
            # Verify result
            assert result == expected_output
            
            # Verify prepare_beats_from_file was called with correct params
            mock_prepare_beats.assert_called_once_with(
                beats_file=beats_file,
                **case["expected_beats_params"]
            )
            
            # Verify BeatVideoGenerator was initialized with correct params
            if case["expected_generator_params"]:
                mock_generator_class.assert_called_once_with(**case["expected_generator_params"])
            else:
                mock_generator_class.assert_called_once_with()
                
            # Verify generate_video was called with correct params
            mock_generator.generate_video.assert_called_once_with(
                audio_clip=mock_audio,
                beats=mock_beats,
                **case["expected_video_params"]
            )
    
    def test_audio_file_not_found(self, tmp_path):
        """Test error handling when audio file doesn't exist."""
        audio_file = tmp_path / "nonexistent_audio.mp3"
        beats_file = tmp_path / "nonexistent_audio.beats"
        
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            generate_single_video_from_files(
                audio_file=audio_file,
                beats_file=beats_file,
                verbose=False
            )


class TestGenerateBatchVideos:
    """Test the generate_batch_videos function."""

    @pytest.fixture
    def mock_directory_structure(self, tmp_path):
        """Create a mock directory structure for testing."""
        # Create minimal test structure
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create a test audio file with associated files
        audio1 = input_dir / "track1.mp3"
        beats1 = input_dir / "track1.beats"
        stats1 = input_dir / "track1._beat_stats"
        
        # Create sample content
        audio1.write_text("mock audio content")
        beats1.write_text('{"timestamps": [1.0, 2.0, 3.0], "audio_file": "track1.mp3"}')
        stats1.write_text('{}')
        
        return {
            "input_dir": input_dir,
            "audio_files": [audio1],
            "beats_files": [beats1]
        }

    @patch('beat_counter.core.video.find_audio_files')
    @patch('beat_counter.core.video.generate_single_video_from_files')
    def test_batch_processing(self, mock_generate_single, mock_find_audio, mock_directory_structure):
        """Test batch processing of audio files."""
        input_dir = mock_directory_structure["input_dir"]
        audio_files = mock_directory_structure["audio_files"]
        
        # Configure mocks
        mock_find_audio.return_value = audio_files
        
        # Setup output paths
        output_paths = [audio_file.with_name(f"{audio_file.stem}_counter.mp4") for audio_file in audio_files]
        mock_generate_single.side_effect = output_paths
        
        # Call the function
        results = generate_batch_videos(
            input_dir=input_dir,
            verbose=False,
            no_progress=True
        )
        
        # Verify results
        assert len(results) == len(audio_files)
        for (rel_path, success, output_path), expected_output in zip(results, output_paths):
            assert success is True
            assert output_path == expected_output
        
        # Verify correct calls
        assert mock_find_audio.call_count == 1
        assert mock_generate_single.call_count == len(audio_files)
    
    @patch('beat_counter.core.video.find_audio_files')
    @patch('beat_counter.core.video.generate_single_video_from_files')
    def test_batch_with_custom_output_dir(self, mock_generate_single, mock_find_audio, mock_directory_structure, tmp_path):
        """Test batch processing with custom output directory."""
        input_dir = mock_directory_structure["input_dir"]
        audio_files = mock_directory_structure["audio_files"]
        output_dir = tmp_path / "output"
        
        # Configure mocks
        mock_find_audio.return_value = audio_files
        
        # Setup expected outputs
        output_paths = [output_dir / f"{audio_file.stem}_counter.mp4" for audio_file in audio_files]
        mock_generate_single.side_effect = output_paths
        
        # Call with custom output directory
        results = generate_batch_videos(
            input_dir=input_dir,
            output_dir=output_dir,
            verbose=False,
            no_progress=True
        )
        
        # Verify results
        assert len(results) == len(audio_files)
        
        # Verify correct output path construction
        for audio_file, expected_output in zip(audio_files, output_paths):
            beats_file = audio_file.with_suffix(".beats")
            mock_generate_single.assert_any_call(
                audio_file=audio_file,
                beats_file=beats_file,
                output_file=expected_output,
                resolution=(720, 540),
                fps=100,
                sample_beats=None,
                tolerance_percent=10.0,
                min_measures=5,
                verbose=False
            )
    
    def test_nonexistent_input_dir(self):
        """Test error handling for nonexistent input directory."""
        nonexistent_dir = pathlib.Path("/nonexistent/directory")
        
        with pytest.raises(FileNotFoundError, match="Input directory not found"):
            generate_batch_videos(input_dir=nonexistent_dir, verbose=False) 