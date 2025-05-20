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
)
from beat_counter.core.beats import Beats, RawBeats


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
    
    def test_beats_file_not_found(self, tmp_path):
        """Test error when beats file doesn't exist."""
        nonexistent_file = tmp_path / "nonexistent.beats"
        
        with pytest.raises(FileNotFoundError, match="Beats file not found"):
            prepare_beats_from_file(beats_file=nonexistent_file)
    
    def test_stats_file_not_found(self, tmp_path):
        """Test error when stats file doesn't exist."""
        # Create beats file but not stats file
        beats_file = tmp_path / "test_audio.beats"
        beats_file.write_text('{"timestamps": [1.0, 2.0, 3.0], "audio_file": "test_audio.mp3"}')
        
        with pytest.raises(FileNotFoundError, match="Beat statistics file not found"):
            prepare_beats_from_file(beats_file=beats_file)
    
    @patch('beat_counter.core.video.RawBeats.load_from_file')
    def test_raw_beats_loading_error(self, mock_load, mock_paths):
        """Test error handling when RawBeats loading fails."""
        beats_file, _ = mock_paths
        
        # Configure mock to raise exception
        mock_load.side_effect = Exception("Failed to load beats")
        
        # Test exception handling
        with pytest.raises(RuntimeError, match="Failed to load raw beats"):
            prepare_beats_from_file(beats_file=beats_file)
    
    @patch('beat_counter.core.video.RawBeats.load_from_file')
    @patch('beat_counter.core.video.Beats')
    def test_beats_reconstruction_error(self, mock_beats_class, mock_load, mock_paths):
        """Test error handling when Beats reconstruction fails."""
        beats_file, _ = mock_paths
        
        # Configure mocks
        mock_raw_beats = MagicMock()
        mock_load.return_value = mock_raw_beats
        
        # Make Beats constructor fail
        mock_beats_class.side_effect = Exception("Failed to reconstruct")
        
        # Test exception handling
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
    def test_successful_generation(self, mock_is_file, mock_generator_class, mock_prepare_beats, mock_audio_clip, mock_paths):
        """Test successful video generation with default parameters."""
        audio_file, beats_file, _, output_file = mock_paths
        
        # Mock that all files exist (to get past the checks)
        mock_is_file.return_value = True
        
        # Configure mocks
        mock_beats = MagicMock()
        mock_prepare_beats.return_value = mock_beats
        
        mock_audio = MagicMock()
        mock_audio_clip.return_value = mock_audio
        
        mock_generator = MagicMock()
        mock_generator.generate_video.return_value = str(output_file)
        mock_generator_class.return_value = mock_generator
        
        # Call the function
        result = generate_single_video_from_files(
            audio_file=audio_file,
            beats_file=beats_file,
            output_file=output_file,
            verbose=True
        )
        
        # Verify the result
        assert result == output_file
        
        # Verify mock calls
        mock_prepare_beats.assert_called_once_with(
            beats_file=beats_file,
            tolerance_percent=10.0,
            min_measures=5
        )
        mock_audio_clip.assert_called_once_with(str(audio_file))
        mock_generator_class.assert_called_once()
        mock_generator.generate_video.assert_called_once_with(
            audio_clip=mock_audio,
            beats=mock_beats,
            output_path=output_file,
            sample_beats=None
        )

    @patch('beat_counter.core.video.AudioFileClip')
    @patch('beat_counter.core.video.prepare_beats_from_file')
    @patch('beat_counter.core.video.BeatVideoGenerator')
    @patch('pathlib.Path.is_file')
    def test_custom_parameters(self, mock_is_file, mock_generator_class, mock_prepare_beats, mock_audio_clip, mock_paths):
        """Test video generation with custom parameters."""
        audio_file, beats_file, _, output_file = mock_paths
        
        # Mock that all files exist (to get past the checks)
        mock_is_file.return_value = True
        
        # Configure mocks
        mock_beats = MagicMock()
        mock_prepare_beats.return_value = mock_beats
        
        mock_audio = MagicMock()
        mock_audio_clip.return_value = mock_audio
        
        mock_generator = MagicMock()
        mock_generator.generate_video.return_value = str(output_file)
        mock_generator_class.return_value = mock_generator
        
        # Custom parameters
        custom_resolution = (1280, 720)
        custom_fps = 120
        custom_tolerance = 15.0
        custom_min_measures = 10
        custom_sample_beats = 5
        
        # Call the function
        result = generate_single_video_from_files(
            audio_file=audio_file,
            beats_file=beats_file,
            output_file=output_file,
            resolution=custom_resolution,
            fps=custom_fps,
            tolerance_percent=custom_tolerance,
            min_measures=custom_min_measures,
            sample_beats=custom_sample_beats,
            verbose=False
        )
        
        # Verify that parameters were passed correctly
        mock_prepare_beats.assert_called_once_with(
            beats_file=beats_file,
            tolerance_percent=custom_tolerance,
            min_measures=custom_min_measures
        )
        mock_audio_clip.assert_called_once_with(str(audio_file))
        mock_generator_class.assert_called_once_with(
            resolution=custom_resolution, 
            fps=custom_fps
        )
        mock_generator.generate_video.assert_called_once_with(
            audio_clip=mock_audio,
            beats=mock_beats,
            output_path=output_file,
            sample_beats=custom_sample_beats
        )

    def test_file_not_found(self, tmp_path):
        """Test error handling when files don't exist."""
        # Non-existent files
        audio_file = tmp_path / "nonexistent_audio.mp3"
        beats_file = tmp_path / "nonexistent_audio.beats"
        
        # Test audio file not found
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            generate_single_video_from_files(
                audio_file=audio_file,
                beats_file=beats_file,
                verbose=False
            )
        
        # Create the audio file but not the beats file
        audio_file.write_text("mock audio content")
        
        # The beats file check is now in prepare_beats_from_file
        # so we don't directly test it here anymore - it will be tested
        # by the beats_file_not_found test in TestPrepareBeatFromFile

    @patch('beat_counter.core.video.prepare_beats_from_file')
    @patch('pathlib.Path.is_file')
    def test_beats_preparation_error(self, mock_is_file, mock_prepare_beats, mock_paths):
        """Test error propagation when beats preparation fails."""
        audio_file, beats_file, _, _ = mock_paths
        
        # Mock that audio file exists
        mock_is_file.return_value = True
        
        # Configure mock to raise an exception
        mock_prepare_beats.side_effect = RuntimeError("Failed to prepare beats")
        
        # Test exception handling
        with pytest.raises(RuntimeError, match="Failed to prepare beats"):
            generate_single_video_from_files(
                audio_file=audio_file,
                beats_file=beats_file,
                verbose=False
            )

    @patch('beat_counter.core.video.AudioFileClip')
    @patch('beat_counter.core.video.prepare_beats_from_file')
    @patch('beat_counter.core.video.BeatVideoGenerator')
    @patch('pathlib.Path.is_file')
    def test_default_output_path(self, mock_is_file, mock_generator_class, mock_prepare_beats, mock_audio_clip, mock_paths):
        """Test that default output path is correctly constructed."""
        audio_file, beats_file, _, _ = mock_paths
        
        # Mock that all files exist (to get past the checks)
        mock_is_file.return_value = True
        
        # Configure mocks
        mock_beats = MagicMock()
        mock_prepare_beats.return_value = mock_beats
        
        mock_audio = MagicMock()
        mock_audio_clip.return_value = mock_audio
        
        mock_generator = MagicMock()
        expected_default_output = audio_file.with_name(f"{audio_file.stem}_counter.mp4")
        mock_generator.generate_video.return_value = str(expected_default_output)
        mock_generator_class.return_value = mock_generator
        
        # Call function without specifying output_file
        result = generate_single_video_from_files(
            audio_file=audio_file,
            beats_file=beats_file,
            output_file=None,  # Use default
            verbose=False
        )
        
        # Verify default output path was used
        assert result == expected_default_output
        mock_generator.generate_video.assert_called_once_with(
            audio_clip=mock_audio,
            beats=mock_beats,
            output_path=expected_default_output,
            sample_beats=None
        )


class TestGenerateBatchVideos:
    """Test the generate_batch_videos function."""

    @pytest.fixture
    def mock_directory_structure(self, tmp_path):
        """Create a mock directory structure for testing."""
        # Main input directory
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create sub-directories
        sub_dir = input_dir / "subdir"
        sub_dir.mkdir()
        
        # Create audio files and corresponding beat files in input dir
        audio1 = input_dir / "track1.mp3"
        beats1 = input_dir / "track1.beats"
        stats1 = input_dir / "track1._beat_stats"
        
        audio2 = input_dir / "track2.mp3"
        beats2 = input_dir / "track2.beats"
        stats2 = input_dir / "track2._beat_stats"
        
        # Create audio file and corresponding beat file in sub dir
        audio3 = sub_dir / "track3.mp3"
        beats3 = sub_dir / "track3.beats"
        stats3 = sub_dir / "track3._beat_stats"
        
        # Create sample content
        audio_content = "mock audio content"
        beats_content = '{"timestamps": [1.0, 2.0, 3.0], "audio_file": "track.mp3"}'
        stats_content = '{}'
        
        # Write content to files
        audio1.write_text(audio_content)
        beats1.write_text(beats_content)
        stats1.write_text(stats_content)
        
        audio2.write_text(audio_content)
        beats2.write_text(beats_content)
        stats2.write_text(stats_content)
        
        audio3.write_text(audio_content)
        beats3.write_text(beats_content)
        stats3.write_text(stats_content)
        
        # Return relevant paths
        return {
            "input_dir": input_dir,
            "sub_dir": sub_dir,
            "audio_files": [audio1, audio2, audio3],
            "beats_files": [beats1, beats2, beats3],
            "stats_files": [stats1, stats2, stats3]
        }

    @patch('beat_counter.core.video.find_audio_files')
    @patch('beat_counter.core.video.generate_single_video_from_files')
    def test_batch_processing(self, mock_generate_single, mock_find_audio, mock_directory_structure):
        """Test successful batch processing of multiple audio files."""
        input_dir = mock_directory_structure["input_dir"]
        audio_files = mock_directory_structure["audio_files"]
        
        # Configure mocks
        mock_find_audio.return_value = audio_files
        
        # Set up mock output paths
        output_paths = [audio_file.with_name(f"{audio_file.stem}_counter.mp4") for audio_file in audio_files]
        # Configure generate_single_video_from_files to return mock output paths
        mock_generate_single.side_effect = output_paths
        
        # Call the function
        results = generate_batch_videos(
            input_dir=input_dir,
            verbose=False,
            no_progress=True
        )
        
        # Verify the results
        assert len(results) == len(audio_files)
        for (rel_path, success, output_path), audio_file, expected_output in zip(results, audio_files, output_paths):
            assert success is True
            assert output_path == expected_output
        
        # Verify mock calls
        assert mock_find_audio.call_count == 1
        assert mock_generate_single.call_count == len(audio_files)
        
        # Verify parameters passed to generate_single_video_from_files
        for i, audio_file in enumerate(audio_files):
            beats_file = audio_file.with_suffix(".beats")
            mock_generate_single.assert_any_call(
                audio_file=audio_file,
                beats_file=beats_file,
                output_file=None,
                resolution=(720, 540),
                fps=100,
                sample_beats=None,
                tolerance_percent=10.0,
                min_measures=5,
                verbose=False
            )

    @patch('beat_counter.core.video.find_audio_files')
    @patch('beat_counter.core.video.generate_single_video_from_files')
    def test_batch_with_custom_output_dir(self, mock_generate_single, mock_find_audio, mock_directory_structure, tmp_path):
        """Test batch processing with custom output directory."""
        input_dir = mock_directory_structure["input_dir"]
        audio_files = mock_directory_structure["audio_files"]
        output_dir = tmp_path / "output"
        
        # Configure mocks
        mock_find_audio.return_value = audio_files
        
        # Setup output paths
        output_paths = []
        for audio_file in audio_files:
            # Calculate the relative path within the input directory
            if audio_file.is_relative_to(input_dir):
                rel_path = audio_file.relative_to(input_dir)
                output_path = output_dir / rel_path.parent / f"{audio_file.stem}_counter.mp4"
            else:
                output_path = output_dir / f"{audio_file.stem}_counter.mp4"
            output_paths.append(output_path)
            
        # Configure generate_single_video_from_files to return these paths
        mock_generate_single.side_effect = output_paths
        
        # Call function with custom output directory
        results = generate_batch_videos(
            input_dir=input_dir,
            output_dir=output_dir,
            verbose=False,
            no_progress=True
        )
        
        # Verify the results
        assert len(results) == len(audio_files)
        
        # Verify mock calls - make sure correct output paths were used
        for i, (audio_file, expected_output) in enumerate(zip(audio_files, output_paths)):
            # For files in subdirectories, the structure should be preserved
            if audio_file.is_relative_to(input_dir):
                rel_path = audio_file.relative_to(input_dir)
                expected_output_dir = output_dir / rel_path.parent
            else:
                expected_output_dir = output_dir
                
            # The specific output file should be specified correctly
            expected_output_file = expected_output_dir / f"{audio_file.stem}_counter.mp4"
            beats_file = audio_file.with_suffix(".beats")
            
            # Verify this specific call
            mock_generate_single.assert_any_call(
                audio_file=audio_file,
                beats_file=beats_file,
                output_file=expected_output_file,
                resolution=(720, 540),
                fps=100,
                sample_beats=None,
                tolerance_percent=10.0,
                min_measures=5,
                verbose=False
            )

    @patch('beat_counter.core.video.find_audio_files')
    def test_no_audio_files(self, mock_find_audio, mock_directory_structure):
        """Test behavior when no audio files are found."""
        input_dir = mock_directory_structure["input_dir"]
        
        # Configure mock to return empty list
        mock_find_audio.return_value = []
        
        # Call the function
        results = generate_batch_videos(
            input_dir=input_dir,
            verbose=False,
            no_progress=True
        )
        
        # Verify the results
        assert results == []

    @patch('beat_counter.core.video.find_audio_files')
    @patch('beat_counter.core.video.generate_single_video_from_files')
    def test_handle_missing_beats_file(self, mock_generate_single, mock_find_audio, mock_directory_structure):
        """Test handling of missing beats file."""
        input_dir = mock_directory_structure["input_dir"]
        audio_files = mock_directory_structure["audio_files"]
        
        # Configure mocks
        mock_find_audio.return_value = audio_files
        
        # Make generate_single_video_from_files fail for the second file
        def generate_side_effect(audio_file, beats_file, **kwargs):
            # Raise error for the second file
            if str(beats_file).endswith("track2.beats"):
                raise FileNotFoundError(f"Beats file not found: {beats_file}")
            # For other files, return a mock output path
            return audio_file.with_name(f"{audio_file.stem}_counter.mp4")
        
        mock_generate_single.side_effect = generate_side_effect
        
        # Call the function
        results = generate_batch_videos(
            input_dir=input_dir,
            verbose=False,
            no_progress=True
        )
        
        # Verify the results
        assert len(results) == len(audio_files)
        
        # First and third files should be processed successfully
        assert results[0][1] is True  # success flag
        assert results[2][1] is True
        
        # Second file should fail
        assert results[1][1] is False
        assert results[1][2] is None  # No output path
        
        # All files should be processed since we're not skipping in advance anymore
        assert mock_generate_single.call_count == 3

    def test_nonexistent_input_dir(self):
        """Test error handling for nonexistent input directory."""
        nonexistent_dir = pathlib.Path("/nonexistent/directory")
        
        with pytest.raises(FileNotFoundError, match="Input directory not found"):
            generate_batch_videos(input_dir=nonexistent_dir, verbose=False) 