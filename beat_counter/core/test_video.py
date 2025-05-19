"""
Unit tests for video.py module.

These tests verify that the video generation functions work correctly,
with a focus on file handling, error conditions, and parameter passing.
"""
import pathlib
import pytest
from unittest.mock import patch, MagicMock, call

from beat_detection.core.video import (
    BeatVideoGenerator,
    generate_single_video_from_files,
    generate_batch_videos,
)
from beat_detection.core.beats import Beats, RawBeats


class TestGenerateSingleVideoFromFiles:
    """Test the generate_single_video_from_files function."""

    @pytest.fixture
    def mock_paths(self, tmp_path):
        """Create temporary paths for tests."""
        audio_file = tmp_path / "test_audio.mp3"
        beats_file = tmp_path / "test_audio.beats"
        output_file = tmp_path / "output_video.mp4"
        # Create empty files
        audio_file.write_text("mock audio content")
        beats_file.write_text('{"timestamps": [1.0, 2.0, 3.0], "audio_file": "test_audio.mp3"}')
        return audio_file, beats_file, output_file

    @patch("beat_detection.core.video.RawBeats.load_from_file")
    @patch("beat_detection.core.video.Beats")
    @patch("beat_detection.core.video.BeatVideoGenerator")
    def test_successful_generation(self, mock_generator_class, mock_beats_class, mock_load, mock_paths):
        """Test successful video generation with default parameters."""
        audio_file, beats_file, output_file = mock_paths
        
        # Configure mocks
        mock_raw_beats = MagicMock()
        mock_raw_beats.timestamps = [1.0, 2.0, 3.0]
        mock_load.return_value = mock_raw_beats
        
        mock_beats = MagicMock()
        mock_beats_class.return_value = mock_beats
        
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
        mock_load.assert_called_once_with(beats_file)
        mock_beats_class.assert_called_once_with(
            raw_beats=mock_raw_beats,
            beats_per_bar=None,
            tolerance_percent=10.0,
            min_measures=5
        )
        mock_generator_class.assert_called_once()
        mock_generator.generate_video.assert_called_once_with(
            audio_path=audio_file,
            beats=mock_beats,
            output_path=output_file,
            sample_beats=None
        )

    @patch("beat_detection.core.video.RawBeats.load_from_file")
    @patch("beat_detection.core.video.Beats")
    @patch("beat_detection.core.video.BeatVideoGenerator")
    def test_custom_parameters(self, mock_generator_class, mock_beats_class, mock_load, mock_paths):
        """Test video generation with custom parameters."""
        audio_file, beats_file, output_file = mock_paths
        
        # Configure mocks
        mock_raw_beats = MagicMock()
        mock_raw_beats.timestamps = [1.0, 2.0, 3.0]
        mock_load.return_value = mock_raw_beats
        
        mock_beats = MagicMock()
        mock_beats_class.return_value = mock_beats
        
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
        mock_beats_class.assert_called_once_with(
            raw_beats=mock_raw_beats,
            beats_per_bar=None,
            tolerance_percent=custom_tolerance,
            min_measures=custom_min_measures
        )
        mock_generator_class.assert_called_once_with(
            resolution=custom_resolution, 
            fps=custom_fps
        )
        mock_generator.generate_video.assert_called_once_with(
            audio_path=audio_file,
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
        
        # Test beats file not found
        with pytest.raises(FileNotFoundError, match="Beats file not found"):
            generate_single_video_from_files(
                audio_file=audio_file,
                beats_file=beats_file,
                verbose=False
            )

    @patch("beat_detection.core.video.RawBeats.load_from_file")
    def test_raw_beats_loading_error(self, mock_load, mock_paths):
        """Test error handling when RawBeats loading fails."""
        audio_file, beats_file, _ = mock_paths
        
        # Configure mock to raise an exception
        mock_load.side_effect = Exception("Failed to load beats")
        
        # Test exception handling
        with pytest.raises(RuntimeError, match="Failed to load raw beats"):
            generate_single_video_from_files(
                audio_file=audio_file,
                beats_file=beats_file,
                verbose=False
            )

    @patch("beat_detection.core.video.RawBeats.load_from_file")
    @patch("beat_detection.core.video.Beats")
    def test_beats_reconstruction_error(self, mock_beats_class, mock_load, mock_paths):
        """Test error handling when Beats reconstruction fails."""
        audio_file, beats_file, _ = mock_paths
        
        # Configure mocks
        mock_raw_beats = MagicMock()
        mock_load.return_value = mock_raw_beats
        
        # Make Beats constructor fail
        mock_beats_class.side_effect = Exception("Failed to reconstruct")
        
        # Test exception handling
        with pytest.raises(RuntimeError, match="Failed to reconstruct Beats object"):
            generate_single_video_from_files(
                audio_file=audio_file,
                beats_file=beats_file,
                verbose=False
            )

    @patch("beat_detection.core.video.RawBeats.load_from_file")
    @patch("beat_detection.core.video.Beats")
    @patch("beat_detection.core.video.BeatVideoGenerator")
    def test_default_output_path(self, mock_generator_class, mock_beats_class, mock_load, mock_paths):
        """Test that default output path is correctly constructed."""
        audio_file, beats_file, _ = mock_paths
        
        # Configure mocks
        mock_raw_beats = MagicMock()
        mock_load.return_value = mock_raw_beats
        
        mock_beats = MagicMock()
        mock_beats_class.return_value = mock_beats
        
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
            audio_path=audio_file,
            beats=mock_beats,
            output_path=expected_default_output,
            sample_beats=None
        )


class TestGenerateBatchVideos:
    """Test the generate_batch_videos function."""

    @pytest.fixture
    def mock_directory_structure(self, tmp_path):
        """Create mock directory structure with audio and beats files."""
        # Create directory structure
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create a subdirectory
        subdir = input_dir / "subdir"
        subdir.mkdir()
        
        # Create audio and beats files
        audio_files = [
            input_dir / "test1.mp3",
            input_dir / "test2.mp3",
            subdir / "test3.mp3"
        ]
        
        beats_files = [
            input_dir / "test1.beats",
            input_dir / "test2.beats",
            subdir / "test3.beats"
        ]
        
        # Create empty files
        for audio_file in audio_files:
            audio_file.write_text("mock audio content")
        
        for beats_file in beats_files:
            beats_file.write_text('{"timestamps": [1.0, 2.0, 3.0], "audio_file": "' + str(beats_file).replace(".beats", ".mp3") + '"}')
        
        return input_dir

    @patch("beat_detection.core.video.find_audio_files")
    @patch("beat_detection.core.video.generate_single_video_from_files")
    def test_batch_processing(self, mock_generate_single, mock_find_audio, mock_directory_structure):
        """Test batch processing of multiple files."""
        input_dir = mock_directory_structure
        
        # Configure mocks
        audio_files = [
            input_dir / "test1.mp3",
            input_dir / "test2.mp3",
            input_dir / "subdir" / "test3.mp3"
        ]
        mock_find_audio.return_value = audio_files
        
        # Mock output paths for the generated videos
        output_paths = [
            input_dir / "test1_counter.mp4",
            input_dir / "test2_counter.mp4",
            input_dir / "subdir" / "test3_counter.mp4"
        ]
        
        # Make generate_single_video_from_files return the expected output paths
        mock_generate_single.side_effect = output_paths
        
        # Call the batch function
        results = generate_batch_videos(
            input_dir=input_dir,
            output_dir=None,  # Use default paths
            verbose=True,
            no_progress=True
        )
        
        # Verify results - should have 3 successful results
        assert len(results) == 3
        for i, (audio_path, success, output_path) in enumerate(results):
            assert success is True
            assert output_path == output_paths[i]
        
        # Verify generate_single_video_from_files was called for each file
        assert mock_generate_single.call_count == 3
        mock_generate_single.assert_has_calls([
            call(
                audio_file=audio_files[0],
                beats_file=audio_files[0].with_suffix(".beats"),
                output_file=None,  # Default
                resolution=(720, 540),  # Default
                fps=100,  # Default
                sample_beats=None,
                tolerance_percent=10.0,
                min_measures=5,
                verbose=True
            ),
            call(
                audio_file=audio_files[1],
                beats_file=audio_files[1].with_suffix(".beats"),
                output_file=None,
                resolution=(720, 540),
                fps=100,
                sample_beats=None,
                tolerance_percent=10.0,
                min_measures=5,
                verbose=True
            ),
            call(
                audio_file=audio_files[2],
                beats_file=audio_files[2].with_suffix(".beats"),
                output_file=None,
                resolution=(720, 540),
                fps=100,
                sample_beats=None,
                tolerance_percent=10.0,
                min_measures=5,
                verbose=True
            )
        ], any_order=True)

    @patch("beat_detection.core.video.find_audio_files")
    @patch("beat_detection.core.video.generate_single_video_from_files")
    def test_batch_with_custom_output_dir(self, mock_generate_single, mock_find_audio, mock_directory_structure, tmp_path):
        """Test batch processing with custom output directory."""
        input_dir = mock_directory_structure
        output_dir = tmp_path / "output"
        
        # Configure mocks
        audio_files = [
            input_dir / "test1.mp3",
            input_dir / "test2.mp3",
            input_dir / "subdir" / "test3.mp3"
        ]
        mock_find_audio.return_value = audio_files
        
        # Expected output paths with custom output directory
        output_paths = [
            output_dir / "test1_counter.mp4",
            output_dir / "test2_counter.mp4",
            output_dir / "subdir" / "test3_counter.mp4"
        ]
        mock_generate_single.side_effect = output_paths
        
        # Call the batch function with custom output directory
        results = generate_batch_videos(
            input_dir=input_dir,
            output_dir=output_dir,
            verbose=False,
            no_progress=True
        )
        
        # Verify each call included the correct output path
        output_paths_in_calls = [call[1]['output_file'] for call in mock_generate_single.call_args_list]
        expected_output_files = [
            output_dir / "test1_counter.mp4",
            output_dir / "test2_counter.mp4",
            output_dir / "subdir" / "test3_counter.mp4"
        ]
        
        # Check that each file has the correct output directory
        for actual, expected in zip(output_paths_in_calls, expected_output_files):
            assert actual == expected

    @patch("beat_detection.core.video.find_audio_files")
    def test_no_audio_files(self, mock_find_audio, mock_directory_structure):
        """Test behavior when no audio files are found."""
        input_dir = mock_directory_structure
        
        # Configure mock to return empty list
        mock_find_audio.return_value = []
        
        # Call the batch function
        results = generate_batch_videos(
            input_dir=input_dir,
            verbose=False,
            no_progress=True
        )
        
        # Should return empty list
        assert results == []

    @patch("beat_detection.core.video.find_audio_files")
    @patch("beat_detection.core.video.generate_single_video_from_files")
    def test_handle_missing_beats_file(self, mock_generate_single, mock_find_audio, mock_directory_structure):
        """Test that missing beats files are properly handled."""
        input_dir = mock_directory_structure
        
        # Configure mocks
        audio_files = [
            input_dir / "test1.mp3",
            input_dir / "test_no_beats.mp3",  # This one doesn't have a beats file
        ]
        mock_find_audio.return_value = audio_files
        
        # Make the first call succeed but second raise FileNotFoundError
        mock_generate_single.side_effect = [
            input_dir / "test1_counter.mp4",  # Success
            FileNotFoundError("Beats file not found")  # Failure
        ]
        
        # Call the batch function
        results = generate_batch_videos(
            input_dir=input_dir,
            verbose=False,
            no_progress=True
        )
        
        # Should have two results, one success and one failure
        assert len(results) == 2
        assert results[0][1] is True  # First file succeeded
        assert results[1][1] is False  # Second file failed
        
        # Only called generate_single_video_from_files once (second call caused exception)
        assert mock_generate_single.call_count == 2

    def test_nonexistent_input_dir(self):
        """Test error handling when input directory doesn't exist."""
        nonexistent_dir = pathlib.Path("/nonexistent/directory")
        
        with pytest.raises(FileNotFoundError, match="Input directory not found"):
            generate_batch_videos(
                input_dir=nonexistent_dir,
                verbose=False
            ) 