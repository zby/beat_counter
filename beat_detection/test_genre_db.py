"""
Unit tests for GenreDB class and genre utilities.
Run with: python -m pytest beat_detection/test_genre_db.py -v
"""

import pytest
from pathlib import Path
import tempfile
from beat_detection.genre_db import GenreDB, parse_genre_from_path


class TestParseGenreFromPath:
    """Tests for parse_genre_from_path function."""
    
    def test_parse_genre_success(self):
        """Should correctly extract genre from valid paths."""
        paths = [
            "/music/by_genre/House/track.wav",
            "/by_genre/Techno/nested/file.mp3",
            "my_music/BY_GENRE/Dubstep/track01.flac",  # Case insensitive
            "random/path/with/by_genre/Drum & Bass/file.ogg"
        ]
        
        expected_genres = ["House", "Techno", "Dubstep", "Drum & Bass"]
        
        for path, expected in zip(paths, expected_genres):
            assert parse_genre_from_path(path) == expected
    
    def test_parse_genre_failure(self):
        """Should raise ValueError for paths without valid genre pattern."""
        invalid_paths = [
            "/music/genres/House/track.wav",  # No "by_genre"
            "/by_genre_not_delimited/Techno/file.mp3",  # Wrong format
            "my_music/by_genre/track01.flac",  # Missing genre segment
            "random/path"  # No pattern at all
        ]
        
        for path in invalid_paths:
            with pytest.raises(ValueError):
                parse_genre_from_path(path)


class TestGenreDB:
    """Tests for GenreDB class."""
    
    @pytest.fixture
    def sample_csv_content(self):
        """Sample CSV content for testing, matching the actual CSV format."""
        return (
            "id,name,beats_per_bar,bpm_range,characteristics\n"
            "1,House,4,120-130,\"Four-on-the-floor rhythm, electronic\"\n"
            "2,Techno,4,120-140,\"Repetitive, machine-like sound\"\n"
            "3,Drum & Bass,4,170-180,\"Fast breakbeats with heavy bass\"\n"
            "4,Dubstep,4,140-150,\"Half-time feel, heavy bass drops\"\n"
        )
    
    @pytest.fixture
    def genre_db_from_content(self, sample_csv_content):
        """Return GenreDB instance initialized with sample content."""
        return GenreDB(csv_content=sample_csv_content)
    
    def test_create_with_csv_content(self, sample_csv_content):
        """Should initialize successfully with csv_content."""
        db = GenreDB(csv_content=sample_csv_content)
        
        # Test internal data structure for a few genres
        assert "house" in db._data  # Keys are lowercased
        assert db._data["house"]["beats_per_bar"] == 4
        assert db._data["house"]["bpm_range"]["min"] == 120
        assert db._data["house"]["bpm_range"]["max"] == 130
        
        assert "techno" in db._data
        assert db._data["techno"]["bpm_range"]["min"] == 120
        assert db._data["techno"]["bpm_range"]["max"] == 140
    
    def test_create_with_csv_path(self, sample_csv_content):
        """Should initialize successfully with csv_path."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as temp_file:
            temp_file.write(sample_csv_content)
        
        try:
            # Test with explicit path
            db = GenreDB(csv_path=Path(temp_file.name))
            
            # Verify data was loaded
            assert "dubstep" in db._data
            assert db._data["dubstep"]["beats_per_bar"] == 4
            assert db._data["dubstep"]["bpm_range"]["min"] == 140
            assert db._data["dubstep"]["bpm_range"]["max"] == 150
        finally:
            # Clean up temp file
            Path(temp_file.name).unlink()
    
    def test_content_priority_over_path(self, sample_csv_content):
        """csv_content should be prioritized over csv_path."""
        # Create a different CSV file that shouldn't be used
        different_content = "id,name,beats_per_bar,bpm_range,characteristics\n1,Trance,4,130-150,\"Hypnotic, repetitive\"\n"
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as temp_file:
            temp_file.write(different_content)
        
        try:
            # Initialize with both path and content
            db = GenreDB(
                csv_path=Path(temp_file.name),
                csv_content=sample_csv_content
            )
            
            # Should use content (which has House but not Trance)
            assert "house" in db._data
            assert "trance" not in db._data
        finally:
            # Clean up temp file
            Path(temp_file.name).unlink()
    
    def test_lookup_genre_error(self, genre_db_from_content):
        """Should raise ValueError for unknown genre."""
        with pytest.raises(ValueError, match="not found in metadata database"):
            genre_db_from_content._lookup_genre_raw("Unknown")
    
    def test_beats_kwargs_for_genre(self, genre_db_from_content):
        """beats_kwargs_for_genre should return correct parameters."""
        # Basic case - no existing dict
        result = genre_db_from_content.beats_kwargs_for_genre("House")
        assert result == {"beats_per_bar": 4}
        assert isinstance(result["beats_per_bar"], int), "beats_per_bar should be an integer"
        
        # With empty existing dict
        result = genre_db_from_content.beats_kwargs_for_genre("House", existing={})
        assert result == {"beats_per_bar": 4}
        assert isinstance(result["beats_per_bar"], int), "beats_per_bar should be an integer"
        
        # With existing dict containing other keys
        result = genre_db_from_content.beats_kwargs_for_genre(
            "House", 
            existing={"some_other_key": "value"}
        )
        assert result == {"some_other_key": "value", "beats_per_bar": 4}
        assert isinstance(result["beats_per_bar"], int), "beats_per_bar should be an integer"
        
        # With existing dict containing beats_per_bar (should override)
        result = genre_db_from_content.beats_kwargs_for_genre(
            "House", 
            existing={"beats_per_bar": 3}  # Different value as integer
        )
        assert result == {"beats_per_bar": 4}  # Genre default overrides existing value
        
        # With existing dict containing beats_per_bar as a list (should override)
        result = genre_db_from_content.beats_kwargs_for_genre(
            "House", 
            existing={"beats_per_bar": [3]}  # Different value as a list
        )
        assert result == {"beats_per_bar": 4}  # Genre default overrides existing value
    
    def test_detector_kwargs_for_genre(self, genre_db_from_content):
        """detector_kwargs_for_genre should return correct parameters."""
        # Basic case - no existing dict
        result = genre_db_from_content.detector_kwargs_for_genre("Dubstep")
        assert result == {"min_bpm": 140, "max_bpm": 150, "beats_per_bar": [4]}
        assert isinstance(result["beats_per_bar"], list), "beats_per_bar should be a list"
        
        # With empty existing dict
        result = genre_db_from_content.detector_kwargs_for_genre("Dubstep", existing={})
        assert result == {"min_bpm": 140, "max_bpm": 150, "beats_per_bar": [4]}
        assert isinstance(result["beats_per_bar"], list), "beats_per_bar should be a list"
        
        # With existing dict containing other keys
        result = genre_db_from_content.detector_kwargs_for_genre(
            "Dubstep", 
            existing={"some_other_key": "value"}
        )
        expected = {
            "some_other_key": "value",
            "min_bpm": 140,
            "max_bpm": 150,
            "beats_per_bar": [4]
        }
        assert result == expected
        assert isinstance(result["beats_per_bar"], list), "beats_per_bar should be a list"
        
        # With existing dict containing some overlapping keys (should override)
        result = genre_db_from_content.detector_kwargs_for_genre(
            "Dubstep", 
            existing={
                "min_bpm": 135,  # Different value
                "other_param": True
            }
        )
        expected = {
            "min_bpm": 140,  # Overridden by genre default
            "max_bpm": 150,  # Added from genre
            "beats_per_bar": [4],  # Added from genre
            "other_param": True  # Preserved
        }
        assert result == expected
        
        # With existing dict containing all overlapping keys (should override all)
        result = genre_db_from_content.detector_kwargs_for_genre(
            "Dubstep", 
            existing={
                "min_bpm": 135,
                "max_bpm": 160,
                "beats_per_bar": [6],
                "other_param": True
            }
        )
        expected = {
            "min_bpm": 140,  # Overridden by genre default
            "max_bpm": 150,  # Overridden by genre default
            "beats_per_bar": [4],  # Overridden by genre default
            "other_param": True  # Preserved
        }
        assert result == expected 