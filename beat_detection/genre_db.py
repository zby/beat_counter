"""
Genre-based default parameter support for beat detection tasks.

This module provides:
1. GenreDB - class for loading and accessing genre-specific parameters
2. parse_genre_from_path - helper to extract genre from file paths
"""

import csv
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union, TextIO
import io


def _parse_bpm_range(bpm_range_str: str) -> Dict[str, int]:
    """
    Parse a BPM range string in the format "min-max" into a dict with min and max keys.
    
    Args:
        bpm_range_str: String in format "min-max" (e.g., "120-140")
        
    Returns:
        Dictionary with "min" and "max" keys containing integer values
        
    Raises:
        ValueError: If the format is invalid
    """
    parts = bpm_range_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid BPM range format: '{bpm_range_str}'. Expected 'min-max'.")
    return {"min": int(parts[0].strip()), "max": int(parts[1].strip())}


def parse_genre_from_path(path: str) -> str:
    """
    Extract the genre from a path containing '/by_genre/<genre>/' pattern.
    
    Args:
        path: File path string to search for genre
        
    Returns:
        The extracted genre name
        
    Raises:
        ValueError: If the path doesn't match the expected pattern
    """
    match = re.search(r"/by_genre/([^/]+)/", path, re.IGNORECASE)
    if match:
        return match.group(1)
    raise ValueError(
        f"Path '{path}' does not match the expected genre pattern '.../by_genre/<genre>/...'"
    )


class GenreDB:
    """
    Provides genre-specific default parameters by reading from a CSV file or string.
    
    The database reads dance music genre information from a CSV with columns:
    - name: Genre name (e.g., "House", "Techno")
    - beats_per_bar: Typical number of beats per bar for the genre
    - bpm_range: Typical BPM range in format "min-max" (e.g., "120-135")
    """

    def __init__(self, csv_path: Optional[Path] = None, csv_content: Optional[str] = None):
        """
        Initialize the genre database from a CSV file or string content.
        
        Args:
            csv_path: Path to CSV file (optional). Defaults to "./data/dance_music_genres.csv"
                      relative to the module.
            csv_content: CSV content as a string (optional). If provided, csv_path is ignored.
        """
        self._data: Dict[str, Dict[str, Any]] = {}
        if csv_content is not None:
            # Prioritize csv_content if provided
            self._load(io.StringIO(csv_content))
        elif csv_path is None:
            # Default path relative to this file
            csv_path = Path(__file__).parent.parent / "data" / "dance_music_genres.csv"
            self._load(csv_path)
        else:
            # Use provided csv_path
            self._load(csv_path)

    def _load(self, source: Union[Path, TextIO]):
        """
        Load genre data from a file path or file-like object.
        
        Args:
            source: Path object or file-like object containing CSV data
        """
        self._data = {}
        if isinstance(source, Path):
            with source.open(newline="") as f:
                reader = csv.DictReader(f)
                self._parse_reader(reader)
        else:  # Assumed to be a TextIO object (like StringIO)
            reader = csv.DictReader(source)
            self._parse_reader(reader)

    def _parse_reader(self, reader: csv.DictReader):
        """
        Parse data from a CSV DictReader into the internal data dictionary.
        
        Args:
            reader: Initialized CSV DictReader object
        """
        for row in reader:
            # Only process rows that have the required fields
            if all(field in row for field in ["name", "beats_per_bar", "bpm_range"]):
                name = row["name"].strip().lower()
                self._data[name] = {
                    "beats_per_bar": int(row["beats_per_bar"]),
                    "bpm_range": _parse_bpm_range(row["bpm_range"]),
                }
                # Optionally store other columns (like 'characteristics') if needed later

    def _lookup_genre_raw(self, genre: str) -> Dict[str, Any]:
        """
        Look up a genre's raw data.
        
        Args:
            genre: Genre name to look up (case-insensitive)
            
        Returns:
            Dictionary containing the genre's parameters
            
        Raises:
            ValueError: If the genre is not found
        """
        key = genre.strip().lower()
        if key not in self._data:
            raise ValueError(f"Genre '{genre}' not found in metadata database")
        return self._data[key].copy()  # Return a copy to prevent mutation of cached data

    def beats_kwargs_for_genre(self, genre: str, existing: Optional[Dict] = None) -> Dict:
        """
        Get Beats constructor arguments for a genre.
        
        Returns a dictionary containing beats_per_bar for the given genre,
        if not already present in 'existing'.
        
        Args:
            genre: Genre name to look up (case-insensitive)
            existing: Optional existing dictionary to merge with
            
        Returns:
            Dictionary with beats_per_bar parameter (if not in existing)
        """
        if existing is None:
            existing = {}
        
        genre_defaults = self._lookup_genre_raw(genre)
        result = existing.copy()

        if "beats_per_bar" not in result:
            result["beats_per_bar"] = genre_defaults["beats_per_bar"]
        
        return result

    def detector_kwargs_for_genre(self, genre: str, existing: Optional[Dict] = None) -> Dict:
        """
        Get BeatDetector constructor arguments for a genre.
        
        Returns a dictionary containing min_bpm, max_bpm, and beats_per_bar for the given genre,
        if not already present in 'existing'.
        
        Args:
            genre: Genre name to look up (case-insensitive)
            existing: Optional existing dictionary to merge with
            
        Returns:
            Dictionary with min_bpm, max_bpm, beats_per_bar parameters (if not in existing)
        """
        if existing is None:
            existing = {}

        genre_defaults = self._lookup_genre_raw(genre)
        result = existing.copy()

        bpm_min, bpm_max = genre_defaults["bpm_range"]["min"], genre_defaults["bpm_range"]["max"]

        if "min_bpm" not in result:
            result["min_bpm"] = bpm_min
        if "max_bpm" not in result:
            result["max_bpm"] = bpm_max
        if "beats_per_bar" not in result:  # beats_per_bar is also relevant for detector
            result["beats_per_bar"] = genre_defaults["beats_per_bar"]
            
        return result 