#!/usr/bin/env python
"""
Simple CLI tool to test the GenreDB implementation with the actual database file.
"""

from beat_detection.genre_db import GenreDB

def main():
    """Load and display genre information from the actual database file."""
    print("Loading GenreDB from default path...")
    db = GenreDB()
    
    # Get a list of all genres
    genres = list(db._data.keys())
    print(f"Found {len(genres)} genres in the database.")
    
    # Display some sample genres
    print("\nSample genres and their parameters:")
    for genre in genres[:5]:  # First 5 genres
        data = db._lookup_genre_raw(genre)
        print(f"\n{genre.title()}:")
        print(f"  Beats per bar: {data['beats_per_bar']}")
        print(f"  BPM range: {data['bpm_range']['min']}-{data['bpm_range']['max']}")
    
    # Test the helper methods
    print("\nTesting helper methods:")
    sample_genre = genres[0]
    
    print(f"\nbeats_kwargs_for_genre({sample_genre}):")
    print(db.beats_kwargs_for_genre(sample_genre))
    
    print(f"\ndetector_kwargs_for_genre({sample_genre}):")
    print(db.detector_kwargs_for_genre(sample_genre))
    
    # Test with existing values
    existing = {"min_bpm": 100, "max_bpm": 200}
    print(f"\ndetector_kwargs_for_genre({sample_genre}, existing={existing}):")
    print(db.detector_kwargs_for_genre(sample_genre, existing=existing))

if __name__ == "__main__":
    main() 