"""
Utilities for saving and loading beat data.
"""

from pathlib import Path
from beat_detection.core.beats import Beats, RawBeats  # Updated import

# Removed unused imports: numpy, Dict, Any, asdict, BeatStatistics


def save_beats(file_path: str, raw_beats_data: RawBeats) -> None:
    """
    Save raw beat data (timestamps, counts) to a JSON file.

    Parameters:
    -----------
    file_path : str
        Path to output file (will be treated as JSON)
    raw_beats_data : RawBeats
        RawBeats object containing the data to save.
    """
    output_path = Path(file_path)
    # Save using RawBeats method
    raw_beats_data.save_to_file(output_path)


def load_raw_beats(file_path: str) -> RawBeats:
    """
    Load raw beat data (timestamps, counts) from a JSON file.

    Parameters:
    -----------
    file_path : str
        Path to input JSON file

    Returns:
    --------
    RawBeats
        Loaded raw beat data.

    Raises:
    -------
    FileNotFoundError:
        If the file_path does not exist.
    ValueError:
        If the file is not valid JSON or missing required keys/structure.
    """
    input_path = Path(file_path)
    # Directly use RawBeats loading method
    return RawBeats.load_from_file(input_path)
