"""
Utilities for saving and loading beat data.
"""

from pathlib import Path
from beat_detection.core.beats import Beats  # Updated import

# Removed unused imports: numpy, Dict, Any, asdict, BeatStatistics


def save_beats(file_path: str, beats: Beats) -> None:
    """
    Save beat data to a file using the Beats object's method.
    
    Parameters:
    -----------
    file_path : str
        Path to output file (will be treated as JSON)
    beats : Beats
        Beat data to save
    """
    output_path = Path(file_path)
    beats.save_to_file(output_path)


def load_beats(file_path: str) -> Beats:
    """
    Load beat data from a file using the Beats object's class method.
    
    Parameters:
    -----------
    file_path : str
        Path to input JSON file
        
    Returns:
    --------
    Beats
        Loaded beat data
    """
    input_path = Path(file_path)
    return Beats.load_from_file(input_path)
