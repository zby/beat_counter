"""
Utilities for saving and loading beat data.
"""

import numpy as np
from typing import Dict, Any
from dataclasses import asdict
from beat_detection.core.detector import Beats, BeatStatistics


def save_beats(file_path: str, beats: Beats) -> None:
    """
    Save beat data to a file.
    
    Parameters:
    -----------
    file_path : str
        Path to output file
    beats : Beats
        Beat data to save
    """
    # Convert to dictionary
    data = {
        'timestamps': beats.timestamps.tolist(),
        'downbeats': beats.downbeats.tolist(),
        'meter': beats.meter,
        'intro_end_idx': beats.intro_end_idx,
        'ending_start_idx': beats.ending_start_idx,
        'irregular_beats': beats.irregular_beats,
        'stats': asdict(beats.stats)
    }
    
    # Save as numpy file
    np.save(file_path, data)


def load_beats(file_path: str) -> Beats:
    """
    Load beat data from a file.
    
    Parameters:
    -----------
    file_path : str
        Path to input file
        
    Returns:
    --------
    Beats
        Loaded beat data
    """
    # Load data
    data = np.load(file_path, allow_pickle=True).item()
    
    # Convert back to Beats object
    return Beats(
        timestamps=np.array(data['timestamps']),
        downbeats=np.array(data['downbeats']),
        meter=data['meter'],
        intro_end_idx=data['intro_end_idx'],
        ending_start_idx=data['ending_start_idx'],
        stats=BeatStatistics(**data['stats']),
        irregular_beats=data['irregular_beats']
    )
