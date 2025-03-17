"""
Reporting and statistics utilities.
"""

import numpy as np
import pathlib
import json
from typing import List, Dict, Tuple, Any, Optional, Union
from ..core.detector import BeatStatistics


def save_beat_timestamps(timestamps: np.ndarray, output_file: Union[str, pathlib.Path], 
                      downbeats: Optional[np.ndarray] = None,
                      intro_end_idx: int = 0,
                      ending_start_idx: Optional[int] = None,
                      detected_meter: Optional[int] = None) -> None:
    """
    Save beat timestamps to a text file, including downbeat, intro, ending, and meter information.
    
    Parameters:
    -----------
    timestamps : numpy.ndarray
        Array of beat timestamps in seconds
    output_file : str or pathlib.Path
        Path to the output file
    downbeats : numpy.ndarray, optional
        Array of indices that correspond to downbeats
    intro_end_idx : int, optional
        Index where the intro ends (0 if no intro detected)
    ending_start_idx : int, optional
        Index where the ending begins (len(timestamps) if no ending detected)
    detected_meter : int, optional
        Detected meter (time signature numerator, typically 2, 3, or 4)
    """
    from .beat_file import save_beat_data
    
    # Convert pathlib.Path to string if needed
    if not isinstance(output_file, str):
        output_file = str(output_file)
    
    # Use the dedicated beat_file module to save the data
    save_beat_data(timestamps, output_file, downbeats, intro_end_idx, ending_start_idx, detected_meter)


def get_beat_statistics_dict(stats: BeatStatistics, irregular_beats: List[int], 
                            filename: Optional[str] = None,
                            detected_meter: Optional[int] = None,
                            duration: Optional[float] = None) -> Dict[str, Any]:
    """
    Convert beat statistics to a dictionary for JSON serialization.
    
    Parameters:
    -----------
    stats : BeatStatistics
        Beat statistics object
    irregular_beats : list of int
        List of indices of irregular beats
    filename : str, optional
        Original filename to include in the report
    detected_meter : int, optional
        Detected meter (time signature numerator)
    duration : float, optional
        Duration of the audio in seconds
        
    Returns:
    --------
    dict
        Dictionary containing beat statistics
    """
    stats_dict = {
        "tempo_bpm": round(stats.tempo_bpm, 1),
        "mean_interval": round(stats.mean_interval, 3),
        "median_interval": round(stats.median_interval, 3),
        "std_interval": round(stats.std_interval, 3),
        "min_interval": round(stats.min_interval, 3),
        "max_interval": round(stats.max_interval, 3),
        "irregular_beats_count": len(irregular_beats),
        "irregularity_percent": round(stats.irregularity_percent, 1),
        "irregular_beat_indices": irregular_beats
    }
    
    if filename:
        stats_dict["filename"] = filename
        
    if detected_meter is not None:
        stats_dict["detected_meter"] = detected_meter
        
    if duration is not None:
        stats_dict["duration"] = round(duration, 3)
    
    return stats_dict


def save_beat_statistics(stats: BeatStatistics, irregular_beats: List[int], 
                         output_file: Union[str, pathlib.Path], 
                         filename: Optional[str] = None,
                         detected_meter: Optional[int] = None,
                         duration: Optional[float] = None) -> None:
    """
    Save beat statistics to a JSON file.
    
    Parameters:
    -----------
    stats : BeatStatistics
        Beat statistics object
    irregular_beats : list of int
        List of indices of irregular beats
    output_file : str or pathlib.Path
        Path to the output file
    filename : str, optional
        Original filename to include in the report
    detected_meter : int, optional
        Detected meter (time signature numerator)
    duration : float, optional
        Duration of the audio in seconds
    """
    # Convert stats to a dictionary and save as JSON
    stats_dict = get_beat_statistics_dict(stats, irregular_beats, filename, detected_meter, duration)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(stats_dict, f, indent=2)


def save_batch_summary(file_stats: List[Tuple[str, Tuple[BeatStatistics, List[int]]]], 
                       output_file: Union[str, pathlib.Path]) -> None:
    """
    Save summary statistics for batch processing.
    
    Parameters:
    -----------
    file_stats : list of (str, (BeatStatistics, List[int]))
        List of tuples with filename and a tuple of (statistics, irregular_beats)
    output_file : str or pathlib.Path
        Path to the output file
    """
    if not file_stats:
        return
    
    # Filter out any None results
    valid_stats = [(filename, stats_tuple) for filename, stats_tuple in file_stats if stats_tuple is not None]
    
    if not valid_stats:
        with open(output_file, 'w') as f:
            f.write("No valid statistics available for any processed files.\n")
        return
        
    # Calculate summary statistics
    tempos = [stats_tuple[0].tempo_bpm for _, stats_tuple in valid_stats]
    avg_tempo = sum(tempos) / len(tempos)
    min_tempo = min(tempos)
    max_tempo = max(tempos)
    
    # Find file with most irregular beats
    most_irregular = max(valid_stats, key=lambda x: x[1][0].irregularity_percent)
    
    with open(output_file, 'w') as f:
        f.write(f"Summary Statistics for {len(valid_stats)} Files\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Average tempo: {avg_tempo:.1f} BPM\n")
        f.write(f"Tempo range: {min_tempo:.1f} - {max_tempo:.1f} BPM\n\n")
        f.write("Individual File Statistics:\n")
        f.write("-" * 50 + "\n")
        for filename, (stats, _) in valid_stats:
            f.write(f"{filename}: {stats.tempo_bpm:.1f} BPM, {stats.irregularity_percent:.1f}% irregular\n")


def print_beat_timestamps(timestamps: np.ndarray, irregular_beats: List[int] = None, 
                         downbeats: Optional[np.ndarray] = None) -> None:
    """
    Print beat timestamps to the console.
    
    Parameters:
    -----------
    timestamps : numpy.ndarray
        Array of beat timestamps in seconds
    irregular_beats : list of int, optional
        List of indices of irregular beats to mark
    downbeats : numpy.ndarray, optional
        Array of indices that correspond to downbeats
    """
    print("Detected Beats (in seconds):")
    for i, beat in enumerate(timestamps, 1):
        # Mark irregular beats and downbeats if provided
        irregular_mark = " (irregular)" if irregular_beats and i in irregular_beats else ""
        downbeat_mark = " (DOWNBEAT)" if downbeats is not None and (i-1) in downbeats else ""
        
        print(f"Beat {i}: {beat:.3f} seconds{irregular_mark}{downbeat_mark}")


def print_statistics(stats: BeatStatistics, irregular_beats: List[int]) -> None:
    """
    Print beat statistics to the console.
    
    Parameters:
    -----------
    stats : BeatStatistics
        Beat statistics object
    irregular_beats : list of int
        List of indices of irregular beats
    """
    print("\nBeat Statistics:")
    print(f"Tempo: {stats.tempo_bpm:.1f} BPM")
    print(f"Mean interval: {stats.mean_interval:.3f} seconds")
    print(f"Median interval: {stats.median_interval:.3f} seconds")
    print(f"Standard deviation: {stats.std_interval:.3f} seconds")
    print(f"Min interval: {stats.min_interval:.3f} seconds")
    print(f"Max interval: {stats.max_interval:.3f} seconds")
    print(f"Irregular beats: {len(irregular_beats)} ({stats.irregularity_percent:.1f}%)")


def print_batch_summary(file_stats: List[Tuple[str, Tuple[BeatStatistics, List[int]]]]) -> None:
    """
    Print summary statistics for batch processing.
    
    Parameters:
    -----------
    file_stats : list of (str, (BeatStatistics, List[int]))
        List of tuples with filename and a tuple of (statistics, irregular_beats)
    """
    if not file_stats:
        return
    
    # Filter out any None results
    valid_stats = [(filename, stats_tuple) for filename, stats_tuple in file_stats if stats_tuple is not None]
    
    if not valid_stats:
        print("\nNo valid statistics available for any processed files.")
        return
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS ACROSS ALL FILES")
    print("=" * 80)
    print(f"Total files processed successfully: {len(valid_stats)} out of {len(file_stats)}")
    
    # Calculate average tempo
    tempos = [stats_tuple[0].tempo_bpm for _, stats_tuple in valid_stats]
    avg_tempo = sum(tempos) / len(tempos)
    min_tempo = min(tempos)
    max_tempo = max(tempos)
    
    print(f"Average tempo: {avg_tempo:.1f} BPM")
    print(f"Tempo range: {min_tempo:.1f} - {max_tempo:.1f} BPM")
    
    # Show file with most irregular beats
    most_irregular = max(valid_stats, key=lambda x: x[1][0].irregularity_percent)
    print(f"Most irregular file: {most_irregular[0]} ({most_irregular[1][0].irregularity_percent:.1f}% irregular beats)")