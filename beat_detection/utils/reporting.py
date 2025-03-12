"""
Reporting and statistics utilities.
"""

import numpy as np
import pathlib
from typing import List, Dict, Tuple, Any, Optional, Union
from ..core.detector import BeatStatistics


def save_beat_timestamps(timestamps: np.ndarray, output_file: Union[str, pathlib.Path], 
                      downbeats: Optional[np.ndarray] = None) -> None:
    """
    Save beat timestamps to a text file, optionally including downbeat information.
    
    Parameters:
    -----------
    timestamps : numpy.ndarray
        Array of beat timestamps in seconds
    output_file : str or pathlib.Path
        Path to the output file
    downbeats : numpy.ndarray, optional
        Array of indices that correspond to downbeats
    """
    if downbeats is None:
        # Just save timestamps if no downbeats provided
        np.savetxt(output_file, timestamps, fmt='%.3f')
    else:
        # Create a 2-column array with beat timestamps and downbeat flags
        # 1 = downbeat, 0 = regular beat
        downbeat_flags = np.zeros(len(timestamps), dtype=int)
        downbeat_flags[downbeats] = 1
        
        # Combine into a single array
        beat_data = np.column_stack((timestamps, downbeat_flags))
        
        # Save with a header
        with open(output_file, 'w') as f:
            f.write("# Beat timestamps in seconds with downbeat flags\n")
            f.write("# Format: timestamp downbeat_flag(1=yes,0=no)\n")
            np.savetxt(f, beat_data, fmt=['%.3f', '%d'])


def save_beat_statistics(stats: BeatStatistics, irregular_beats: List[int], 
                         output_file: Union[str, pathlib.Path], 
                         filename: Optional[str] = None) -> None:
    """
    Save beat statistics to a text file.
    
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
    """
    with open(output_file, 'w') as f:
        if filename:
            f.write(f"File: {filename}\n")
        f.write(f"Tempo: {stats.tempo_bpm:.1f} BPM\n")
        f.write(f"Mean interval: {stats.mean_interval:.3f} seconds\n")
        f.write(f"Median interval: {stats.median_interval:.3f} seconds\n")
        f.write(f"Standard deviation: {stats.std_interval:.3f} seconds\n")
        f.write(f"Min interval: {stats.min_interval:.3f} seconds\n")
        f.write(f"Max interval: {stats.max_interval:.3f} seconds\n")
        f.write(f"Irregular beats: {len(irregular_beats)} ({stats.irregularity_percent:.1f}%)\n")
        if irregular_beats:
            f.write("\nIrregular beats (indices):\n")
            f.write(", ".join(map(str, irregular_beats)))


def save_batch_summary(file_stats: List[Tuple[str, BeatStatistics]], 
                       output_file: Union[str, pathlib.Path]) -> None:
    """
    Save summary statistics for batch processing.
    
    Parameters:
    -----------
    file_stats : list of (str, BeatStatistics)
        List of tuples with filename and statistics
    output_file : str or pathlib.Path
        Path to the output file
    """
    if not file_stats:
        return
        
    # Calculate summary statistics
    tempos = [stats.tempo_bpm for _, stats in file_stats]
    avg_tempo = sum(tempos) / len(tempos)
    min_tempo = min(tempos)
    max_tempo = max(tempos)
    
    # Find file with most irregular beats
    most_irregular = max(file_stats, key=lambda x: x[1].irregularity_percent)
    
    with open(output_file, 'w') as f:
        f.write(f"Summary Statistics for {len(file_stats)} Files\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Average tempo: {avg_tempo:.1f} BPM\n")
        f.write(f"Tempo range: {min_tempo:.1f} - {max_tempo:.1f} BPM\n\n")
        f.write("Individual File Statistics:\n")
        f.write("-" * 50 + "\n")
        for filename, stats in file_stats:
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


def print_batch_summary(file_stats: List[Tuple[str, BeatStatistics]]) -> None:
    """
    Print summary statistics for batch processing.
    
    Parameters:
    -----------
    file_stats : list of (str, BeatStatistics)
        List of tuples with filename and statistics
    """
    if not file_stats:
        return
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS ACROSS ALL FILES")
    print("=" * 80)
    print(f"Total files processed: {len(file_stats)}")
    
    # Calculate average tempo
    tempos = [stats.tempo_bpm for _, stats in file_stats]
    avg_tempo = sum(tempos) / len(tempos)
    min_tempo = min(tempos)
    max_tempo = max(tempos)
    
    print(f"Average tempo: {avg_tempo:.1f} BPM")
    print(f"Tempo range: {min_tempo:.1f} - {max_tempo:.1f} BPM")
    
    # Show file with most irregular beats
    most_irregular = max(file_stats, key=lambda x: x[1].irregularity_percent)
    print(f"Most irregular file: {most_irregular[0]} ({most_irregular[1].irregularity_percent:.1f}% irregular beats)")