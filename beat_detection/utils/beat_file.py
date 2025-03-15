"""
Utilities for loading and saving beat data files.
"""

import numpy as np
from typing import Tuple, Optional


def save_beat_data(timestamps: np.ndarray, output_file: str,
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
    output_file : str
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
    # Set default ending_start_idx if not provided
    if ending_start_idx is None:
        ending_start_idx = len(timestamps)
    
    if downbeats is None:
        # Create a simple header with intro, ending, and meter information
        with open(output_file, 'w') as f:
            f.write(f"# Beat timestamps in seconds\n")
            f.write(f"# INTRO_END_IDX={intro_end_idx}\n")
            f.write(f"# ENDING_START_IDX={ending_start_idx}\n")
            if detected_meter is not None:
                f.write(f"# DETECTED_METER={detected_meter}\n")
            np.savetxt(f, timestamps, fmt='%.3f')
    else:
        # Create a 2-column array with beat timestamps and downbeat flags
        # 1 = downbeat, 0 = regular beat
        downbeat_flags = np.zeros(len(timestamps), dtype=int)
        downbeat_flags[downbeats] = 1
        
        # Combine into a single array
        beat_data = np.column_stack((timestamps, downbeat_flags))
        
        # Save with a header that includes intro, ending, and meter information
        with open(output_file, 'w') as f:
            f.write("# Beat timestamps in seconds with downbeat flags\n")
            f.write("# Format: timestamp downbeat_flag(1=yes,0=no)\n")
            f.write(f"# INTRO_END_IDX={intro_end_idx}\n")
            f.write(f"# ENDING_START_IDX={ending_start_idx}\n")
            if detected_meter is not None:
                f.write(f"# DETECTED_METER={detected_meter}\n")
            np.savetxt(f, beat_data, fmt=['%.3f', '%d'])


def load_beat_data(beats_file: str) -> Tuple[np.ndarray, np.ndarray, int, int, Optional[int]]:
    """
    Load beat data from a text file with numpy's savetxt format.
    
    Parameters:
    -----------
    beats_file : str
        Path to the beat data file
    
    Returns:
    --------
    Tuple containing:
        - beat_timestamps (np.ndarray): Array of beat timestamps in seconds
        - downbeats (np.ndarray): Array of indices that correspond to downbeats
        - intro_end_idx (int): Index where the intro ends
        - ending_start_idx (int): Index where the ending begins
        - detected_meter (Optional[int]): Detected meter (time signature numerator)
    """
    intro_end_idx = 0
    ending_start_idx = None
    detected_meter = None
    
    # Read the file and parse the header information
    with open(beats_file, 'r') as f:
        lines = f.readlines()
        
        # Extract metadata from comment lines
        for line in lines:
            if line.startswith('#'):
                # Parse intro end index
                if 'INTRO_END_IDX=' in line:
                    intro_end_idx = int(line.split('=')[1].strip())
                # Parse ending start index
                elif 'ENDING_START_IDX=' in line:
                    ending_start_idx = int(line.split('=')[1].strip())
                # Parse detected meter
                elif 'DETECTED_METER=' in line:
                    detected_meter = int(line.split('=')[1].strip())
            else:
                break
        
        # Load the data (skip comment lines)
        data = np.loadtxt(beats_file, comments='#')
    
    # Check if we have a 2-column array (timestamps and downbeat flags)
    if len(data.shape) > 1 and data.shape[1] > 1:
        # Extract timestamps and downbeat flags
        timestamps = data[:, 0]
        downbeat_flags = data[:, 1].astype(int)
        downbeats = np.where(downbeat_flags == 1)[0]
    else:
        # Only timestamps, no downbeats
        timestamps = data
        downbeats = np.array([])
    
    # If ending_start_idx is None, set it to the length of timestamps
    if ending_start_idx is None:
        ending_start_idx = len(timestamps)
    
    return timestamps, downbeats, intro_end_idx, ending_start_idx, detected_meter
