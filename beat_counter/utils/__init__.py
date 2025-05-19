"""
Utilities for beat detection and analysis.
"""

# Import utilities so they can be accessed directly from beat_detection.utils
from .beats_compare import (
    find_beats_files,
    normalize_path,
    find_matching_timestamps,
    compare_arrays,
    compare_beats_files
)
