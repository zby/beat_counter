"""
Beat detection core functionality.
"""

# Import detector modules to ensure they're registered
from beat_counter.core.detectors import madmom, beat_this

from beat_counter.core.registry import get as get_beat_detector
from beat_counter.core.pipeline import extract_beats, process_batch
from beat_counter.core.beats import Beats, RawBeats

__all__ = [
    "get_beat_detector", 
    "extract_beats", 
    "process_batch", 
    "Beats", 
    "RawBeats"
]
