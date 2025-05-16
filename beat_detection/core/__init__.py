"""
Beat detection core functionality.
"""

# Import detector modules to ensure they're registered
from beat_detection.core.detectors import madmom, beat_this

from beat_detection.core.registry import get as get_beat_detector
from beat_detection.core.beats import Beats, RawBeats

__all__ = ["get_beat_detector", "Beats", "RawBeats"]
