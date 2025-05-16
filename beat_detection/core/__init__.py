"""
Beat detection core functionality.
"""

from beat_detection.core.registry import get as get_beat_detector
from beat_detection.core.beats import Beats, RawBeats

__all__ = ["get_beat_detector", "Beats", "RawBeats"]
