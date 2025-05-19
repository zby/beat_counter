"""
Beat detector implementations.

This package contains concrete implementations of the BeatDetector protocol.
"""

# Import detector modules to ensure decorators are executed
from beat_counter.core.detectors import madmom, beat_this

# Re-export detector classes for convenience
from beat_counter.core.detectors.madmom import MadmomBeatDetector
from beat_counter.core.detectors.beat_this import BeatThisDetector

__all__ = ["MadmomBeatDetector", "BeatThisDetector"] 