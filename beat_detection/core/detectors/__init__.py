"""
Beat detector implementations.

This package contains concrete implementations of the BeatDetector protocol.
"""

# Re-export detector classes for convenience
from beat_detection.core.detectors.madmom import MadmomBeatDetector
from beat_detection.core.detectors.beat_this import BeatThisDetector

__all__ = ["MadmomBeatDetector", "BeatThisDetector"] 