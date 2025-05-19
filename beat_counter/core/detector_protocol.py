"""
Core beat detection functionality and protocols.
"""

from typing import Protocol, runtime_checkable
from pathlib import Path
from beat_detection.core.beats import RawBeats


@runtime_checkable
class BeatDetector(Protocol):
    """Protocol for beat detection algorithms."""

    def detect_beats(self, audio_path: str | Path) -> RawBeats:
        """
        Detects beats in an audio file and returns raw beat information.

        Parameters:
        -----------
        audio_path : str | Path
            Path to the input audio file.

        Returns:
        --------
        RawBeats
            Object containing only timestamps and beat counts.
        """
        ...

