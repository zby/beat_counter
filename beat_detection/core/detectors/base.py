"""
Base class for beat detectors with common functionality.
"""

from pathlib import Path
from pydub import AudioSegment
from beat_detection.core.beats import BeatCalculationError

class BaseBeatDetector:
    """Base class for beat detectors with common functionality."""

    def _get_audio_duration(self, audio_path: str | Path) -> float:
        """Get the duration of an audio file in seconds.

        Parameters:
        -----------
        audio_path : str | Path
            Path to the audio file.

        Returns:
        --------
        float
            Duration of the audio file in seconds.

        Raises:
        -------
        BeatCalculationError
            If the audio file cannot be loaded or processed.
        """
        try:
            # Load the audio file using pydub
            audio = AudioSegment.from_file(str(audio_path))
            duration = len(audio) / 1000.0  # Convert milliseconds to seconds
            if duration <= 0:
                raise BeatCalculationError(f"Invalid audio duration: {duration} seconds")
            return duration
        except Exception as e:
            raise BeatCalculationError(f"Failed to get audio duration: {e}") from e 