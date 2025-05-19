"""
Base class for beat detectors with common functionality.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple
from pathlib import Path
from pydub import AudioSegment
from beat_detection.core.beats import BeatCalculationError
import beat_detection.utils.constants as constants


@dataclass(slots=True, frozen=True)
class DetectorConfig:
    """Unified configuration for all beat detectors.
    
    Analysis of the existing detectors shows they share these common parameters.
    """
    min_bpm: int = 60
    max_bpm: int = 240
    fps: Optional[int] = None
    beats_per_bar: Optional[Union[List[int], Tuple[int, ...]]] = field(
        default_factory=lambda: constants.SUPPORTED_BEATS_PER_BAR
    )


class BaseBeatDetector:
    """Base class for beat detectors with common functionality."""
    
    def __init__(self, cfg: DetectorConfig) -> None:
        """Initialize the detector with a configuration object.
        
        Parameters:
        -----------
        cfg : DetectorConfig
            Configuration object with detector parameters.
        """
        self.cfg = cfg

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