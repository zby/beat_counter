"""
Core beat detection functionality and protocols.
"""

import numpy as np
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable, Protocol, runtime_checkable
from pathlib import Path
from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
from madmom.processors import IOProcessor, Processor
from beat_detection.utils.constants import SUPPORTED_BEATS_PER_BAR
from beat_detection.core.beats import RawBeats, Beats, BeatCalculationError


@runtime_checkable
class BeatDetector(Protocol):
    """Protocol for beat detection algorithms."""

    def detect(self, audio_path: str | Path) -> RawBeats:
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


class MadmomBeatDetector:
    """Detect beats and downbeats in audio files using Madmom."""

    def __init__(
        self,
        min_bpm: int = 60,
        max_bpm: int = 240,
        progress_callback: Optional[Callable[[float], None]] = None,
        fps: int = 100,
    ):
        """
        Initialize the MadmomBeatDetector.

        Parameters:
        ----------
        min_bpm : int
            Minimum tempo to consider (default: 60).
        max_bpm : int
            Maximum tempo to consider (default: 240).
        progress_callback : Optional[Callable[[float], None]]
            Callback function for progress updates (0.0 to 1.0) (default: None).
        fps : int
            Frames per second expected for the activation functions (default: 100).

        Raises:
        -------
        BeatCalculationError
            If BPM parameters are invalid.
        """
        # --- Input Validation (Moved to __init__ for fail-fast) ---
        if not isinstance(min_bpm, int) or min_bpm <= 0:
            raise BeatCalculationError(
                f"Invalid min_bpm: {min_bpm}. Must be a positive integer."
            )
        if not isinstance(max_bpm, int) or max_bpm <= min_bpm:
            raise BeatCalculationError(
                f"Invalid max_bpm: {max_bpm}. Must be > min_bpm ({min_bpm})."
            )
        if not isinstance(fps, int) or fps <= 0:
            raise BeatCalculationError(
                f"Invalid fps: {fps}. Must be a positive integer."
            )
        # --- End Validation ---

        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.progress_callback = progress_callback
        self.fps = fps

        # Initialize processors
        try:
            self.downbeat_processor = RNNDownBeatProcessor()
            # Initialize downbeat tracker
            self.downbeat_tracker = DBNDownBeatTrackingProcessor(
                beats_per_bar=SUPPORTED_BEATS_PER_BAR,  # Let madmom choose based on activation
                min_bpm=float(self.min_bpm),
                max_bpm=float(self.max_bpm),
                fps=float(self.fps),
            )
        except Exception as e:
            # Catch potential madmom initialization errors (though validation should prevent most)
            raise BeatCalculationError(f"Failed to initialize Madmom processors: {e}") from e

    def detect(self, audio_path: str | Path) -> RawBeats:
        """
        Detects beats in an audio file using Madmom.

        Implements the BeatDetector protocol.

        Parameters:
        -----------
        audio_path : str | Path
            Path to the input audio file.

        Returns:
        --------
        RawBeats
            Object containing detected timestamps and beat counts.

        Raises:
        -------
        BeatCalculationError
            If no beats are detected or an error occurs during processing.
        FileNotFoundError
            If the audio_path does not exist.
        """
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        cb = self.progress_callback

        def report_progress(value: float):
            if cb:
                try:
                    cb(value)
                except Exception as e:
                    # Fail fast but informatively if callback fails
                    print(f"Warning: Progress callback failed: {e}")

        report_progress(0.0)

        # Detect downbeats, getting only raw data (no bpb calculation here)
        raw_beats_with_counts = self._detect_downbeats(str(audio_path)) # Pass string path

        report_progress(0.8) # Approx progress after detection

        if raw_beats_with_counts is None or len(raw_beats_with_counts) == 0:
            raise BeatCalculationError(
                f"No beats detected in file (Madmom DBN Tracker): {audio_path}"
            )

        beat_timestamps = raw_beats_with_counts[:, 0]
        beat_counts = raw_beats_with_counts[:, 1].astype(int)

        report_progress(1.0)

        # Return simplified RawBeats object (timestamps and counts only)
        # Validation happens within RawBeats __post_init__
        return RawBeats(
            timestamps=beat_timestamps,
            beat_counts=beat_counts,
        )

    def _detect_downbeats(self, audio_file_path: str) -> np.ndarray:
        """Detect beats in an audio file using Madmom.

        Parameters:
        -----------
        audio_file_path : str
            Path to the input audio file.

        Returns:
        --------
        np.ndarray
            Array of shape (N, 2) containing beat times and beat counts.

        Raises:
        -------
        BeatCalculationError
            If an error occurs during processing.
        """
        try:
            # Detect beat activations
            downbeat_activations = self.downbeat_processor(audio_file_path)

            # Pass activations to the DBNBeatTrackingProcessor
            raw_downbeats = self.downbeat_tracker(downbeat_activations)

        except Exception as e:
            # Catch potential Madmom errors
            raise BeatCalculationError(f"Madmom processing failed: {e}") from e

        if raw_downbeats is None or len(raw_downbeats) == 0:
            raise BeatCalculationError(
                "Madmom DBNDownBeatTrackingProcessor returned no beats."
            )

        # Determine beats_per_bar from the max beat number reported by madmom
        try:
            # --- Shape Validation ---
            # Check number of dimensions first
            if raw_downbeats.ndim != 2:
                raise BeatCalculationError(
                    f"Madmom output array has unexpected shape (ndim != 2): {raw_downbeats.shape}"
                )
            # Then check number of columns
            if raw_downbeats.shape[1] < 2:
                raise BeatCalculationError(
                    f"Madmom output array has unexpected shape (columns < 2): {raw_downbeats.shape}"
                )
            # --- End Shape Validation ---

            detected_beats_per_bar = int(np.max(raw_downbeats[:, 1]))
            if detected_beats_per_bar not in SUPPORTED_BEATS_PER_BAR:
                # Check if beats_per_bar is 0 or negative, which is definitely invalid
                if detected_beats_per_bar <= 0:
                    raise BeatCalculationError(
                        f"Madmom detected an invalid beats_per_bar: {detected_beats_per_bar}"
                    )
                # If it's positive but not in SUPPORTED_BEATS_PER_BAR, issue a warning and maybe default?
                warnings.warn(
                    f"Madmom detected an unsupported beats_per_bar: {detected_beats_per_bar}. Using it anyway."
                )
        except (ValueError, IndexError) as e:
            # This catch might now be less likely for shape errors, but keep for np.max etc.
            raise BeatCalculationError(
                f"Could not process madmom output: {e}"
            ) from e

        # Return the full array and the determined beats_per_bar
        return raw_downbeats
