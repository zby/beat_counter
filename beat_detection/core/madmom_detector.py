"""
Madmom-specific beat detection implementation.
"""

import numpy as np
import warnings
from pathlib import Path
from typing import Optional, List, Tuple, Union
from pydub import AudioSegment

from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

from beat_detection.core.beats import RawBeats, BeatCalculationError
from beat_detection.core.detector_protocol import BeatDetector
from beat_detection.core.base_detector import BaseBeatDetector

import beat_detection.utils.constants as constants

class MadmomBeatDetector(BaseBeatDetector): # Inherit from BaseBeatDetector
    """Detect beats and downbeats in audio files using Madmom."""

    def __init__(
        self,
        beats_per_bar: Optional[Union[List[int], Tuple[int, ...]]] = constants.SUPPORTED_BEATS_PER_BAR,
        min_bpm: Optional[int] = None,
        max_bpm: Optional[int] = None,
        fps: Optional[int] = constants.FPS,
    ):
        """
        Initialize the MadmomBeatDetector.

        Parameters:
        ----------
        beats_per_bar : Optional[Union[List[int], Tuple[int, ...]]], optional
            A list or tuple of integers representing the number of beats per bar
            to guide the downbeat tracker (e.g., [3, 4]).
            Defaults to SUPPORTED_BEATS_PER_BAR (typically [3, 4]).
            If None, Madmom's processor throws an error.
        min_bpm : Optional[int], optional
            Minimum tempo to consider. If None, Madmom's default is used.
        max_bpm : Optional[int], optional
            Maximum tempo to consider. If None, Madmom's default is used.
        fps : Optional[int], optional
            Frames per second for activation functions. If None, Madmom's default is used.

        Raises:
        -------
        BeatCalculationError
            If provided parameters are invalid.
        """
        # --- Input Validation (Applied only if values are provided or for defaults) ---
        if beats_per_bar is not None: # This includes the default SUPPORTED_BEATS_PER_BAR
            if not isinstance(beats_per_bar, (list, tuple)):
                raise BeatCalculationError(
                    f"Invalid beats_per_bar: {beats_per_bar}. Must be a list or tuple of integers if provided."
                )
            if not beats_per_bar: # Check for empty list/tuple
                 raise BeatCalculationError(
                    f"Invalid beats_per_bar: {beats_per_bar}. Cannot be an empty list/tuple."
                )
            for bpb_val in beats_per_bar:
                if not isinstance(bpb_val, int) or bpb_val <= 0:
                    raise BeatCalculationError(
                        f"Invalid value in beats_per_bar: {bpb_val}. All values must be positive integers."
                    )
        # If beats_per_bar is explicitly None, Madmom might use a very broad default for its processor
        # or potentially error if the processor strictly requires it. 
        # Our default is SUPPORTED_BEATS_PER_BAR, so this condition is met unless a caller explicitly passes None.

        if min_bpm is not None:
            if not isinstance(min_bpm, int) or min_bpm <= 0:
                raise BeatCalculationError(
                    f"Invalid min_bpm: {min_bpm}. Must be a positive integer if provided."
                )
        if max_bpm is not None:
            if not isinstance(max_bpm, int) or max_bpm <= 0:
                 raise BeatCalculationError(
                    f"Invalid max_bpm: {max_bpm}. Must be a positive integer if provided."
                )
        if min_bpm is not None and max_bpm is not None:
            if max_bpm <= min_bpm:
                raise BeatCalculationError(
                    f"Invalid max_bpm: {max_bpm}. Must be > min_bpm ({min_bpm}) if both are provided."
                )
        if fps is not None:
            if not isinstance(fps, int) or fps <= 0:
                raise BeatCalculationError(
                    f"Invalid fps: {fps}. Must be a positive integer if provided."
                )
        # --- End Validation ---

        self.beats_per_bar = beats_per_bar
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.fps = fps

        self.downbeat_processor = RNNDownBeatProcessor()

        dbn_kwargs = { 'beats_per_bar': self.beats_per_bar }
        
        if self.min_bpm is not None:
            dbn_kwargs['min_bpm'] = float(self.min_bpm)
        if self.max_bpm is not None:
            dbn_kwargs['max_bpm'] = float(self.max_bpm)
        if self.fps is not None:
            dbn_kwargs['fps'] = int(self.fps)

        self.downbeat_tracker = DBNDownBeatTrackingProcessor(**dbn_kwargs)


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

    def detect_beats(self, audio_path: str | Path) -> RawBeats:
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

        # Get audio duration for clip_length
        clip_length = self._get_audio_duration(audio_path)

        # Detect downbeats, getting only raw data (no bpb calculation here)
        raw_beats_with_counts = self._detect_downbeats(str(audio_path)) # Pass string path

        if raw_beats_with_counts is None or len(raw_beats_with_counts) == 0:
            raise BeatCalculationError(
                f"No beats detected in file (Madmom DBN Tracker): {audio_path}"
            )

        beat_timestamps = raw_beats_with_counts[:, 0]
        beat_counts = raw_beats_with_counts[:, 1].astype(int)

        # Return RawBeats object with clip_length from audio duration
        # Validation happens within RawBeats __post_init__
        return RawBeats(
            timestamps=beat_timestamps,
            beat_counts=beat_counts,
            clip_length=clip_length,
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

            # Pass activations to the DBNDownBeatTrackingProcessor
            raw_downbeats = self.downbeat_tracker(downbeat_activations)

        except Exception as e:
            # Catch potential Madmom errors
            raise BeatCalculationError(f"Madmom processing failed: {e}") from e

        if raw_downbeats is None or len(raw_downbeats) == 0:
            raise BeatCalculationError(
                "Madmom DBNDownBeatTrackingProcessor returned no beats."
            )

        # --- Shape Validation (ensure Madmom output is as expected) ---
        if raw_downbeats.ndim != 2:
            raise BeatCalculationError(
                f"Madmom output array has unexpected shape (ndim != 2): {raw_downbeats.shape}"
            )
        if raw_downbeats.shape[1] < 2: # Must have at least time and count columns
            raise BeatCalculationError(
                f"Madmom output array has unexpected shape (columns < 2): {raw_downbeats.shape}"
            )
        # --- End Shape Validation ---

        # check if Madmom's determined meter is one of the specified options.
        try:
            # Madmom's output (raw_downbeats[:, 1]) contains the beat number (1, 2, 3, etc.)
            # The maximum of this column gives the meter it effectively detected/used.
            detected_bpb_from_output = int(np.max(raw_downbeats[:, 1]))

            if detected_bpb_from_output <= 0:
                # This indicates an issue with Madmom's output or our understanding of it.
                raise BeatCalculationError(
                    f"Madmom output implies an invalid beats_per_bar: {detected_bpb_from_output}. "
                    f"Expected positive value. Raw output max count: {np.max(raw_downbeats[:, 1])}"
                )

            # Check if the single detected value is one of the options we provided/defaulted to.
            if detected_bpb_from_output not in self.beats_per_bar:
                warnings.warn(
                    f"Madmom's output indicates a beats_per_bar of {detected_bpb_from_output}, "
                    f"which is not one of the configured options in self.beats_per_bar={self.beats_per_bar}. "
                    f"Using Madmom's output counts directly.",
                    UserWarning
                )
        except ValueError: # Handles cases like empty raw_downbeats or non-numeric data in counts column
            # This might occur if raw_downbeats[:, 1] is empty or contains non-castable values.
            raise BeatCalculationError(
                "Could not determine beats_per_bar from Madmom's output for comparison. "
                f"Problematic raw output second column (counts): {raw_downbeats[:, 1]}"
            )
        except IndexError: # If raw_downbeats somehow doesn't have a second column after passing shape checks
                raise BeatCalculationError(
                "Could not access beat counts (second column) from Madmom's output for comparison. "
                f"Madmom output shape: {raw_downbeats.shape}"
            )

        return raw_downbeats 