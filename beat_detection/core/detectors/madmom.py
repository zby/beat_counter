"""
Madmom-specific beat detection implementation.
"""

import numpy as np
import warnings
from pathlib import Path
from typing import Optional, List, Tuple, Union

from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

from beat_detection.core.beats import RawBeats, BeatCalculationError
from beat_detection.core.registry import register
from beat_detection.core.detectors.base import BaseBeatDetector, DetectorConfig
import beat_detection.utils.constants as constants

MADMOM_DEFAULT_FPS = 100

@register("madmom")
class MadmomBeatDetector(BaseBeatDetector):
    """Detect beats and downbeats in audio files using Madmom."""

    def __init__(self, cfg: DetectorConfig):
        """
        Initialize the MadmomBeatDetector.

        Parameters:
        ----------
        cfg : DetectorConfig
            Configuration object with detector parameters.

        Raises:
        -------
        BeatCalculationError
            If provided parameters are invalid.
        """
        super().__init__(cfg)

        # Determine effective FPS for madmom post-processing
        # Priority: cfg.fps if provided, otherwise MADMOM_DEFAULT_FPS
        self._madmom_postprocessor_fps = self.cfg.fps if self.cfg.fps is not None else MADMOM_DEFAULT_FPS

        # --- Input Validation (Applied from the config object and postprocessor_fps) ---
        if self.cfg.beats_per_bar is not None:
            if not isinstance(self.cfg.beats_per_bar, (list, tuple)):
                raise BeatCalculationError(
                    f"Invalid beats_per_bar: {self.cfg.beats_per_bar}. Must be a list or tuple of integers if provided."
                )
            if not self.cfg.beats_per_bar:  # Check for empty list/tuple
                 raise BeatCalculationError(
                    f"Invalid beats_per_bar: {self.cfg.beats_per_bar}. Cannot be an empty list/tuple."
                )
            for bpb_val in self.cfg.beats_per_bar:
                if not isinstance(bpb_val, int) or bpb_val <= 0:
                    raise BeatCalculationError(
                        f"Invalid value in beats_per_bar: {bpb_val}. All values must be positive integers."
                    )

        if self.cfg.min_bpm is not None:
            if not isinstance(self.cfg.min_bpm, int) or self.cfg.min_bpm <= 0:
                raise BeatCalculationError(
                    f"Invalid min_bpm: {self.cfg.min_bpm}. Must be a positive integer if provided."
                )
        if self.cfg.max_bpm is not None:
            if not isinstance(self.cfg.max_bpm, int) or self.cfg.max_bpm <= 0:
                 raise BeatCalculationError(
                    f"Invalid max_bpm: {self.cfg.max_bpm}. Must be a positive integer if provided."
                )
        if self.cfg.min_bpm is not None and self.cfg.max_bpm is not None:
            if self.cfg.max_bpm <= self.cfg.min_bpm:
                raise BeatCalculationError(
                    f"Invalid max_bpm: {self.cfg.max_bpm}. Must be > min_bpm ({self.cfg.min_bpm}) if both are provided."
                )
        
        # Validate the postprocessor_fps (whether from cfg.fps or default)
        if not isinstance(self._madmom_postprocessor_fps, int) or self._madmom_postprocessor_fps <= 0:
            raise BeatCalculationError(
                f"Invalid postprocessor FPS for Madmom: {self._madmom_postprocessor_fps}. Must be a positive integer."
            )
        # --- End Validation ---

        self.downbeat_processor = RNNDownBeatProcessor()

        dbn_kwargs = { 'beats_per_bar': self.cfg.beats_per_bar }
        
        if self.cfg.min_bpm is not None:
            dbn_kwargs['min_bpm'] = float(self.cfg.min_bpm)
        if self.cfg.max_bpm is not None:
            dbn_kwargs['max_bpm'] = float(self.cfg.max_bpm)
        
        # Use the validated postprocessor_fps for the DBN tracker
        dbn_kwargs['fps'] = int(self._madmom_postprocessor_fps)

        self.downbeat_tracker = DBNDownBeatTrackingProcessor(**dbn_kwargs)

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
                f"Madmom output array has too few columns: {raw_downbeats.shape}"
            )
        # --- End Shape Validation ---

        return raw_downbeats 