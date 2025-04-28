"""
Core beat detection functionality.
"""

import numpy as np
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable, Protocol
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from beat_detection.utils.constants import SUPPORTED_BEATS_PER_BAR
from beat_detection.core.beats import Beats, BeatCalculationError


class BeatDetector:
    """Detect beats and downbeats in audio files."""

    def __init__(
        self,
        min_bpm: int = 60,
        max_bpm: int = 240,
        tolerance_percent: float = 10.0,
        min_measures: int = 5,
        beats_per_bar: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        fps: int = 100,
    ):
        """
        Initialize the BeatDetector.

        Parameters:
        ----------
        min_bpm : int
            Minimum tempo to consider (default: 60).
        max_bpm : int
            Maximum tempo to consider (default: 240).
        tolerance_percent : float
            Allowed deviation (%) for classifying beats as regular (default: 10.0).
        min_measures : int
            Minimum number of full measures required for stable tempo/beats_per_bar (default: 5).
        beats_per_bar : Optional[int]
            Expected beats per bar. If None, it's auto-detected (default: None).
        progress_callback : Optional[Callable[[str, float], None]]
            Callback function for progress updates (default: None).
        fps : int
            Frames per second expected for the activation functions (default: 100).
        """
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.tolerance_percent = tolerance_percent
        self.min_measures = min_measures
        self.beats_per_bar = beats_per_bar
        self.progress_callback = progress_callback
        self.fps = fps

        # Initialize processors
        self.downbeat_processor = RNNDownBeatProcessor()
        # Initialize downbeat tracker
        self.downbeat_tracker = DBNDownBeatTrackingProcessor(
            beats_per_bar=SUPPORTED_BEATS_PER_BAR,  # Let madmom choose based on activation
            min_bpm=float(self.min_bpm),
            max_bpm=float(self.max_bpm),
            fps=float(self.fps),  # Use self.fps here too (now it's a float)
        )

    def detect_beats(self, audio_file: str) -> Beats:
        """
        Detect beats and downbeats in an audio file and return a Beats object.

        Parameters:
        -----------
        audio_file : str
            Path to the input audio file

        Returns:
        --------
        Beats
            Object containing all beat-related information
        """
        cb = self.progress_callback  # Use the instance callback directly

        def report_progress(stage: str, value: float):
            if cb:
                if isinstance(cb, Callable) and not isinstance(cb, Protocol):
                    try:
                        cb(value)
                    except TypeError:
                        print(
                            f"Note: Progress callback signature mismatch during {stage}."
                        )
                        pass

        if cb:  # Check if instance callback exists
            report_progress("start", 0.0)

        # Detect beats and downbeats together using DBNDownBeatTrackingProcessor

        # Detect downbeats, beats_per_bar, and get all beat info (timestamps and counts)
        # The _detect_downbeats method now returns the raw array and beats_per_bar
        raw_beats_with_counts, detected_beats_per_bar = self._detect_downbeats(
            audio_file
        )

        if raw_beats_with_counts is None or len(raw_beats_with_counts) == 0:
            raise BeatCalculationError(
                f"No beats detected in file (DBN Tracker): {audio_file}"
            )

        beat_timestamps = raw_beats_with_counts[:, 0]
        beat_counts = raw_beats_with_counts[:, 1].astype(
            int
        )  # Ensure counts are integers

        if cb:
            report_progress("downbeats_detected", 0.8)

        # Create Beats object using the factory method which handles calculations
        try:
            # Pass the extracted beat_counts to the factory method
            beats_obj = Beats.from_timestamps(
                timestamps=beat_timestamps,
                beats_per_bar=detected_beats_per_bar,
                beat_counts=beat_counts,  # Pass the counts from the processor
                tolerance_percent=self.tolerance_percent,
                min_measures=self.min_measures,
            )
        except BeatCalculationError as e:
            raise BeatCalculationError(
                f"Error creating Beats object for {audio_file}: {e}"
            ) from e

        if cb:
            report_progress("analysis_complete", 1.0)

        return beats_obj

    def _detect_downbeats(self, audio_file: str) -> Tuple[np.ndarray, int]:
        """
        Detect beats and downbeats using madmom's DBNDownBeatTrackingProcessor.
        Determines the beats_per_bar based on the processor's output.

        Parameters:
        ----------
        audio_file : str
            Path to the input audio file

        Returns:
        --------
        Tuple[np.ndarray, int]
            Tuple containing:
                - raw_beats_with_counts: Nx2 array where column 0 is timestamp, column 1 is beat count.
                - beats_per_bar: Detected beats per bar (time signature numerator, e.g., 2, 3, 4).

        Raises:
        ------
        BeatCalculationError
            If beats/downbeats cannot be determined or beats_per_bar is invalid.
        """
        # --- Input Validation ---
        if not isinstance(self.min_bpm, int) or self.min_bpm <= 0:
            raise BeatCalculationError(
                f"Invalid min_bpm: {self.min_bpm}. Must be a positive integer."
            )
        if not isinstance(self.max_bpm, int) or self.max_bpm <= self.min_bpm:
            raise BeatCalculationError(
                f"Invalid max_bpm: {self.max_bpm}. Must be a positive integer greater than min_bpm ({self.min_bpm})."
            )
        # --- End Validation ---

        # Detect downbeat activations first
        downbeat_activations = self.downbeat_processor(audio_file)

        # Pass activations to the DBNDownBeatTrackingProcessor
        # This processor handles both beat tracking and downbeat counting
        raw_downbeats = self.downbeat_tracker(downbeat_activations)

        if raw_downbeats is None or len(raw_downbeats) == 0:
            raise BeatCalculationError(
                "Madmom DBNDownBeatTrackingProcessor returned no beats."
            )

        # Determine beats_per_bar from the max beat number reported by madmom
        try:
            # Ensure there's a second column before accessing it
            if raw_downbeats.shape[1] < 2:
                raise BeatCalculationError(
                    "Madmom output array has unexpected shape (less than 2 columns)."
                )

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
            raise BeatCalculationError(
                f"Could not determine beats_per_bar from madmom output: {e}"
            ) from e

        # Return the full array and the determined beats_per_bar
        return raw_downbeats, detected_beats_per_bar
