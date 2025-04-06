"""
Core beat detection functionality.
"""

import numpy as np
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable, Protocol
from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from madmom.features.tempo import CombFilterTempoHistogramProcessor
from beat_detection.utils.constants import SUPPORTED_METERS
from beat_detection.core.beats import Beats, BeatCalculationError


class BeatDetector:
    """Detect beats and downbeats in audio files."""
    
    def __init__(self, min_bpm: int = 60, max_bpm: int = 240, 
                 tolerance_percent: float = 10.0, min_consistent_measures: int = 5,
                 beats_per_bar: Optional[int] = None,
                 progress_callback: Optional[Callable[[str, float], None]] = None,
                 fps: int = 100):
        """
        Initialize the beat detector.
        
        Parameters:
        -----------
        min_bpm : int
            Minimum beats per minute to detect
        max_bpm : int
            Maximum beats per minute to detect
        tolerance_percent : float
            Percentage tolerance for beat intervals
        min_consistent_measures : int
            Minimum number of consistent measures required for stable section analysis in Beats class.
        beats_per_bar : Optional[int]
            Number of beats per bar to use for downbeat alignment. If None, will try all supported meters (2, 3, 4).
            Default: None
        fps : int
            Frames per second expected for the activation functions (default: 100).
        """
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.tolerance_percent = tolerance_percent
        self.min_consistent_measures = min_consistent_measures
        self.beats_per_bar = beats_per_bar
        self.progress_callback = progress_callback
        self.fps = fps
        
        # Initialize processors
        self.beat_processor = RNNBeatProcessor()
        self.downbeat_processor = RNNDownBeatProcessor()
        # Initialize beat tracker using self.fps
        self.beat_tracker = BeatTrackingProcessor(fps=self.fps)
        # Initialize downbeat tracker
        self.downbeat_tracker = DBNDownBeatTrackingProcessor(
            beats_per_bar=SUPPORTED_METERS, # Let madmom choose based on activation
            min_bpm=self.min_bpm, 
            max_bpm=self.max_bpm, 
            fps=self.fps # Use self.fps here too
        )
    
    def detect_beats(
        self,
        audio_file: str
    ) -> Beats:
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
        cb = self.progress_callback # Use the instance callback directly
        
        def report_progress(stage: str, value: float):
            if cb:
                if isinstance(cb, Callable) and not isinstance(cb, Protocol): 
                    try: 
                        cb(value) 
                    except TypeError:
                        print(f"Note: Progress callback signature mismatch during {stage}.")
                        pass

        if cb: # Check if instance callback exists
            report_progress("start", 0.0)
            
        # Detect beats
        beat_activations = self.beat_processor(audio_file)
        beat_timestamps = self.beat_tracker(beat_activations)
        
        if len(beat_timestamps) == 0:
            raise BeatCalculationError(f"No beats detected in file: {audio_file}")
        
        if cb:
            report_progress("beats_detected", 0.4)
            
        # Detect downbeats and meter
        downbeat_indices, meter = self._detect_downbeats(audio_file, beat_timestamps)
        
        if cb:
            report_progress("downbeats_detected", 0.8)
            
        # Create Beats object using the factory method which handles calculations
        try:
            beats_obj = Beats.from_timestamps(
                timestamps=beat_timestamps,
                downbeat_indices=downbeat_indices,
                meter=meter,
                tolerance_percent=self.tolerance_percent,
                min_consistent_measures=self.min_consistent_measures
            )
        except BeatCalculationError as e:
            raise BeatCalculationError(f"Error creating Beats object for {audio_file}: {e}") from e
            
        if cb:
            report_progress("analysis_complete", 1.0)
            
        return beats_obj
    
    def _detect_downbeats(self, audio_file: str, beat_timestamps: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Detect downbeats using madmom and determine the meter.
        
        Parameters:
        -----------
        audio_file : str
            Path to the input audio file
        beat_timestamps : np.ndarray
            Array of beat timestamps
            
        Returns:
        --------
        Tuple[np.ndarray, int]
            Tuple containing:
                - downbeat_indices: Array of indices into `beat_timestamps` that are downbeats.
                - meter: Detected meter (time signature numerator, e.g., 2, 3, 4).
                
        Raises:
        ------
        BeatCalculationError
            If downbeats cannot be determined or meter is invalid.
        """
        # --- Input Validation --- 
        if not isinstance(self.min_bpm, int) or self.min_bpm <= 0:
            raise BeatCalculationError(f"Invalid min_bpm: {self.min_bpm}. Must be a positive integer.")
        if not isinstance(self.max_bpm, int) or self.max_bpm <= self.min_bpm:
            raise BeatCalculationError(
                f"Invalid max_bpm: {self.max_bpm}. Must be a positive integer greater than min_bpm ({self.min_bpm})."
            )
        # --- End Validation --- 
        
        # Detect downbeats using RNN
        downbeat_activations = self.downbeat_processor(audio_file)
        
        # Pass SUPPORTED_METERS to let madmom determine best fit based on activations
        # Add fps=100 and rely on default internal histogram processor
        # Use the initialized downbeat tracker
        raw_downbeats = self.downbeat_tracker(downbeat_activations)
        
        if raw_downbeats is None or len(raw_downbeats) == 0:
            raise BeatCalculationError("Madmom DBNDownBeatTrackingProcessor returned no downbeats.")
            
        # Determine meter from the max beat number reported by madmom
        try:
            meter = int(np.max(raw_downbeats[:, 1]))
            if meter not in SUPPORTED_METERS:
                 # This case might indicate poor detection or unusual music
                 # Fallback or specific handling might be needed. Raising error for now.
                 raise BeatCalculationError(f"Madmom detected an unsupported meter: {meter}")
        except (ValueError, IndexError) as e:
             raise BeatCalculationError(f"Could not determine meter from madmom output: {e}") from e

        # Extract timestamps where beat_number is 1 (downbeats)
        downbeat_timestamps = raw_downbeats[raw_downbeats[:, 1] == 1, 0]

        if len(downbeat_timestamps) == 0:
            # This can happen if detection is poor or meter > 1 but no beat '1' detected
            raise BeatCalculationError("No downbeats (beat number 1) found in madmom output.")

        # Find the indices in beat_timestamps that correspond to downbeat_timestamps
        print(f"beat_timestamps: {beat_timestamps}")
        print(f"downbeat_timestamps: {downbeat_timestamps}")
        downbeat_indices = self._align_downbeats_to_beats(beat_timestamps, downbeat_timestamps)
        print(f"downbeat_indices: {downbeat_indices}")

        if not downbeat_indices:
             raise BeatCalculationError("Could not align any downbeat timestamps with detected beat timestamps.")
             
        return np.array(downbeat_indices, dtype=int), meter

    def _align_downbeats_to_beats(
        self, 
        beat_timestamps: np.ndarray, 
        downbeat_timestamps: np.ndarray, 
        search_tolerance: float = 0.02
    ) -> List[int]:
        """
        Align downbeat timestamps to the closest beat timestamps within a tolerance.

        Parameters:
        -----------
        beat_timestamps : np.ndarray
            Sorted array of detected beat timestamps.
        downbeat_timestamps : np.ndarray
            Sorted array of detected downbeat timestamps.
        search_tolerance : float, optional
            Tolerance in seconds for matching timestamps (default is 0.02).

        Returns:
        --------
        List[int]
            List of indices in `beat_timestamps` that correspond to the aligned downbeats.
        """
        downbeat_indices = []
        current_beat_idx = 0
        
        for dbt in downbeat_timestamps:
            # Search efficiently in the sorted beat timestamps array
            while current_beat_idx < len(beat_timestamps) and beat_timestamps[current_beat_idx] < dbt - search_tolerance:
                current_beat_idx += 1
            
            # Add a small epsilon to the tolerance comparison to handle floating point inaccuracies
            # This ensures that differences *exactly* equal to the tolerance are included.
            epsilon = 1e-9 
            if current_beat_idx < len(beat_timestamps) and abs(beat_timestamps[current_beat_idx] - dbt) <= search_tolerance + epsilon:
                downbeat_indices.append(current_beat_idx)
                # Move past this beat for the next search to avoid re-matching
                current_beat_idx += 1
            # else: Downbeat timestamp didn't align closely with any beat timestamp. 
            # Consider logging or warning if necessary, but per fail-fast, we might just let the final check catch it.
            # print(f"Warning: Downbeat at time {dbt:.3f} did not align with any detected beat timestamp.")

        return downbeat_indices