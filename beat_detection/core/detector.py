"""
Core beat detection functionality.
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from madmom.features.tempo import CombFilterTempoHistogramProcessor
from beat_detection.utils.constants import SUPPORTED_METERS


@dataclass
class BeatStatistics:
    """Statistics about detected beats."""
    mean_interval: float
    median_interval: float
    std_interval: float
    min_interval: float
    max_interval: float
    irregularity_percent: float
    tempo_bpm: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy serialization."""
        return {
            'mean_interval': self.mean_interval,
            'median_interval': self.median_interval,
            'std_interval': self.std_interval,
            'min_interval': self.min_interval,
            'max_interval': self.max_interval,
            'irregularity_percent': self.irregularity_percent,
            'tempo_bpm': self.tempo_bpm
        }


class BeatDetector:
    """Detect beats and downbeats in audio files."""
    
    def __init__(self, min_bpm: int = 60, max_bpm: int = 240, fps: int = 100, 
                 tolerance_percent: float = 10.0, min_consistent_beats: int = 8,
                 beats_per_bar: Optional[int] = None):
        """
        Initialize the beat detector.
        
        Parameters:
        -----------
        min_bpm : int
            Minimum beats per minute to detect
        max_bpm : int
            Maximum beats per minute to detect
        fps : int
            Frames per second for processing
        tolerance_percent : float
            Percentage tolerance for beat intervals
        min_consistent_beats : int
            Minimum number of consistent beats to consider a stable section
        beats_per_bar : Optional[int]
            Number of beats per bar to use for downbeat alignment. If None, will try all supported meters (2, 3, 4).
            Default: None
        """
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.fps = fps
        self.tolerance_percent = tolerance_percent
        self.min_consistent_beats = min_consistent_beats
        self.beats_per_bar = beats_per_bar
        
        # Initialize processors
        self.beat_processor = RNNBeatProcessor()
        self.downbeat_processor = RNNDownBeatProcessor()
    
    def detect_beats(self, audio_file: str, skip_intro: bool = True, skip_ending: bool = True) -> Tuple[np.ndarray, BeatStatistics, List[int], np.ndarray, int, int, int]:
        """
        Detect beat timestamps and downbeats in an audio file.
        
        Parameters:
        -----------
        audio_file : str
            Path to the input audio file
        skip_intro : bool
            Whether to detect and skip intro sections
        skip_ending : bool
            Whether to detect and skip ending sections
            
        Returns:
        --------
        numpy.ndarray
            Array of beat timestamps in seconds
        BeatStatistics
            Beat analysis statistics
        list
            Indices of irregular beats
        numpy.ndarray
            Array of downbeat indices (which beats are downbeats)
        int
            Index where the intro ends (0 if no intro detected or skip_intro is False)
        int
            Index where the ending begins (len(beats) if no ending detected or skip_ending is False)
        int
            Detected meter (time signature numerator, typically 3 or 4)
        """
        # Detect beats
        beat_activations = self.beat_processor(audio_file)
        
        # Create a dedicated TempoHistogramProcessor instance to avoid deprecation warning
        tempo_histogram_processor = CombFilterTempoHistogramProcessor(
            min_bpm=self.min_bpm,
            max_bpm=self.max_bpm,
            fps=self.fps
        )
        
        beat_tracking = BeatTrackingProcessor(
            min_bpm=self.min_bpm,
            max_bpm=self.max_bpm,
            fps=self.fps,
            histogram_processor=tempo_histogram_processor
        )
        
        original_beats = beat_tracking(beat_activations)
        
        # Store the original beats for returning intro/ending indices
        all_beats = np.copy(original_beats)
        
        # Analyze the beats to find irregularities
        stats, irregular_beats = self._analyze_beat_intervals(original_beats)
        
        # Initialize intro and ending indices
        intro_end_idx = 0
        ending_start_idx = len(original_beats)
        
        # Detect intro if requested
        if skip_intro:
            intro_end_idx = self._detect_intro_end(original_beats, stats)
            
            if intro_end_idx > 0:
                original_beats = original_beats[intro_end_idx:]
                # Recalculate statistics after skipping intro
                stats, irregular_beats = self._analyze_beat_intervals(original_beats)
        
        # Detect ending if requested
        if skip_ending and len(original_beats) > self.min_consistent_beats:
            ending_idx = self._detect_ending_start(original_beats, stats)
            
            if ending_idx < len(original_beats) and ending_idx > 0:
                # Adjust ending_start_idx to be relative to all_beats
                ending_start_idx = intro_end_idx + ending_idx
                
                # Trim the ending
                original_beats = original_beats[:ending_idx]
                # Recalculate statistics after skipping ending
                stats, irregular_beats = self._analyze_beat_intervals(original_beats)
        else:
            # If we didn't skip the ending, ending_start_idx should be relative to all_beats
            ending_start_idx = intro_end_idx + len(original_beats)
        
        # Warn about irregular beats
        if irregular_beats:
            warnings.warn(f"Found {len(irregular_beats)} irregular beats out of {len(original_beats)-1} intervals")
        
        # Detect downbeats
        downbeats = self._detect_downbeats(audio_file, original_beats)
        
        # Detect meter (time signature) based on beat patterns
        detected_meter = self._detect_meter(original_beats, downbeats)
        
        # If meter detection failed, raise an exception
        if detected_meter == -1:
            raise ValueError("Failed to detect a consistent time signature. The audio may have irregular beats, mixed time signatures, or not enough data.")
        
        return original_beats, stats, irregular_beats, downbeats, intro_end_idx, ending_start_idx, detected_meter
    
    def _detect_meter(self, beats: np.ndarray, downbeats: np.ndarray) -> int:
        """
        Detect the most likely meter (time signature numerator) based on beat patterns.
        
        Parameters:
        -----------
        beats : numpy.ndarray
            Array of beat timestamps in seconds
        downbeats : numpy.ndarray
            Array of indices that correspond to downbeats
            
        Returns:
        --------
        int
            Detected meter (2, 3, or 4)
            Returns -1 if the meter cannot be reliably determined
        """
        if len(beats) < 12 or len(downbeats) < 3:
            # Not enough data to make a reliable determination
            print("Error: Not enough beats or downbeats to determine time signature")
            return -1
        
        # Calculate the number of beats between consecutive downbeats
        beat_counts = []
        for i in range(1, len(downbeats)):
            beat_counts.append(downbeats[i] - downbeats[i-1])
        
        # Count occurrences of each beat count
        meter_counts = {}
        for meter in SUPPORTED_METERS:
            lower_bound = meter - 0.5
            upper_bound = meter + 0.5
            meter_counts[meter] = sum(1 for count in beat_counts if lower_bound <= count <= upper_bound)
        
        # Calculate irregular count (beats that don't match any supported meter)
        irregular_count = len(beat_counts) - sum(meter_counts.values())
        
        # Calculate the percentage of each meter
        total_measures = len(beat_counts)
        meter_percentages = {}
        for meter in SUPPORTED_METERS:
            meter_percentages[meter] = (meter_counts[meter] / total_measures) * 100 if total_measures > 0 else 0
        
        percent_irregular = (irregular_count / total_measures) * 100 if total_measures > 0 else 0
        
        # Log the results
        meter_log = ", ".join([f"{meter}/4: {meter_percentages[meter]:.1f}%" for meter in SUPPORTED_METERS])
        print(f"Time signature analysis: {meter_log}, Irregular: {percent_irregular:.1f}%")
        
        # Check if there are too many irregular measures
        if percent_irregular > 30:
            print("Error: Too many irregular measures detected")
            return -1
        
        # Check if there's a mix of time signatures with no clear winner
        mixed_signatures = False
        for i, meter1 in enumerate(SUPPORTED_METERS):
            for meter2 in SUPPORTED_METERS[i+1:]:
                if meter_percentages[meter1] >= 30 and meter_percentages[meter2] >= 30:
                    mixed_signatures = True
                    break
            if mixed_signatures:
                break
                
        if mixed_signatures:
            print("Error: Mixed time signatures detected")
            return -1
        
        # Determine the most likely meter
        for meter in SUPPORTED_METERS:
            if meter_percentages[meter] > 50:
                return meter
                
        # If no clear winner
        print("Error: No clear time signature detected")
        return -1
    
    def _detect_downbeats(self, audio_file: str, beats: np.ndarray) -> np.ndarray:
        """
        Detect downbeats in the audio file and align them with detected beats.
        
        Parameters:
        -----------
        audio_file : str
            Path to the input audio file
        beats : numpy.ndarray
            Array of beat timestamps
            
        Returns:
        --------
        numpy.ndarray
            Array of indices in the beats array that correspond to downbeats
        """
        # Run the downbeat processor
        downbeat_activations = self.downbeat_processor(audio_file)
        
        # Process with DBN downbeat tracker
        beats_per_bar_param = [self.beats_per_bar] if self.beats_per_bar is not None else SUPPORTED_METERS
        downbeat_tracker = DBNDownBeatTrackingProcessor(
            beats_per_bar=beats_per_bar_param, 
            fps=self.fps
        )
        
        # Get downbeats with their beat positions (1.0 means downbeat)
        beats_with_positions = downbeat_tracker(downbeat_activations)
        
        if len(beats_with_positions) == 0:
            # Fall back to simple method if downbeat detection fails
            return self._estimate_downbeats(beats)
        
        # Extract just the timestamps of the downbeats (where beat position is 1)
        downbeat_times = beats_with_positions[beats_with_positions[:, 1] == 1][:, 0]
        
        # Now we need to find the indices of beats that are closest to downbeats
        downbeat_indices = []
        
        for downbeat_time in downbeat_times:
            # Find the closest beat to this downbeat
            idx = np.argmin(np.abs(beats - downbeat_time))
            if idx not in downbeat_indices:
                downbeat_indices.append(idx)
        
        return np.array(downbeat_indices)
    
    def _estimate_downbeats(self, beats: np.ndarray) -> np.ndarray:
        """
        Estimate downbeats based on regular intervals when detection fails.
        
        Parameters:
        -----------
        beats : numpy.ndarray
            Array of beat timestamps
            
        Returns:
        --------
        numpy.ndarray
            Array of indices in the beats array that correspond to estimated downbeats
        """
        # Simple fallback: assume the first beat is a downbeat and use beats_per_bar
        downbeat_indices = np.arange(0, len(beats), self.beats_per_bar)
        return downbeat_indices
    
    def _analyze_beat_intervals(self, beat_timestamps: np.ndarray) -> Tuple[BeatStatistics, List[int]]:
        """
        Analyze beat intervals to detect irregularities.
        
        Parameters:
        -----------
        beat_timestamps : numpy.ndarray
            Array of beat timestamps in seconds
            
        Returns:
        --------
        BeatStatistics
            Beat interval statistics
        list
            List of indices of irregular beats
        """
        if len(beat_timestamps) < 2:
            # Return empty statistics if not enough beats
            return BeatStatistics(
                mean_interval=0,
                median_interval=0,
                std_interval=0,
                min_interval=0,
                max_interval=0,
                irregularity_percent=0,
                tempo_bpm=0
            ), []
        
        # Calculate intervals between consecutive beats
        intervals = np.diff(beat_timestamps)
        
        # Calculate statistics
        mean_interval = np.mean(intervals)
        median_interval = np.median(intervals)
        std_interval = np.std(intervals)
        min_interval = np.min(intervals)
        max_interval = np.max(intervals)
        
        # Detect irregularities (beats that are too far from the median)
        tolerance = median_interval * (self.tolerance_percent / 100)
        lower_bound = median_interval - tolerance
        upper_bound = median_interval + tolerance
        
        irregular_beats = []
        for i, interval in enumerate(intervals):
            if interval < lower_bound or interval > upper_bound:
                # i+1 because intervals[i] is between timestamps[i] and timestamps[i+1]
                irregular_beats.append(i+1)
        
        stats = BeatStatistics(
            mean_interval=mean_interval,
            median_interval=median_interval,
            std_interval=std_interval,
            min_interval=min_interval,
            max_interval=max_interval,
            irregularity_percent=(len(irregular_beats) / len(intervals)) * 100,
            tempo_bpm=60 / median_interval  # Convert interval to BPM
        )
        
        return stats, irregular_beats
    
    def _detect_intro_end(self, beat_timestamps: np.ndarray, 
                          stats: BeatStatistics) -> int:
        """
        Detect where the introduction ends and the regular beat pattern begins.
        
        Parameters:
        -----------
        beat_timestamps : numpy.ndarray
            Array of beat timestamps in seconds
        stats : BeatStatistics
            Beat interval statistics
            
        Returns:
        --------
        int
            Index where the intro ends and regular beats begin, or 0 if no intro detected
        """
        if len(beat_timestamps) < self.min_consistent_beats + 1:
            return 0
        
        intervals = np.diff(beat_timestamps)
        median_interval = stats.median_interval
        tolerance = median_interval * 0.1  # 10% tolerance
        
        # Slide a window through the beats to find a consistent section
        for i in range(len(intervals) - self.min_consistent_beats + 1):
            window = intervals[i:i+self.min_consistent_beats]
            if np.all(np.abs(window - median_interval) < tolerance):
                return i
        
        return 0
        
    def _detect_ending_start(self, beat_timestamps: np.ndarray, 
                           stats: BeatStatistics) -> int:
        """
        Detect where the regular beat pattern ends and the ending section begins.
        
        Parameters:
        -----------
        beat_timestamps : numpy.ndarray
            Array of beat timestamps in seconds
        stats : BeatStatistics
            Beat interval statistics
            
        Returns:
        --------
        int
            Index where the regular beats end and ending begins, or len(beat_timestamps) if no ending detected
        """
        if len(beat_timestamps) < self.min_consistent_beats + 1:
            return len(beat_timestamps)
        
        intervals = np.diff(beat_timestamps)
        median_interval = stats.median_interval
        tolerance = median_interval * 0.1  # 10% tolerance
        
        # Slide a window through the beats from the end to find where consistency breaks
        for i in range(len(intervals) - self.min_consistent_beats, -1, -1):
            window = intervals[i:i+self.min_consistent_beats]
            if np.all(np.abs(window - median_interval) < tolerance):
                # We found the last consistent window, so the ending starts after it
                return i + self.min_consistent_beats
        
        return len(beat_timestamps)