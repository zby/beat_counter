"""
Core beat detection functionality.
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor


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
                 beats_per_bar: int = 4):
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
        beats_per_bar : int
            Number of beats per bar to use for downbeat alignment (default: 4)
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
    
    def detect_beats(self, audio_file: str, skip_intro: bool = True) -> Tuple[np.ndarray, BeatStatistics, List[int], np.ndarray]:
        """
        Detect beat timestamps and downbeats in an audio file.
        
        Parameters:
        -----------
        audio_file : str
            Path to the input audio file
        skip_intro : bool
            Whether to detect and skip intro sections
            
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
        """
        # Detect beats
        beat_activations = self.beat_processor(audio_file)
        
        beat_tracking = BeatTrackingProcessor(
            min_bpm=self.min_bpm,
            max_bpm=self.max_bpm,
            fps=self.fps
        )
        
        beats = beat_tracking(beat_activations)
        
        # Analyze the beats to find irregularities
        stats, irregular_beats = self._analyze_beat_intervals(beats)
        
        # Detect and skip the intro if requested
        if skip_intro:
            intro_end_idx = self._detect_intro_end(beats, stats)
            
            if intro_end_idx > 0:
                beats = beats[intro_end_idx:]
                # Recalculate statistics after skipping intro
                stats, irregular_beats = self._analyze_beat_intervals(beats)
        
        # Warn about irregular beats
        if irregular_beats:
            warnings.warn(f"Found {len(irregular_beats)} irregular beats out of {len(beats)-1} intervals")
        
        # Detect downbeats
        downbeats = self._detect_downbeats(audio_file, beats)
        
        return beats, stats, irregular_beats, downbeats
    
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
        downbeat_tracker = DBNDownBeatTrackingProcessor(
            beats_per_bar=[self.beats_per_bar], 
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