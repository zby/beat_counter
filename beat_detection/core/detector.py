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
from beat_detection.core.beats import Beats, BeatStatistics


class BeatDetector:
    """Detect beats and downbeats in audio files."""
    
    def __init__(self, min_bpm: int = 60, max_bpm: int = 240, 
                 tolerance_percent: float = 10.0, min_consistent_beats: int = 8,
                 beats_per_bar: Optional[int] = None,
                 progress_callback: Optional[Callable[[str, float], None]] = None):
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
        min_consistent_beats : int
            Minimum number of consistent beats to consider a stable section
        beats_per_bar : Optional[int]
            Number of beats per bar to use for downbeat alignment. If None, will try all supported meters (2, 3, 4).
            Default: None
        """
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.tolerance_percent = tolerance_percent
        self.min_consistent_beats = min_consistent_beats
        self.beats_per_bar = beats_per_bar
        self.progress_callback = progress_callback
        
        # Initialize processors
        self.beat_processor = RNNBeatProcessor()
        self.downbeat_processor = RNNDownBeatProcessor()
    
    def detect_beats(
        self,
        audio_file: str,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Beats:
        """
        Detect beats in an audio file.
        
        Parameters:
        -----------
        audio_file : str
            Path to the input audio file
        progress_callback : Optional[Callable[[float], None]]
            Optional callback function for progress updates
            
        Returns:
        --------
        Beats
            Object containing all beat-related information
        """
        if progress_callback:
            progress_callback(0.0)
            
        # Detect beats
        beat_activations = self.beat_processor(audio_file)
        beats = self.beat_tracker(beat_activations)
        
        if progress_callback:
            progress_callback(0.3)
            
        # Analyze beat patterns
        intervals = np.diff(beats)
        stats = BeatStatistics(
            mean_interval=np.mean(intervals),
            median_interval=np.median(intervals),
            std_interval=np.std(intervals),
            min_interval=np.min(intervals),
            max_interval=np.max(intervals),
            irregularity_percent=0.0,  # Will be updated after irregularity analysis
            tempo_bpm=60.0 / np.mean(intervals),
            total_beats=len(beats)
        )
        
        # Store original beats for returning intro/ending indices
        original_beats = beats.copy()
        
        if progress_callback:
            progress_callback(0.5)
            
        # Analyze irregularities
        irregular_beats = self._analyze_irregularities(beats, intervals)
        stats.irregularity_percent = len(irregular_beats) / len(beats) * 100
        
        if irregular_beats:
            warnings.warn(f"Found {len(irregular_beats)} irregular beats")
            
        if progress_callback:
            progress_callback(0.7)
            
        # Detect intro and ending if requested
        intro_end_idx = 0
        ending_start_idx = len(beats)
        
        if self.skip_intro:
            intro_end_idx = self._detect_intro(beats, intervals)
            beats = beats[intro_end_idx:]
            
        if self.skip_ending:
            ending_start_idx = self._detect_ending(beats, intervals)
            beats = beats[:ending_start_idx]
            
        if progress_callback:
            progress_callback(0.8)
            
        # Detect downbeats and meter
        downbeats, meter = self._detect_downbeats(audio_file, beats)
        
        if progress_callback:
            progress_callback(1.0)
            
        return Beats(
            timestamps=original_beats,
            downbeats=downbeats,
            meter=meter,
            intro_end_idx=intro_end_idx,
            ending_start_idx=ending_start_idx,
            stats=stats,
            irregular_beats=irregular_beats
        )
    
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
    
    def _detect_downbeats(self, audio_file: str, beats: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Detect downbeats in an audio file.
        
        Parameters:
        -----------
        audio_file : str
            Path to the input audio file
        beats : np.ndarray
            Array of beat timestamps
            
        Returns:
        --------
        Tuple[np.ndarray, int]
            Tuple containing:
                - Array of downbeat indices
                - Detected meter (time signature numerator)
        """
        # Detect downbeats using RNN
        downbeat_activations = self.downbeat_processor(audio_file)
        
        # Create a dedicated TempoHistogramProcessor instance to avoid deprecation warning
        tempo_histogram_processor = CombFilterTempoHistogramProcessor(
            min_bpm=self.min_bpm,
            max_bpm=self.max_bpm
        )
        
        downbeat_tracking = DBNDownBeatTrackingProcessor(
            min_bpm=self.min_bpm,
            max_bpm=self.max_bpm,
            histogram_processor=tempo_histogram_processor
        )
        
        downbeats = downbeat_tracking(downbeat_activations)
        
        # If downbeat detection failed, fall back to simple estimation
        if len(downbeats) == 0:
            # Estimate downbeats based on beat intervals
            intervals = np.diff(beats)
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            # Find potential downbeats by looking for longer intervals
            potential_downbeats = []
            for i in range(len(intervals)):
                if intervals[i] > mean_interval + std_interval:
                    potential_downbeats.append(i + 1)
            
            if potential_downbeats:
                # Use the most common interval between potential downbeats to estimate meter
                downbeat_intervals = np.diff(potential_downbeats)
                meter = int(np.median(downbeat_intervals))
                
                # Generate downbeats based on the estimated meter
                downbeats = np.arange(0, len(beats), meter)
            else:
                # If no potential downbeats found, assume 4/4 time
                meter = 4
                downbeats = np.arange(0, len(beats), meter)
        else:
            # Use the most common interval between detected downbeats to estimate meter
            downbeat_intervals = np.diff(downbeats)
            meter = int(np.median(downbeat_intervals))
            
        return downbeats, meter
    
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
                tempo_bpm=0,
                total_beats=0
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
            tempo_bpm=60 / median_interval,  # Convert interval to BPM
            total_beats=len(intervals)
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