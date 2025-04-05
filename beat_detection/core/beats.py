"""
Core beat data structures and utilities.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


class BeatCalculationError(Exception):
    """Custom exception for errors during Beats object creation."""
    pass


@dataclass
class BeatStatistics:
    """Statistics about detected beats."""
    mean_interval: float
    median_interval: float
    std_interval: float
    min_interval: float
    max_interval: float
    irregularity_percent: float  # Percentage based on interval deviation initially
    tempo_bpm: float
    total_beats: int
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy serialization."""
        return {
            'mean_interval': self.mean_interval,
            'median_interval': self.median_interval,
            'std_interval': self.std_interval,
            'min_interval': self.min_interval,
            'max_interval': self.max_interval,
            'irregularity_percent': self.irregularity_percent,
            'tempo_bpm': self.tempo_bpm,
            'total_beats': self.total_beats
        }


@dataclass
class BeatInfo:
    """Information about a single detected beat."""
    timestamp: float
    index: int
    is_irregular_interval: bool = False # Irregular based on time interval from previous beat
    beat_count: int = 0                 # 1-based count within the measure, 0 for undetermined/irregular
    is_downbeat: bool = False           # Whether this beat was marked as a downbeat in the input
    
    @property
    def is_irregular(self) -> bool:
        """Returns True if the beat is irregular for any reason."""
        return self.is_irregular_interval or self.beat_count == 0
        
    def to_dict(self) -> Dict:
        """Convert BeatInfo object to a dictionary."""
        return {
            "timestamp": self.timestamp,
            "index": self.index,
            "is_downbeat": self.is_downbeat,
            "is_irregular_interval": self.is_irregular_interval,
            "is_irregular": self.is_irregular,
            "beat_count": self.beat_count,
        }


@dataclass
class Beats:
    """Container for all calculated beat-related information."""
    
    beat_list: List[BeatInfo]
    overall_stats: BeatStatistics  # Statistics for the entire track
    regular_stats: BeatStatistics  # Statistics for the regular section only
    meter: int
    tolerance_percent: float # The percentage tolerance used for interval calculations
    tolerance_interval: float # The calculated absolute tolerance in seconds
    min_consistent_measures: int # The minimum number of consistent measures required for analysis
    start_regular_beat_idx: int # Index of the first beat considered part of the regular section
    end_regular_beat_idx: int   # Index+1 of the last beat considered part of the regular section (exclusive index)

    @classmethod
    def from_timestamps(cls, 
                        timestamps: np.ndarray, 
                        downbeat_indices: np.ndarray, 
                        meter: int, 
                        tolerance_percent: float = 10.0,
                        min_consistent_measures: int = 5 # Minimum consistent measures required
                       ) -> 'Beats':
        """
        Factory method to create a Beats object from raw timestamp data.
        Calculates statistics and irregularities.
        
        Raises:
        -------
        BeatCalculationError
            If beat statistics or counts cannot be reliably calculated.
        """
        num_beats = len(timestamps)
        end_regular_beat_idx_calc = num_beats 
        start_regular_beat_idx_calc = 0 
        
        # --- Validation Checks --- 
        if meter <= 0:
             raise BeatCalculationError(f"Invalid meter provided: {meter}. Meter must be positive.")
             
        required_beats = meter * min_consistent_measures 
        if num_beats < required_beats:
            raise BeatCalculationError(
                f"Insufficient number of beats ({num_beats}) for analysis with meter {meter}. "
                f"Requires at least {required_beats} beats ({min_consistent_measures} measures)."
            )
        # --- End Validation Checks ---
            
        beat_list: List[BeatInfo] = []

        # 1. Calculate intervals and median interval
        intervals = np.diff(timestamps)
        median_interval = np.median(intervals)

        if median_interval <= 0:
             raise BeatCalculationError(
                 f"Cannot calculate reliable beat statistics: Median interval is {median_interval:.4f}. "
                 f"Check input timestamps: {timestamps[:5]}..."
             )
             
        # 2. Calculate remaining stats and tolerance
        tempo_bpm = 60 / median_interval
        tolerance_interval_calculated = median_interval * (tolerance_percent / 100.0) # Store calculated interval
        
        # 3. Initial irregularity check based on intervals
        interval_irregularities = [False] # First beat cannot be irregular by interval
        for interval in intervals:
             interval_irregularities.append(not (median_interval - tolerance_interval_calculated <= interval <= median_interval + tolerance_interval_calculated))
        
        irregularity_percent = (sum(interval_irregularities) / num_beats) * 100

        # Calculate overall statistics
        overall_stats = BeatStatistics(
            mean_interval=np.mean(intervals),
            median_interval=median_interval,
            std_interval=np.std(intervals),
            min_interval=np.min(intervals),
            max_interval=np.max(intervals),
            irregularity_percent=irregularity_percent,
            tempo_bpm=tempo_bpm,
            total_beats=num_beats
        )

        # 4. Populate BeatInfo list and calculate beat counts/count irregularities
        beat_list = []
        last_downbeat_idx = -1
        for i, ts in enumerate(timestamps):
            is_irregular_interval = interval_irregularities[i]
            is_downbeat = i in downbeat_indices
            
            if is_downbeat:
                last_downbeat_idx = i
                beat_count = 1
            elif last_downbeat_idx == -1:
                 beat_count = 0 # Undetermined count
            else:
                beat_count = i - last_downbeat_idx + 1

            display_beat_count = beat_count
            if beat_count > meter:
                display_beat_count = 0  # Mark as undetermined if count exceeds meter

            beat_list.append(BeatInfo(
                timestamp=ts,
                index=i,
                is_irregular_interval=is_irregular_interval,
                beat_count=display_beat_count,
                is_downbeat=is_downbeat
            ))
            
        # 5. Find the longest regular sequence using the static helper
        try:
            start_idx, end_idx, _ = cls._find_longest_regular_sequence_static(
                beat_list, tolerance_percent, meter, min_consistent_measures
            )
            start_regular_beat_idx_calc = start_idx
            end_regular_beat_idx_calc = end_idx + 1 # Convert inclusive end index to exclusive
        except BeatCalculationError as e:
             # Re-raise with more context if finding the sequence failed
             raise BeatCalculationError(f"Could not determine a stable regular section: {e}") from e

        # 6. Calculate statistics for the identified regular section
        if start_regular_beat_idx_calc >= end_regular_beat_idx_calc:
             # This case might happen if the sequence finding logic has issues or input is degenerate
             raise BeatCalculationError(
                 f"Invalid regular section bounds calculated: start={start_regular_beat_idx_calc}, end={end_regular_beat_idx_calc}. "
                 f"Cannot calculate regular statistics."
             )
             
        regular_intervals = intervals[start_regular_beat_idx_calc:end_regular_beat_idx_calc-1] # Slicing intervals needs exclusive upper bound - 1
        
        # Check if regular_intervals is empty, which can happen if the regular section is just 1 beat.
        if len(regular_intervals) == 0:
             if (end_regular_beat_idx_calc - start_regular_beat_idx_calc) == 1:
                 # Handle the case of a single regular beat - stats are ill-defined or trivial
                 regular_median_interval = 0.0 # Or perhaps NaN, or based on overall median? Defaulting to 0 for now.
                 regular_tempo_bpm = 0.0 # Tempo undefined for single beat
                 regular_mean_interval = 0.0
                 regular_std_interval = 0.0
                 regular_min_interval = 0.0
                 regular_max_interval = 0.0
                 regular_irregularity_percent = 0.0 # No intervals to be irregular
             else:
                  # This case suggests an issue with indexing or logic if the section is > 1 beat but intervals are empty
                  raise BeatCalculationError(
                      f"Internal error: Regular section has {end_regular_beat_idx_calc - start_regular_beat_idx_calc} beats, "
                      f"but no intervals were extracted for statistics. Indices: {start_regular_beat_idx_calc} to {end_regular_beat_idx_calc-1}"
                  )
        else:
            regular_median_interval = np.median(regular_intervals)
            if regular_median_interval <= 0:
                 raise BeatCalculationError(
                     f"Cannot calculate reliable regular beat statistics: Median interval is {regular_median_interval:.4f}. "
                     f"Regular section indices: {start_regular_beat_idx_calc}-{end_regular_beat_idx_calc}. "
                     f"Intervals: {regular_intervals[:5]}..."
                 )
            regular_tempo_bpm = 60 / regular_median_interval
            regular_mean_interval=np.mean(regular_intervals)
            regular_std_interval=np.std(regular_intervals)
            regular_min_interval=np.min(regular_intervals)
            regular_max_interval=np.max(regular_intervals)
             # Calculate irregularity within the regular section bounds
            num_regular_beats = end_regular_beat_idx_calc - start_regular_beat_idx_calc
            regular_section_irregularities = interval_irregularities[start_regular_beat_idx_calc:end_regular_beat_idx_calc]
            regular_irregularity_percent = (sum(regular_section_irregularities) / num_regular_beats) * 100 if num_regular_beats > 0 else 0.0
        
        regular_stats = BeatStatistics(
            mean_interval=regular_mean_interval,
            median_interval=regular_median_interval,
            std_interval=regular_std_interval,
            min_interval=regular_min_interval,
            max_interval=regular_max_interval,
            irregularity_percent=regular_irregularity_percent,
            tempo_bpm=regular_tempo_bpm,
            total_beats=end_regular_beat_idx_calc - start_regular_beat_idx_calc
        )

        return cls(beat_list=beat_list, 
                   overall_stats=overall_stats,
                   regular_stats=regular_stats,
                   meter=meter, 
                   tolerance_percent=tolerance_percent, 
                   tolerance_interval=tolerance_interval_calculated,
                   min_consistent_measures=min_consistent_measures,
                   start_regular_beat_idx=start_regular_beat_idx_calc,
                   end_regular_beat_idx=end_regular_beat_idx_calc)

    def to_dict(self) -> Dict:
        """Convert the Beats object to a dictionary suitable for JSON serialization."""
        return {
            "meter": self.meter,
            "tolerance_percent": self.tolerance_percent,
            "tolerance_interval": self.tolerance_interval,
            "min_consistent_measures": self.min_consistent_measures,
            "start_regular_beat_idx": self.start_regular_beat_idx,
            "end_regular_beat_idx": self.end_regular_beat_idx,
            "overall_stats": self.overall_stats.to_dict(),
            "regular_stats": self.regular_stats.to_dict(),
            "beat_list": [beat.to_dict() for beat in self.beat_list],
        }

    def get_beat_info_at_time(self, t: float) -> Optional[BeatInfo]:
        """
        Get information about the beat active at time t.
        Returns None if time t is before the first beat.
        """
        if not self.beat_list or t < self.beat_list[0].timestamp:
            return None
            
        insert_idx = np.searchsorted([b.timestamp for b in self.beat_list], t, side='right')
        current_beat_idx = insert_idx - 1
        
        if current_beat_idx < 0:
            return None
            
        return self.beat_list[current_beat_idx]

    def get_beat_count_at_time(self, t: float) -> int:
        """Get the beat count (1-meter, or 0 if before first downbeat) at time t."""
        beat_info = self.get_beat_info_at_time(t)
        return beat_info.beat_count if beat_info else 0

    def is_downbeat_at_time(self, t: float) -> bool:
        """Check if the beat at time t is a downbeat."""
        beat_info = self.get_beat_info_at_time(t)
        return beat_info.is_downbeat if beat_info else False
        
    def is_irregular_at_time(self, t: float) -> bool:
        """Check if the beat at time t is irregular."""
        beat_info = self.get_beat_info_at_time(t)
        return beat_info.is_irregular if beat_info else False

    @property
    def timestamps(self) -> np.ndarray:
        """Return numpy array of all beat timestamps."""
        return np.array([b.timestamp for b in self.beat_list])

    @property
    def downbeat_indices(self) -> np.ndarray:
        """Return numpy array of indices corresponding to downbeats."""
        return np.array([b.index for b in self.beat_list if b.is_downbeat])
        
    @property
    def irregular_beat_indices(self) -> List[int]:
        """Return list of indices corresponding to irregular beats."""
        return [b.index for b in self.beat_list if b.is_irregular]

    def get_regular_beats(self) -> List[BeatInfo]:
        """Get a list of regular BeatInfo objects."""
        return [b for b in self.beat_list if not b.is_irregular]

    def get_irregular_beats(self) -> List[BeatInfo]:
        """Get a list of irregular BeatInfo objects."""
        return [b for b in self.beat_list if b.is_irregular]

    def get_regular_downbeats(self) -> List[BeatInfo]:
        """Get a list of regular downbeat BeatInfo objects."""
        return [b for b in self.beat_list if b.is_downbeat and not b.is_irregular]

    def get_irregular_downbeats(self) -> List[BeatInfo]:
        """Get a list of irregular downbeat BeatInfo objects."""
        return [b for b in self.beat_list if b.is_downbeat and b.is_irregular]

    @staticmethod
    def _find_longest_regular_sequence_static(beat_list: List[BeatInfo], 
                                              tolerance_percent: float, 
                                              meter: int, 
                                              min_consistent_measures: int
                                             ) -> Tuple[int, int, float]:
        """
        Find the longest sequence of beats where the interval irregularity percentage 
        is low and minimum length requirements are met.
        Only considers interval irregularities for threshold check, but excludes beats 
        with undetermined counts (beat_count == 0) entirely from potential sequences.
        
        Parameters:
        -----------
        beat_list : List[BeatInfo]
            The list of BeatInfo objects to analyze.
        tolerance_percent : float
            The tolerance percentage used to define interval regularity.
        meter: int
            The musical meter.
        min_consistent_measures: int
            Minimum number of measures the regular section must span.
            
        Returns:
        --------
        Tuple[int, int, float]
            start_idx: Index of first beat in the sequence
            end_idx: Index of last beat in the sequence (inclusive)
            irregularity_percent: Actual interval irregularity percentage in the sequence
            
        Raises:
        -------
        BeatCalculationError
            If no suitable sequence meeting the criteria is found.
        """
        if not beat_list:
            raise BeatCalculationError("Cannot find regular sequence in empty beat list")
            
        if tolerance_percent < 0 or tolerance_percent > 100:
             # Although tolerance_percent is used for calculation, treat it like threshold for validation here
            raise BeatCalculationError(f"Invalid tolerance percent: {tolerance_percent}. Must be between 0 and 100.")

        required_beats = meter * min_consistent_measures
        if len(beat_list) < required_beats:
             # This check should technically be caught earlier in from_timestamps, but good to have defensively
             raise BeatCalculationError(f"Input beat list length ({len(beat_list)}) is less than required beats ({required_beats}).")

        # Calculate median interval from the entire sequence for reference
        timestamps = np.array([b.timestamp for b in beat_list])
        intervals = np.diff(timestamps)
        if len(intervals) == 0:
             # Handle case with 0 or 1 beat
             if len(beat_list) >= required_beats: # Should not happen if required_beats > 1
                raise BeatCalculationError("Insufficient intervals to calculate median for sequence finding.")
             elif len(beat_list) > 0 : # Single beat might be valid if required_beats is 1 (meter=1, min_measures=1)
                 if required_beats <= 1:
                     return 0, 0, 0.0 # A single beat is trivially regular sequence
                 else:
                     raise BeatCalculationError(f"Only one beat found, less than required {required_beats}.")
             else: # No beats
                  raise BeatCalculationError("Cannot find regular sequence in empty beat list (no intervals).")
                  
        median_interval = np.median(intervals)
        if median_interval <= 0:
            raise BeatCalculationError(f"Median interval ({median_interval:.4f}) is non-positive, cannot determine regularity.")
            
        tolerance_interval = median_interval * (tolerance_percent / 100.0)
        # Use the tolerance_percent itself also as the irregularity threshold for simplicity here
        irregularity_threshold = tolerance_percent 

        # Initialize variables for tracking the longest valid sequence
        max_valid_length = 0
        best_start = -1
        best_end = -1
        best_irregularity = 100.0

        # Try all possible start positions
        for start in range(len(beat_list)):
            # Skip if starting beat has irregular (undetermined) count
            if beat_list[start].beat_count == 0:
                continue
                
            irregular_interval_count = 0
            
            # For each start position, try extending the sequence
            for end in range(start, len(beat_list)):
                # Stop if we hit a beat with irregular count
                if beat_list[end].beat_count == 0:
                    break # End the inner loop, sequence broken by undetermined beat count
                    
                # Check interval irregularity (only for beats after the first in the sequence)
                if end > start:
                    interval = beat_list[end].timestamp - beat_list[end-1].timestamp
                    # Use the pre-calculated tolerance_interval
                    if abs(interval - median_interval) > tolerance_interval:
                        irregular_interval_count += 1
                    
                current_length = end - start + 1
                num_intervals = current_length - 1
                
                if num_intervals > 0: 
                    current_interval_irregularity = (irregular_interval_count / num_intervals) * 100
                else:
                    current_interval_irregularity = 0.0 # Single beat sequence has 0% interval irregularity

                # Check if current sequence meets criteria:
                # 1. Interval irregularity is below threshold
                # 2. Length is sufficient (>= required_beats)
                if current_interval_irregularity <= irregularity_threshold:
                     if current_length >= required_beats:
                          # This is a *valid* sequence meeting all criteria
                          if current_length > max_valid_length:
                               # Found a new longest valid sequence
                               max_valid_length = current_length
                               best_start = start
                               best_end = end
                               best_irregularity = current_interval_irregularity
                elif num_intervals > 0 : 
                     # Exceeded interval irregularity threshold, stop extending from this start
                     # We only break if we've actually evaluated at least one interval
                     break 

        if best_start == -1: # Check if we ever found a valid sequence
            # Try to find *any* sequence even if too short, for error reporting
            min_overall_irregularity = 100.0
            found_any_sequence = False
            # (Simplified search for error message - just find minimum irregularity)
            for i in range(1, len(beat_list)):
                 if beat_list[i].beat_count != 0 and beat_list[i-1].beat_count != 0:
                     interval = beat_list[i].timestamp - beat_list[i-1].timestamp
                     if abs(interval - median_interval) <= tolerance_interval:
                          min_overall_irregularity = 0.0 # At least one regular interval exists
                          found_any_sequence = True
                          break
                     else: # Found an irregular interval
                         found_any_sequence = True 
                         # Keep track of minimum irregularity? More complex logic needed here.
                         # For simplicity, we'll just report failure to meet length/threshold.
            
            error_message = (f"No regular sequence found meeting the criteria: "
                             f"minimum length {required_beats} beats ({min_consistent_measures} measures) "
                             f"and max interval irregularity {irregularity_threshold:.1f}%.")
            if not found_any_sequence and len(beat_list) > 1:
                 error_message += " All intervals are irregular or sequences are broken by undetermined beat counts."
            elif len(beat_list) <= 1 and required_beats > len(beat_list):
                 error_message += f" Insufficient total beats ({len(beat_list)})."
            
            raise BeatCalculationError(error_message)

        return best_start, best_end, best_irregularity 