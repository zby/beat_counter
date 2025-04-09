"""
Core beat data structures and utilities.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterable
import json
from pathlib import Path


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
            'mean_interval': float(self.mean_interval),
            'median_interval': float(self.median_interval),
            'std_interval': float(self.std_interval),
            'min_interval': float(self.min_interval),
            'max_interval': float(self.max_interval),
            'irregularity_percent': float(self.irregularity_percent),
            'tempo_bpm': float(self.tempo_bpm),
            'total_beats': int(self.total_beats)
        }


@dataclass
class BeatInfo:
    """Information about a single detected beat."""
    timestamp: float
    index: int
    is_irregular_interval: bool = False # Irregular based on time interval from previous beat
    beat_count: int = 0                 # 1-based count within the measure, 0 for undetermined/irregular
    
    @property
    def is_irregular(self) -> bool:
        """Returns True if the beat is irregular for any reason."""
        return self.is_irregular_interval or self.beat_count == 0
        
    def to_dict(self) -> Dict:
        """Convert BeatInfo object to a dictionary, excluding derived properties."""
        return {
            "timestamp": float(self.timestamp),
            "index": int(self.index),
            # Ensure standard Python bool types for JSON serialization
            "is_irregular_interval": bool(self.is_irregular_interval),
            # Exclude "is_irregular" as it's a derived property
            "beat_count": int(self.beat_count),
        }


@dataclass
class Beats:
    """Container for all calculated beat-related information."""
    
    beat_list: List[BeatInfo]
    overall_stats: BeatStatistics  # Statistics for the entire track
    regular_stats: BeatStatistics  # Statistics for the regular section only
    beats_per_bar: int
    tolerance_percent: float # The percentage tolerance used for interval calculations
    tolerance_interval: float # The calculated absolute tolerance in seconds
    min_measures: int # The minimum number of consistent measures required for analysis
    start_regular_beat_idx: int # Index of the first beat considered part of the regular section
    end_regular_beat_idx: int   # Index+1 of the last beat considered part of the regular section (exclusive index)

    @classmethod
    def from_timestamps(cls, 
                        timestamps: np.ndarray, 
                        beats_per_bar: int, 
                        beat_counts: np.ndarray, # Add new parameter for pre-calculated counts
                        tolerance_percent: float = 10.0,
                        min_measures: int = 5 # Minimum consistent measures required
                       ) -> 'Beats':
        """
        Factory method to create a Beats object from raw timestamp data.
        Calculates statistics and irregularities.
        
        Raises:
        -------
        BeatCalculationError
            If beat statistics or counts cannot be reliably calculated.
        """
        # --- Input Validation --- 
        if not isinstance(timestamps, np.ndarray) or timestamps.ndim != 1:
            raise BeatCalculationError("Timestamps must be a 1D numpy array.")
        
        num_beats = len(timestamps)
        
        # Check for strictly increasing timestamps
        # The required_beats check later ensures num_beats >= 2, so np.diff is safe
        intervals = np.diff(timestamps)
        if not np.all(intervals > 0):
            # Find first non-positive interval index for better error message
            first_bad_idx = np.where(intervals <= 0)[0][0]
            raise BeatCalculationError(
                f"Timestamps must be strictly increasing. Error found after index {first_bad_idx} "
                f"(timestamps: {timestamps[first_bad_idx]:.4f} -> {timestamps[first_bad_idx+1]:.4f})"
            )

        # --- Validation Checks --- 
        if not isinstance(tolerance_percent, (int, float)) or tolerance_percent < 0:
            raise BeatCalculationError(
                f"Invalid tolerance_percent provided: {tolerance_percent}. Must be a non-negative number."
            )

        # Ensure beats_per_bar is at least 2
        if beats_per_bar <= 1:
             raise BeatCalculationError(f"Invalid beats_per_bar provided: {beats_per_bar}. Must be 2 or greater.")

        end_regular_beat_idx_calc = num_beats
        start_regular_beat_idx_calc = 0

        # This check implicitly handles num_beats <= 1 if min_measures >= 1
        required_beats = beats_per_bar * min_measures
        if num_beats < required_beats:
            raise BeatCalculationError(
                f"Insufficient number of beats ({num_beats}) for analysis with beats_per_bar {beats_per_bar}. "
                f"Requires at least {required_beats} beats ({min_measures} measures)."
            )
        # --- End Validation Checks ---

        beat_list: List[BeatInfo] = []

        # 1. Calculate intervals and median interval
        # This will be empty if num_beats <= 1, handled later by stats calculations
        intervals = np.diff(timestamps)

        # Calculate median interval - needed for tolerance
        # Raises error if intervals is empty, but num_beats < required_beats check prevents this
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

        # Calculate overall statistics (temporarily, irregularity_percent recalculated later)
        initial_irregularity_percent = (sum(interval_irregularities) / num_beats) * 100 if num_beats > 0 else 0.0
        overall_stats = BeatStatistics(
            mean_interval=np.mean(intervals) if len(intervals) > 0 else 0.0,
            median_interval=median_interval,
            std_interval=np.std(intervals) if len(intervals) > 0 else 0.0,
            min_interval=np.min(intervals) if len(intervals) > 0 else 0.0,
            max_interval=np.max(intervals) if len(intervals) > 0 else 0.0,
            irregularity_percent=initial_irregularity_percent, # Will be updated later
            tempo_bpm=tempo_bpm,
            total_beats=num_beats
        )

        # 4. Populate BeatInfo list using provided beat_counts
        beat_list = []
        for i, ts in enumerate(timestamps):
            is_irregular_interval = interval_irregularities[i]
            # Use the provided beat count directly
            original_beat_count = beat_counts[i]
            
            # Apply the same logic as before: counts > beats_per_bar are considered irregular (display as 0)
            display_beat_count = original_beat_count
            if original_beat_count > beats_per_bar or original_beat_count <= 0: # Also treat non-positive counts as irregular
                display_beat_count = 0
                # We might also want to flag this beat specifically if needed,
                # but relying on display_beat_count == 0 for irregularity check should suffice.
                
            beat_list.append(BeatInfo(
                timestamp=ts,
                index=i,
                is_irregular_interval=is_irregular_interval,
                beat_count=display_beat_count, # Use the adjusted count
            ))
            
        # 5. Recalculate overall irregularity based on the final beat_list status
        # This now includes irregularities from counts (beat_count == 0)
        final_irregular_count = sum(1 for beat in beat_list if beat.is_irregular)
        final_irregularity_percent = (final_irregular_count / num_beats) * 100 if num_beats > 0 else 0.0
        # Update overall_stats with the final irregularity percentage
        # Keep the original interval-based irregularity in overall_stats if needed for specific reporting,
        # but the primary irregularity measure should reflect the final state.
        # For simplicity, we'll overwrite it here. If needed, store both.
        overall_stats.irregularity_percent = final_irregularity_percent

        # 6. Find the longest regular sequence using the static helper
        try:
            start_idx, end_idx, _ = cls._find_longest_regular_sequence_static(
                beat_list, tolerance_percent, beats_per_bar
            )
            start_regular_beat_idx_calc = start_idx
            end_regular_beat_idx_calc = end_idx + 1 # Convert inclusive end index to exclusive
            
            # Check if the found sequence meets the minimum length requirement
            sequence_length = end_regular_beat_idx_calc - start_regular_beat_idx_calc
            required_beats = beats_per_bar * min_measures
            if sequence_length < required_beats:
                raise BeatCalculationError(
                    f"Longest regular sequence found ({sequence_length} beats) is shorter than required "
                    f"({required_beats} beats = {min_measures} measures of {beats_per_bar}/X time)."
                )
        except BeatCalculationError as e:
             # Re-raise with more context if finding the sequence failed
             raise BeatCalculationError(f"Could not determine a stable regular section: {e}") from e

        # 7. Calculate statistics for the identified regular section
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
                   beats_per_bar=beats_per_bar, 
                   tolerance_percent=tolerance_percent, 
                   tolerance_interval=tolerance_interval_calculated,
                   min_measures=min_measures,
                   start_regular_beat_idx=start_regular_beat_idx_calc,
                   end_regular_beat_idx=end_regular_beat_idx_calc)

    def to_dict(self) -> Dict:
        """Convert the Beats object to a dictionary suitable for JSON serialization."""
        return {
            "beats_per_bar": int(self.beats_per_bar),
            "tolerance_percent": float(self.tolerance_percent),
            "tolerance_interval": float(self.tolerance_interval),
            "min_measures": int(self.min_measures),
            "start_regular_beat_idx": int(self.start_regular_beat_idx),
            "end_regular_beat_idx": int(self.end_regular_beat_idx),
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

    def get_info_at_time(self, t: float) -> Tuple[int, float, int]:
        """
        Get count, time since last beat, and beat index at time t.
        
        Returns a tuple of (count, time_since_beat, beat_idx) where:
        - count: The beat count (0 if before first beat, irregular, or not in regular section)
        - time_since_beat: Time in seconds since the last beat (0.0 if before first beat)
        - beat_idx: Index of the beat in the beat_list (-1 if before first beat)
        
        The count will be 0 for any beats outside the regular interval or for irregular beats,
        otherwise it will be the beat_count from the BeatInfo.
        """
        beat_info = self.get_beat_info_at_time(t)
        
        # If no beat found or before first beat
        if beat_info is None:
            return 0, 0.0, -1
            
        # Extract beat index
        beat_idx = beat_info.index
        
        # Return 0 count if the beat is outside the regular interval or is irregular
        if beat_idx < self.start_regular_beat_idx or beat_idx > self.end_regular_beat_idx or beat_info.is_irregular:
            return 0, t - beat_info.timestamp, beat_idx
            
        # Return count and time since beat for regular intervals
        return beat_info.beat_count, t - beat_info.timestamp, beat_idx

    @property
    def timestamps(self) -> np.ndarray:
        """Return numpy array of all beat timestamps."""
        return np.array([b.timestamp for b in self.beat_list])

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

    def iterate_beats(self) -> Iterable[Tuple[float, int]]:
        """
        Iterate through beats, yielding timestamp and beat count.

        Yields:
        -------
        Tuple[float, int]
            A tuple containing (timestamp, beat_count) for each beat.
        """
        for beat in self.beat_list:
            yield beat.timestamp, beat.beat_count

    @staticmethod
    def _find_longest_regular_sequence_static(beat_list: List[BeatInfo],
                                              tolerance_percent: float,
                                              beats_per_bar: int
                                             ) -> Tuple[int, int, float]:
        """
        Find the longest sequence of beats where intervals are regular and beat counts are determined.

        Regularity Conditions:
        1. The sequence must start with a downbeat (beat_count == 1).
        2. All beats within the sequence must have `beat_count > 0`.
        3. The time interval between any two consecutive beats in the sequence must be
           within `tolerance_interval` of the overall `median_interval`.
        4. Beat counts must increment correctly (1, 2, ..., beats_per_bar, 1, ...).

        Parameters:
        -----------
        beat_list : List[BeatInfo]
            The list of BeatInfo objects to analyze.
        tolerance_percent : float
            The tolerance percentage used to define interval regularity relative to the median.
        beats_per_bar : int
            The time signature's upper numeral (e.g., 4 for 4/4 time).

        Returns:
        --------
        Tuple[int, int, float]
            start_idx: Index of first beat in the longest sequence
            end_idx: Index of last beat in the longest sequence (inclusive)
            irregularity_percent: Actual interval irregularity percentage in the sequence
                                  (calculated *after* finding the sequence).

        Raises:
        -------
        BeatCalculationError
            If no regular sequence is found, or if basic calculations fail.
        """
        if not beat_list:
            raise BeatCalculationError("Cannot find regular sequence in empty beat list.")

        if beats_per_bar <= 1:
            raise BeatCalculationError(f"Invalid beats_per_bar ({beats_per_bar}) passed to sequence finder. Must be > 1.")

        if not (0 <= tolerance_percent <= 100):
            raise BeatCalculationError(f"Invalid tolerance percent: {tolerance_percent}. Must be between 0 and 100.")

        # Calculate overall median interval for reference
        timestamps = np.array([b.timestamp for b in beat_list])
        intervals = np.diff(timestamps)

        # The num_beats < required_beats check in from_timestamps should prevent
        # len(intervals) == 0 if min_measures >= 1 and beats_per_bar >= 2.
        # However, keep a check for safety, but it shouldn't return a valid sequence.
        if len(intervals) == 0:
            raise BeatCalculationError(
                "Cannot find regular sequence: Not enough intervals to analyze (requires at least 1 interval)."
            )

        median_interval = np.median(intervals)
        if median_interval <= 0:
            raise BeatCalculationError(f"Median interval ({median_interval:.4f}) is non-positive, cannot determine regularity.")

        tolerance_interval = median_interval * (tolerance_percent / 100.0)

        # --- Single Pass Logic ---
        best_start = -1
        best_end = -1
        max_len = 0

        # Only start a sequence from a downbeat (beat count 1)
        current_start = 0 if beat_list[0].beat_count == 1 else -1

        # Start from the second beat (if available)
        for i in range(1, len(beat_list)):
            # First, check if we're starting a new sequence
            if current_start == -1:
                # Only start new sequences at downbeats (beat_count == 1)
                if beat_list[i].beat_count == 1:
                    current_start = i
            # If we're in an existing sequence, check interval regularity
            elif beat_list[i].beat_count > 0:
                interval = beat_list[i].timestamp - beat_list[i-1].timestamp
                prev_count = beat_list[i-1].beat_count
                current_count = beat_list[i].beat_count
                expected_next_count = (prev_count % beats_per_bar) + 1

                # Check BOTH interval regularity AND correct count sequence
                is_interval_regular = abs(interval - median_interval) <= tolerance_interval
                is_count_correct = (current_count == expected_next_count)

                if not (is_interval_regular and is_count_correct):
                    # Irregular interval OR incorrect count - end the sequence
                    current_len = (i - 1) - current_start + 1
                    if current_len > max_len:
                        max_len = current_len
                        best_start = current_start
                        best_end = i - 1
                    # Look for a new sequence only if this beat is a downbeat
                    current_start = i if beat_list[i].beat_count == 1 else -1
            else:
                # Irregular beat_count (zero) - end the sequence
                current_len = (i - 1) - current_start + 1
                if current_len > max_len:
                    max_len = current_len
                    best_start = current_start
                    best_end = i - 1
                # Look for a new sequence only if this beat is a downbeat
                current_start = i if beat_list[i].beat_count == 1 else -1

        # --- Check the last sequence after the loop finishes ---
        if current_start != -1:
            current_len = len(beat_list) - current_start
            if current_len > max_len:
                max_len = current_len
                best_start = current_start
                best_end = len(beat_list) - 1

        if best_start == -1:
            raise BeatCalculationError(
                f"No regular sequence found with the given tolerance ({tolerance_percent}%)."
            )

        # Calculate irregularity percentage for the *found* best sequence
        best_sequence_intervals = np.diff([b.timestamp for b in beat_list[best_start:best_end+1]])
        irregular_interval_count = 0
        if len(best_sequence_intervals) > 0:
             for interval in best_sequence_intervals:
                 if abs(interval - median_interval) > tolerance_interval:
                      irregular_interval_count += 1
             best_irregularity = (irregular_interval_count / len(best_sequence_intervals)) * 100
        else:
             best_irregularity = 0.0 # Single beat sequence has 0% irregularity

        return best_start, best_end, best_irregularity

    # --- Serialization Methods ---

    def save_to_file(self, file_path: Path):
        """Serialize the Beats object to a JSON file."""
        file_path = Path(file_path) # Ensure it's a Path object
        file_path.parent.mkdir(parents=True, exist_ok=True) # Create directory if needed
        data_dict = self.to_dict()
        try:
            with file_path.open('w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=4)
        except IOError as e:
            raise BeatCalculationError(f"Error writing Beats object to {file_path}: {e}") from e
        except TypeError as e:
             # This might happen if numpy types weren't properly converted in to_dict
             raise BeatCalculationError(f"Error serializing Beats object data to JSON: {e}") from e

    @classmethod
    def load_from_file(cls, file_path: Path) -> 'Beats':
        """Deserialize a Beats object from a JSON file."""
        file_path = Path(file_path) # Ensure it's a Path object
        if not file_path.is_file():
             raise FileNotFoundError(f"Beats file not found: {file_path}")
             
        try:
            with file_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
        except IOError as e:
            raise BeatCalculationError(f"Error reading Beats file {file_path}: {e}") from e
        except json.JSONDecodeError as e:
            raise BeatCalculationError(f"Error decoding JSON from Beats file {file_path}: {e}") from e
            
        try:
            # Reconstruct BeatInfo list
            beat_list = [BeatInfo(**beat_data) for beat_data in data['beat_list']]
            
            # Reconstruct BeatStatistics
            overall_stats = BeatStatistics(**data['overall_stats'])
            regular_stats = BeatStatistics(**data['regular_stats'])
            
            # Explicitly convert/validate types before final instantiation
            try:
                beats_per_bar_val = int(data['beats_per_bar'])
                tolerance_percent_val = float(data['tolerance_percent'])
                tolerance_interval_val = float(data['tolerance_interval'])
                min_measures_val = int(data['min_measures'])
                start_regular_beat_idx_val = int(data['start_regular_beat_idx'])
                end_regular_beat_idx_val = int(data['end_regular_beat_idx'])
            except (ValueError, TypeError) as e:
                # Specific handling for type conversion errors
                raise BeatCalculationError(f"Type or value error reconstructing Beats object from {file_path}: {e}") from e
            
            # Create the Beats object - ensure all required fields are present
            instance = cls(
                beat_list=beat_list,
                overall_stats=overall_stats,
                regular_stats=regular_stats,
                beats_per_bar=beats_per_bar_val,
                tolerance_percent=tolerance_percent_val,
                tolerance_interval=tolerance_interval_val,
                min_measures=min_measures_val,
                start_regular_beat_idx=start_regular_beat_idx_val,
                end_regular_beat_idx=end_regular_beat_idx_val
            )
            return instance # Return the successfully created instance
        except KeyError as e:
            raise BeatCalculationError(f"Missing expected key '{e}' in Beats file {file_path}. File may be corrupt or incompatible.") from e
        except (TypeError, ValueError) as e: # Catch remaining conversion errors
            # Might happen if data types in file don't match dataclass fields or fail conversion
            raise BeatCalculationError(f"Type or value error reconstructing Beats object from {file_path}: {e}") from e

    # --- End Serialization Methods --- 