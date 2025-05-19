"""
Core beat data structures and utilities.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterable, Type, TypeVar
import json
from pathlib import Path
import dataclasses

# Define TypeVar for RawBeats *before* Beats class uses it
T_RawBeats = TypeVar("T_RawBeats", bound="RawBeats")

MAX_START_TIME = 30.0

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
    irregularity_percent: float  # Percentage based on beats with count == 0
    tempo_bpm: float
    total_beats: int

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy serialization."""
        return {
            "mean_interval": float(self.mean_interval),
            "median_interval": float(self.median_interval),
            "std_interval": float(self.std_interval),
            "min_interval": float(self.min_interval),
            "max_interval": float(self.max_interval),
            "irregularity_percent": float(self.irregularity_percent),
            "tempo_bpm": float(self.tempo_bpm),
            "total_beats": int(self.total_beats),
        }


@dataclass
class Beats:
    """Container for all calculated beat-related information."""

    beat_data: (
        np.ndarray
    )  # Shape (N, 2): Column 0=timestamp, Column 1=beat_count (0 for irregular/undetermined)
    overall_stats: BeatStatistics  # Statistics for the entire track
    regular_stats: BeatStatistics  # Statistics for the regular section only
    beats_per_bar: int
    tolerance_percent: float  # The percentage tolerance used for interval calculations
    tolerance_interval: float  # The calculated absolute tolerance in seconds
    min_measures: int  # The minimum number of consistent measures required for analysis
    start_regular_beat_idx: (
        int  # Index of the first beat considered part of the regular section
    )
    end_regular_beat_idx: int  # Index+1 of the last beat considered part of the regular section (exclusive index)
    clip_length: float  # Total length of the audio clip in seconds

    def __init__(
        self,
        raw_beats: "RawBeats", # Use forward reference string
        beats_per_bar: Optional[int] = None,
        tolerance_percent: float = 10.0,
        min_measures: int = 5,  # Minimum consistent measures required
        max_start_time: float = MAX_START_TIME,
    ):
        """
        Initializes the Beats object from raw beat data.
        Calculates statistics and identifies the longest regular section.
        Infers beats_per_bar from raw_beats.beat_counts if not provided.

        Raises:
        -------\
        BeatCalculationError
            If beat statistics or the regular section cannot be reliably calculated,
            or if beats_per_bar cannot be inferred (e.g., empty counts).
        ValueError
            If input RawBeats data is invalid (handled by RawBeats __post_init__).
        """
        # --- Input Validation (RawBeats validation happens in its __post_init__) ---
        if not isinstance(raw_beats, RawBeats):
            raise TypeError("Input must be a RawBeats object.")

        timestamps = raw_beats.timestamps
        input_beat_counts = raw_beats.beat_counts # Use original counts for inference
        num_beats = len(timestamps)
        self.clip_length = raw_beats.clip_length  # Store clip_length from raw_beats

        # --- Infer beats_per_bar if not provided ---
        if beats_per_bar is None:
            if num_beats == 0 or len(input_beat_counts) == 0:
                raise BeatCalculationError(
                    "Cannot infer beats_per_bar: No beats provided in RawBeats."
                )
            # Exclude potential zero counts (used for errors/flags) from inference
            valid_counts = input_beat_counts[input_beat_counts > 0]
            if len(valid_counts) == 0:
                 raise BeatCalculationError(
                    "Cannot infer beats_per_bar: No valid (non-zero) beat counts found."
                )
            inferred_bpb = int(np.max(valid_counts))
            if inferred_bpb <= 1:
                 raise BeatCalculationError(
                    f"Inferred beats_per_bar ({inferred_bpb}) is invalid. Must be > 1."
                )
            self.beats_per_bar = inferred_bpb
        else:
             # --- Validate provided beats_per_bar ---
            if not isinstance(beats_per_bar, int) or beats_per_bar <= 1:
                raise BeatCalculationError(
                    f"Invalid beats_per_bar provided: {beats_per_bar}. Must be an integer > 1."
                )
            self.beats_per_bar = beats_per_bar

        # Continue with existing validation/calculation logic, using self.beats_per_bar
        # --- Further Validation ---
        if not isinstance(tolerance_percent, (int, float)) or tolerance_percent < 0:
            raise BeatCalculationError(
                f"Invalid tolerance_percent provided: {tolerance_percent}. Must be a non-negative number."
            )

        # 1. Prepare beat_data array using the determined/validated beats_per_bar
        # Apply the logic: counts > self.beats_per_bar or <= 0 are considered irregular (represented as 0)
        processed_beat_counts = np.where(
            (input_beat_counts > 0) & (input_beat_counts <= self.beats_per_bar),
            input_beat_counts,
            0,  # Mark invalid/out-of-range counts as 0
        ).astype(
            int
        )  # Ensure integer type

        # Stack timestamps and processed counts into the main data array
        self.beat_data = np.stack((timestamps, processed_beat_counts), axis=1)

        # 2. Calculate intervals and overall median interval
        if num_beats <= 1:
            # Handle cases with 0 or 1 beat
            median_interval = 0.0
            tolerance_interval_calculated = 0.0
            overall_stats = BeatStatistics(
                mean_interval=0.0,
                median_interval=0.0,
                std_interval=0.0,
                min_interval=0.0,
                max_interval=0.0,
                irregularity_percent=(\
                    100.0 if num_beats == 1 and processed_beat_counts[0] == 0 else 0.0
                ), # 1 beat is regular if count is valid
                tempo_bpm=0.0,
                total_beats=num_beats,
            )
            # Cannot find a regular sequence if fewer than 2 beats
            start_regular_beat_idx_calc = 0
            end_regular_beat_idx_calc = num_beats # Treat as all irregular/degenerate
            regular_stats = BeatStatistics( # Empty stats
                mean_interval=0.0,
                median_interval=0.0,
                std_interval=0.0,
                min_interval=0.0,
                max_interval=0.0,
                irregularity_percent=0.0,
                tempo_bpm=0.0,
                total_beats=0,
            )

            # We still need to check min_measures, even if we can't find a regular sequence
            required_beats = self.beats_per_bar * min_measures
            if num_beats < required_beats:
                 raise BeatCalculationError(
                    f"Insufficient number of beats ({num_beats}) for analysis with beats_per_bar {self.beats_per_bar}. "
                    f"Requires at least {required_beats} beats ({min_measures} measures)."
                 )

            # Assign calculated values and return early for 0 or 1 beat cases
            self.overall_stats = overall_stats
            self.regular_stats = regular_stats
            self.tolerance_percent = tolerance_percent
            self.tolerance_interval = tolerance_interval_calculated
            self.min_measures = min_measures
            self.start_regular_beat_idx = start_regular_beat_idx_calc
            self.end_regular_beat_idx = end_regular_beat_idx_calc
            return # Exit __init__

        # --- Calculations for 2+ beats ---
        intervals = np.diff(self.beat_data[:, 0]) # Use timestamp column
        median_interval = np.median(intervals)
        if median_interval <= 0:
            first_bad_idx = np.where(np.diff(timestamps) <= 0)[0][0]
            raise BeatCalculationError(
                f"Cannot calculate reliable beat statistics: Median interval is {median_interval:.4f}. "
                f"Timestamp sequence issue near index {first_bad_idx} "
                f"(timestamps: {timestamps[first_bad_idx]:.4f} -> {timestamps[first_bad_idx+1]:.4f})?"
            )

        tempo_bpm = 60 / median_interval
        tolerance_interval_calculated = median_interval * (tolerance_percent / 100.0)

        # 3. Calculate overall statistics (irregularity based on count==0)
        irregular_count = np.sum(self.beat_data[:, 1] == 0) # Count where beat_count is 0
        overall_irregularity_percent = (\
            (irregular_count / num_beats) * 100 if num_beats > 0 else 0.0
        )

        overall_stats = BeatStatistics(
            mean_interval=np.mean(intervals),
            median_interval=median_interval,
            std_interval=np.std(intervals),
            min_interval=np.min(intervals),
            max_interval=np.max(intervals),
            irregularity_percent=overall_irregularity_percent,
            tempo_bpm=tempo_bpm,
            total_beats=num_beats,
        )

        # 4. Find the longest regular sequence using the static helper
        # Pass the beat_data array directly
        # Specify max_start_time
        start_idx, end_idx, _ = self._find_longest_regular_sequence_static(
            self.beat_data,
            self.beats_per_bar,
            median_interval,
            tolerance_interval_calculated,
            max_start_time=max_start_time, # Pass max_start_time argument
        )
        start_regular_beat_idx_calc = start_idx
        end_regular_beat_idx_calc = (\
            end_idx + 1
        ) # Convert inclusive end index to exclusive

        # Check if the found sequence meets the minimum length requirement
        sequence_length = end_regular_beat_idx_calc - start_regular_beat_idx_calc
        required_beats = self.beats_per_bar * min_measures
        if sequence_length < required_beats:
            # This error will now propagate directly
            raise BeatCalculationError(
                f"Longest regular sequence found ({sequence_length} beats from index {start_idx} to {end_idx}) is shorter than required "
                f"({required_beats} beats = {min_measures} measures of {self.beats_per_bar}/X time)."
            )

        # 5. Calculate statistics for the identified regular section
        if start_regular_beat_idx_calc >= end_regular_beat_idx_calc:
            # This case should ideally be prevented by the sequence length check above,
            # but keep as a safeguard.
             raise BeatCalculationError(
                f"Invalid regular section bounds calculated: start={start_regular_beat_idx_calc}, end={end_regular_beat_idx_calc} "
                f"(length {end_regular_beat_idx_calc - start_regular_beat_idx_calc}). Cannot calculate regular statistics."
            )


        # Extract timestamps for the regular section
        regular_timestamps = self.beat_data[\
            start_regular_beat_idx_calc:end_regular_beat_idx_calc, 0
        ]
        num_regular_beats = len(regular_timestamps)

        # The sequence length check above ensures num_regular_beats >= required_beats.
        # If required_beats >= 2 (which it should be if beats_per_bar > 1, min_measures >= 1),
        # then num_regular_beats will be >= 2.
        # Therefore, we don't need the explicit check for num_regular_beats <= 1 here.

        regular_intervals = np.diff(regular_timestamps)
        if (\
            len(regular_intervals) == 0 and num_regular_beats > 1
        ): # Should not happen if num_regular_beats > 1
            raise BeatCalculationError(\
                f"Internal error: Regular section has {num_regular_beats} beats, "\
                f"but no intervals were extracted for statistics. Indices: {start_regular_beat_idx_calc} to {end_regular_beat_idx_calc-1}"\
            )
        elif num_regular_beats <= 1: # Handle case where required_beats was 1 (e.g. bpb=1, min_measures=1 - though bpb=1 is disallowed)
             regular_mean_interval = 0.0
             regular_median_interval = 0.0
             regular_std_interval = 0.0
             regular_min_interval = 0.0
             regular_max_interval = 0.0
             regular_tempo_bpm = 0.0
             regular_irregularity_percent = 0.0
        else:
            regular_median_interval = np.median(regular_intervals)
            if regular_median_interval <= 0:
                raise BeatCalculationError(\
                    f"Cannot calculate reliable regular beat statistics: Median interval is {regular_median_interval:.4f}. "\
                    f"Regular section indices: {start_regular_beat_idx_calc}-{end_regular_beat_idx_calc}. "\
                    f"Intervals: {regular_intervals[:5]}..."\
                )
            regular_tempo_bpm = 60 / regular_median_interval
            regular_mean_interval = np.mean(regular_intervals)
            regular_std_interval = np.std(regular_intervals)
            regular_min_interval = np.min(regular_intervals)
            regular_max_interval = np.max(regular_intervals)

            # Calculate irregularity within the regular section (based on count==0)
            regular_section_counts = self.beat_data[\
                start_regular_beat_idx_calc:end_regular_beat_idx_calc, 1
            ]
            regular_section_irregular_count = np.sum(regular_section_counts == 0)
            regular_irregularity_percent = (\
                regular_section_irregular_count / num_regular_beats
            ) * 100

        regular_stats = BeatStatistics(
            mean_interval=regular_mean_interval,
            median_interval=regular_median_interval,
            std_interval=regular_std_interval,
            min_interval=regular_min_interval,
            max_interval=regular_max_interval,
            irregularity_percent=regular_irregularity_percent, # Based on count==0 within the section
            tempo_bpm=regular_tempo_bpm,
            total_beats=num_regular_beats,
        )

        # 6. Assign final instance attributes
        self.overall_stats = overall_stats
        self.regular_stats = regular_stats
        # self.beats_per_bar = beats_per_bar # Assigned earlier
        self.tolerance_percent = tolerance_percent
        self.tolerance_interval = tolerance_interval_calculated
        self.min_measures = min_measures
        self.start_regular_beat_idx = start_regular_beat_idx_calc
        self.end_regular_beat_idx = end_regular_beat_idx_calc

    def to_dict(self) -> Dict:
        """Convert the Beats object to a dictionary suitable for JSON serialization."""
        # Convert beat_data numpy array to a list of dictionaries for better readability/compatibility
        beat_list_dict = [
            {"timestamp": float(ts), "count": int(count)}
            for ts, count in self.beat_data
        ]

        return {
            "beats_per_bar": int(self.beats_per_bar),
            "tolerance_percent": float(self.tolerance_percent),
            "tolerance_interval": float(self.tolerance_interval),
            "min_measures": int(self.min_measures),
            "start_regular_beat_idx": int(self.start_regular_beat_idx),
            "end_regular_beat_idx": int(self.end_regular_beat_idx),
            "overall_stats": self.overall_stats.to_dict(),
            "regular_stats": self.regular_stats.to_dict(),
            # "beat_list": [beat.to_dict() for beat in self.beat_list], # Old version
            "beat_list": beat_list_dict,  # Store the converted list
            "clip_length": float(self.clip_length),
        }

    def get_info_at_time(self, t: float) -> Tuple[int, float, int]:
        """
        Get count, time since last beat, and beat index at time t.

        Returns a tuple of (count, time_since_beat, beat_idx) where:
        - count: The beat count (0 if before first beat, irregular, outside regular section, or too far past last beat)
        - time_since_beat: Time in seconds since the last beat (0.0 if before first beat or too far past last beat)
        - beat_idx: Index of the beat in the beat_data array (-1 if before first beat, -1 if too far past last beat)
        """
        if self.beat_data.shape[0] == 0 or t < self.beat_data[0, 0]:
            return 0, 0.0, -1  # Before first beat or no beats

        # Check if we're too far past the last beat
        last_beat_time = self.beat_data[-1, 0]
        if t > last_beat_time + self.tolerance_interval:
            return 0, 0.0, -1  # Too far past last beat

        timestamps = self.beat_data[:, 0]
        # Find the index of the beat *before* or *at* time t
        insert_idx = np.searchsorted(timestamps, t, side="right")
        current_beat_idx = insert_idx - 1

        if current_beat_idx < 0:
            # This case should be covered by the initial check, but included for safety
            return 0, 0.0, -1

        timestamp_at_idx = self.beat_data[current_beat_idx, 0]
        count_at_idx = int(self.beat_data[current_beat_idx, 1])  # Ensure python int
        time_since_beat = t - timestamp_at_idx

        # Check if the beat is outside the identified regular section or marked irregular (count == 0)
        is_irregular = (
            current_beat_idx < self.start_regular_beat_idx
            or current_beat_idx
            >= self.end_regular_beat_idx  # Use >= for exclusive end index
            or count_at_idx == 0
        )

        if is_irregular:
            return 0, time_since_beat, current_beat_idx
        else:
            return count_at_idx, time_since_beat, current_beat_idx

    @property
    def timestamps(self) -> np.ndarray:
        """Return numpy array of all beat timestamps."""
        return self.beat_data[:, 0]

    @property
    def counts(self) -> np.ndarray:
        """Return numpy array of all beat counts."""
        return self.beat_data[:, 1]

    @property
    def irregular_beat_indices(self) -> np.ndarray:
        """Return numpy array of indices corresponding to irregular beats (count == 0)."""
        # Find where count is 0
        indices = np.where(self.beat_data[:, 1] == 0)[0]
        return indices

    def iterate_beats(self) -> Iterable[Tuple[float, int]]:
        """
        Iterate through beats, yielding timestamp and beat count.

        Yields:
        -------
        Tuple[float, int]
            A tuple containing (timestamp, beat_count) for each beat.
        """
        for i in range(self.beat_data.shape[0]):
            yield float(self.beat_data[i, 0]), int(self.beat_data[i, 1])

    @staticmethod
    def _find_longest_regular_sequence_static(
        beat_data: np.ndarray,  # Changed from beat_list
        beats_per_bar: int,
        # Pass median and tolerance to avoid recalculation
        median_interval: float,
        tolerance_interval: float,
        max_start_time: float, # New parameter: maximum allowed start time for the sequence
    ) -> Tuple[int, int, float]:
        """
        Find the longest sequence of beats starting no later than `max_start_time` where
        intervals are regular and beat counts are valid and sequential.

        Regularity Conditions:
        1. The sequence must start with a downbeat (beat_count == 1).
        2. The timestamp of the first beat (`beat_data[start_idx, 0]`) must be `<= max_start_time`.
        3. All beats within the sequence must have `beat_count > 0`.
        4. The time interval between consecutive beats must be within `tolerance_interval` of the overall `median_interval`.
        5. Beat counts must increment correctly (1, 2, ..., beats_per_bar, 1, ...).

        Parameters:
        -----------
        beat_data : np.ndarray
            The array containing timestamps (col 0) and beat counts (col 1).
        beats_per_bar : int
            The time signature's upper numeral.
        median_interval : float
            Pre-calculated median interval for the whole dataset.
        tolerance_interval : float
             Pre-calculated absolute tolerance in seconds.
        max_start_time : float
            The latest allowed timestamp for the *first beat* of the sequence.

        Returns:
        --------
        Tuple[int, int, float]
            start_idx: Index of first beat in the longest valid sequence
            end_idx: Index of last beat in the longest valid sequence (inclusive)
            irregularity_percent: Interval irregularity percentage within the sequence (unused externally now)

        Raises:
        -------
        BeatCalculationError
            If no regular sequence starting within `max_start_time` is found,
            or if basic validation fails.
        """
        num_beats = beat_data.shape[0]
        if num_beats < 2:  # Cannot have a sequence or interval with fewer than 2 beats
            raise BeatCalculationError(
                f"Cannot find regular sequence: Needs at least 2 beats, found {num_beats}."
            )

        # --- Single Pass Logic ---
        best_start = -1
        best_end = -1
        max_len = 0

        current_start = -1
        current_start_time = -1.0

        # Check if the very first beat can start a sequence *within the time limit*
        first_beat_time = beat_data[0, 0]
        first_beat_count = int(beat_data[0, 1])
        if first_beat_count == 1 and first_beat_time <= max_start_time:
            current_start = 0
            current_start_time = first_beat_time


        for i in range(1, num_beats):
            current_timestamp = beat_data[i, 0]
            current_count = int(beat_data[i, 1])  # Ensure python int for logic
            prev_timestamp = beat_data[i - 1, 0]
            prev_count = int(beat_data[i - 1, 1])  # Ensure python int

            interval = current_timestamp - prev_timestamp

            is_interval_regular = abs(interval - median_interval) <= tolerance_interval

            if current_start != -1:
                # --- Currently in a potential sequence ---
                expected_next_count = (prev_count % beats_per_bar) + 1
                is_count_correct = current_count == expected_next_count
                is_current_count_valid = current_count > 0

                sequence_ended = not (is_interval_regular and is_count_correct and is_current_count_valid)

                if sequence_ended:
                    # Irregularity found - End the current sequence
                    current_len = i - current_start # Length = (i-1) - current_start + 1

                    # Check if this completed sequence is the best *valid* one found so far
                    # (It must have started within the time limit, checked when current_start was set)
                    if current_len > max_len:
                         max_len = current_len
                         best_start = current_start
                         best_end = i - 1 # Sequence ended at the previous beat

                    # Reset: Start a new potential sequence *only* if the current beat is '1' AND within time limit
                    current_start = -1 # Assume reset
                    current_start_time = -1.0
                    if current_count == 1 and current_timestamp <= max_start_time:
                        current_start = i
                        current_start_time = current_timestamp
                # else: Sequence continues, do nothing until it ends or file ends

            elif current_count == 1 and current_timestamp <= max_start_time:
                # --- Not currently in a sequence, check if this beat can start one ---
                # Needs to be count '1' and within time limit
                current_start = i
                current_start_time = current_timestamp

            # else: Not in a sequence, or current beat cannot start a valid one. Continue searching.


        # --- Check the last sequence after the loop finishes ---
        if current_start != -1:
            # Sequence potentially ran to the end of the file. Check if it's the best valid one.
            current_len = num_beats - current_start
            # (Start time was already checked when current_start was set)
            if current_len > max_len:
                max_len = current_len
                best_start = current_start
                best_end = num_beats - 1


        if best_start == -1:
            raise BeatCalculationError(
                f"No regular sequence starting with beat count '1' within the first {max_start_time:.1f} seconds was found "
                f"with the given tolerance interval of ±{tolerance_interval:.3f} seconds around the median interval of {median_interval:.3f} seconds."
            )

        # Calculate irregularity percentage *within* the found best sequence (based on intervals)
        # This is mostly for internal verification/debugging now, not stored in stats.
        best_irregularity = 0.0
        if best_start < best_end:  # Need at least two points for intervals
            best_sequence_timestamps = beat_data[best_start : best_end + 1, 0]
            best_sequence_intervals = np.diff(best_sequence_timestamps)
            if len(best_sequence_intervals) > 0:
                irregular_interval_count = np.sum(
                    np.abs(best_sequence_intervals - median_interval)
                    > tolerance_interval
                )
                best_irregularity = (
                    irregular_interval_count / len(best_sequence_intervals)
                ) * 100

        return best_start, best_end, best_irregularity

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # Added to avoid AttributeError during partial init
        """Return a safe, informative representation even if partially constructed."""
        if not hasattr(self, "beat_data"):
            # Partially constructed
            return f"<Beats (partially initialized) at {hex(id(self))}>"
        try:
            beats_len = self.beat_data.shape[0]
        except Exception:
            beats_len = "?"
        bpb = getattr(self, "beats_per_bar", "?")
        start_idx = getattr(self, "start_regular_beat_idx", "?")
        end_idx = getattr(self, "end_regular_beat_idx", "?")
        clip_len = getattr(self, "clip_length", "?")
        return (
            f"<Beats beats={beats_len}, bpb={bpb}, regular_section={start_idx}-{end_idx}, "
            f"clip_length={clip_len:.1f}s at {hex(id(self))}>"
        )


@dataclass
class RawBeats:
    """Stores the raw timestamp/count data detected from audio."""

    timestamps: np.ndarray
    beat_counts: np.ndarray
    clip_length: float  # Total length of the audio clip in seconds

    def __post_init__(self):
        # Basic validation moved from Beats.from_timestamps
        if not isinstance(self.timestamps, np.ndarray) or self.timestamps.ndim != 1:
            raise ValueError("Timestamps must be a 1D numpy array.")
        if not isinstance(self.beat_counts, np.ndarray) or self.beat_counts.ndim != 1:
            raise ValueError("Beat counts must be a 1D numpy array.")
        if len(self.timestamps) != len(self.beat_counts):
            raise ValueError(
                f"Timestamp count ({len(self.timestamps)}) does not match beat count ({len(self.beat_counts)}).")
        if not isinstance(self.clip_length, (int, float)) or self.clip_length <= 0:
            raise ValueError(f"clip_length must be a positive number, got {self.clip_length}")
        if len(self.timestamps) > 0 and self.timestamps[-1] > self.clip_length:
            raise ValueError(
                f"Last timestamp ({self.timestamps[-1]:.4f}) exceeds clip_length ({self.clip_length:.4f})")
        # Check for strictly increasing timestamps only if there's more than one
        if len(self.timestamps) > 1:
            intervals = np.diff(self.timestamps)
            if not np.all(intervals > 0):
                first_bad_idx = np.where(intervals <= 0)[0][0]
                raise ValueError(
                    f"Timestamps must be strictly increasing. Error found after index {first_bad_idx} "
                    f"(timestamps: {self.timestamps[first_bad_idx]:.4f} -> {self.timestamps[first_bad_idx+1]:.4f})")

    def save_to_file(self, path: Path | str) -> None:
        """Saves the raw beat data to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data_to_save = {
            "timestamps": self.timestamps.tolist(),
            "beat_counts": self.beat_counts.astype(int).tolist(),
            "clip_length": float(self.clip_length),
        }
        with path.open("w") as f:
            json.dump(data_to_save, f, indent=4)

    @classmethod
    def load_from_file(cls: Type[T_RawBeats], path: Path | str) -> T_RawBeats:
        """Loads raw beat data from a JSON file."""
        load_path = Path(path)
        if not load_path.is_file():
            raise FileNotFoundError(f"Beat file not found: {load_path}")

        with load_path.open("r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON from {load_path}: {e}") from e

        # Check for required keys
        required_keys = {
            "timestamps",
            "beat_counts",
            "clip_length",
        }
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            raise ValueError(
                f"File {load_path} is missing required keys: {sorted(list(missing_keys))}.")

        timestamps_list = data["timestamps"]
        beat_counts_list = data["beat_counts"]
        clip_length = float(data["clip_length"])

        if len(timestamps_list) != len(beat_counts_list):
            raise ValueError(
                f"Mismatched lengths in {load_path}: "
                f"{len(timestamps_list)} timestamps vs {len(beat_counts_list)} counts.")

        timestamps = np.array(timestamps_list, dtype=float)
        beat_counts = np.array(beat_counts_list, dtype=int)

        # Rely on __post_init__ for final validation
        return cls(
            timestamps=timestamps,
            beat_counts=beat_counts,
            clip_length=clip_length,
        )

