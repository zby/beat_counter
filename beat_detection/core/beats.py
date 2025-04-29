"""
Core beat data structures and utilities.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterable, Type, TypeVar
import json
from pathlib import Path
import dataclasses

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

    @classmethod
    def from_timestamps(
        cls,
        timestamps: np.ndarray,
        beats_per_bar: int,
        beat_counts: np.ndarray,  # Use pre-calculated counts
        tolerance_percent: float = 10.0,
        min_measures: int = 5,  # Minimum consistent measures required
        max_start_time: float = MAX_START_TIME,
    ) -> "Beats":
        """
        Factory method to create a Beats object from raw timestamp data and counts.
        Calculates statistics and identifies the longest regular section.

        Raises:
        -------
        BeatCalculationError
            If beat statistics or the regular section cannot be reliably calculated.
        """
        # --- Input Validation ---
        if not isinstance(timestamps, np.ndarray) or timestamps.ndim != 1:
            raise BeatCalculationError("Timestamps must be a 1D numpy array.")
        if not isinstance(beat_counts, np.ndarray) or beat_counts.ndim != 1:
            raise BeatCalculationError("beat_counts must be a 1D numpy array.")
        if len(timestamps) != len(beat_counts):
            raise BeatCalculationError(
                f"Timestamp count ({len(timestamps)}) does not match beat_counts count ({len(beat_counts)})."
            )

        num_beats = len(timestamps)

        # Check for strictly increasing timestamps
        intervals = np.diff(timestamps)
        if num_beats > 1 and not np.all(intervals > 0):
            first_bad_idx = np.where(intervals <= 0)[0][0]
            raise BeatCalculationError(
                f"Timestamps must be strictly increasing. Error found after index {first_bad_idx} "
                f"(timestamps: {timestamps[first_bad_idx]:.4f} -> {timestamps[first_bad_idx+1]:.4f})"
            )

        if not isinstance(tolerance_percent, (int, float)) or tolerance_percent < 0:
            raise BeatCalculationError(
                f"Invalid tolerance_percent provided: {tolerance_percent}. Must be a non-negative number."
            )

        if beats_per_bar <= 1:
            raise BeatCalculationError(
                f"Invalid beats_per_bar provided: {beats_per_bar}. Must be 2 or greater."
            )

        # Check minimum required beats AFTER potentially processing beat_counts
        # We need the median interval first to find the regular sequence.

        # This check implicitly handles num_beats <= 1 if min_measures >= 1
        # Moved this check lower, after median interval calculation, as the requirement
        # is primarily about having enough beats for a *stable* section analysis.
        # required_beats = beats_per_bar * min_measures
        # if num_beats < required_beats:
        #     raise BeatCalculationError(
        #         f"Insufficient number of total beats ({num_beats}) for analysis start with beats_per_bar {beats_per_bar}. "
        #         f"Requires at least {required_beats} beats ({min_measures} measures)."
        #     )
        # --- End Basic Validation ---

        # 1. Prepare beat_data array
        # Apply the logic: counts > beats_per_bar or <= 0 are considered irregular (represented as 0)
        processed_beat_counts = np.where(
            (beat_counts > 0) & (beat_counts <= beats_per_bar),
            beat_counts,
            0,  # Mark invalid/out-of-range counts as 0
        ).astype(
            int
        )  # Ensure integer type

        # Stack timestamps and processed counts into the main data array
        beat_data = np.stack((timestamps, processed_beat_counts), axis=1)

        # 2. Calculate intervals and overall median interval
        if num_beats <= 1:
            # Handle cases with 0 or 1 beat
            median_interval = 0.0
            tempo_bpm = 0.0
            tolerance_interval_calculated = 0.0
            overall_stats = BeatStatistics(
                mean_interval=0.0,
                median_interval=0.0,
                std_interval=0.0,
                min_interval=0.0,
                max_interval=0.0,
                irregularity_percent=(
                    100.0 if num_beats == 1 and processed_beat_counts[0] == 0 else 0.0
                ),  # 1 beat is regular if count is valid
                tempo_bpm=0.0,
                total_beats=num_beats,
            )
            # Cannot find a regular sequence if fewer than 2 beats
            start_regular_beat_idx_calc = 0
            end_regular_beat_idx_calc = num_beats  # Treat as all irregular/degenerate
            regular_stats = BeatStatistics(  # Empty stats
                mean_interval=0.0,
                median_interval=0.0,
                std_interval=0.0,
                min_interval=0.0,
                max_interval=0.0,
                irregularity_percent=0.0,
                tempo_bpm=0.0,
                total_beats=0,
            )
            if num_beats >= 1:
                # Need to check the min_measures requirement even for few beats
                required_beats = beats_per_bar * min_measures
                if num_beats < required_beats:
                    raise BeatCalculationError(
                        f"Insufficient number of beats ({num_beats}) for analysis with beats_per_bar {beats_per_bar}. "
                        f"Requires at least {required_beats} beats ({min_measures} measures)."
                    )
                # If we pass the check but still have < 2 beats, it implies min_measures allows it,
                # but we still can't calculate regular stats properly. The empty stats above handle this.

            # Construct and return early for 0 or 1 beat cases
            return cls(
                beat_data=beat_data,
                overall_stats=overall_stats,
                regular_stats=regular_stats,
                beats_per_bar=beats_per_bar,
                tolerance_percent=tolerance_percent,
                tolerance_interval=tolerance_interval_calculated,
                min_measures=min_measures,
                start_regular_beat_idx=start_regular_beat_idx_calc,
                end_regular_beat_idx=end_regular_beat_idx_calc,
            )

        # --- Calculations for 2+ beats ---
        intervals = np.diff(beat_data[:, 0])  # Use timestamp column
        median_interval = np.median(intervals)
        if median_interval <= 0:
            raise BeatCalculationError(
                f"Cannot calculate reliable beat statistics: Median interval is {median_interval:.4f}. "
                f"Check input timestamps: {beat_data[:5, 0]}..."
            )

        tempo_bpm = 60 / median_interval
        tolerance_interval_calculated = median_interval * (tolerance_percent / 100.0)

        # 3. Calculate overall statistics (irregularity based on count==0)
        irregular_count = np.sum(beat_data[:, 1] == 0)  # Count where beat_count is 0
        overall_irregularity_percent = (
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
        try:
            # Pass the beat_data array directly
            # Specify max_start_time=30.0
            start_idx, end_idx, _ = cls._find_longest_regular_sequence_static(
                beat_data,
                tolerance_percent,
                beats_per_bar,
                median_interval,
                tolerance_interval_calculated,
                max_start_time=MAX_START_TIME,
            )
            start_regular_beat_idx_calc = start_idx
            end_regular_beat_idx_calc = (
                end_idx + 1
            )  # Convert inclusive end index to exclusive

            # Check if the found sequence meets the minimum length requirement
            sequence_length = end_regular_beat_idx_calc - start_regular_beat_idx_calc
            required_beats = beats_per_bar * min_measures
            if sequence_length < required_beats:
                raise BeatCalculationError(
                    f"Longest regular sequence found ({sequence_length} beats from index {start_idx} to {end_idx}) is shorter than required "
                    f"({required_beats} beats = {min_measures} measures of {beats_per_bar}/X time)."
                )
        except BeatCalculationError as e:
            # Re-raise with more context if finding the sequence failed
            raise BeatCalculationError(
                f"Could not determine a stable regular section: {e}"
            ) from e

        # 5. Calculate statistics for the identified regular section
        if start_regular_beat_idx_calc >= end_regular_beat_idx_calc:
            raise BeatCalculationError(
                f"Invalid regular section bounds calculated: start={start_regular_beat_idx_calc}, end={end_regular_beat_idx_calc}. "
                f"Cannot calculate regular statistics."
            )

        # Extract timestamps for the regular section
        regular_timestamps = beat_data[
            start_regular_beat_idx_calc:end_regular_beat_idx_calc, 0
        ]
        num_regular_beats = len(regular_timestamps)

        if num_regular_beats <= 1:
            # Handle case where regular section has 0 or 1 beat (shouldn't happen if required_beats > 1)
            regular_mean_interval = 0.0
            regular_median_interval = 0.0
            regular_std_interval = 0.0
            regular_min_interval = 0.0
            regular_max_interval = 0.0
            regular_tempo_bpm = 0.0
            regular_irregularity_percent = (
                0.0  # No intervals or counts to check within the section
            )
        else:
            regular_intervals = np.diff(regular_timestamps)
            if (
                len(regular_intervals) == 0
            ):  # Should not happen if num_regular_beats > 1
                raise BeatCalculationError(
                    f"Internal error: Regular section has {num_regular_beats} beats, "
                    f"but no intervals were extracted for statistics. Indices: {start_regular_beat_idx_calc} to {end_regular_beat_idx_calc-1}"
                )

            regular_median_interval = np.median(regular_intervals)
            if regular_median_interval <= 0:
                raise BeatCalculationError(
                    f"Cannot calculate reliable regular beat statistics: Median interval is {regular_median_interval:.4f}. "
                    f"Regular section indices: {start_regular_beat_idx_calc}-{end_regular_beat_idx_calc}. "
                    f"Intervals: {regular_intervals[:5]}..."
                )
            regular_tempo_bpm = 60 / regular_median_interval
            regular_mean_interval = np.mean(regular_intervals)
            regular_std_interval = np.std(regular_intervals)
            regular_min_interval = np.min(regular_intervals)
            regular_max_interval = np.max(regular_intervals)

            # Calculate irregularity within the regular section (based on count==0)
            regular_section_counts = beat_data[
                start_regular_beat_idx_calc:end_regular_beat_idx_calc, 1
            ]
            regular_section_irregular_count = np.sum(regular_section_counts == 0)
            regular_irregularity_percent = (
                regular_section_irregular_count / num_regular_beats
            ) * 100

        regular_stats = BeatStatistics(
            mean_interval=regular_mean_interval,
            median_interval=regular_median_interval,
            std_interval=regular_std_interval,
            min_interval=regular_min_interval,
            max_interval=regular_max_interval,
            irregularity_percent=regular_irregularity_percent,  # Based on count==0 within the section
            tempo_bpm=regular_tempo_bpm,
            total_beats=num_regular_beats,
        )

        # 6. Construct the final Beats object
        return cls(
            beat_data=beat_data,
            overall_stats=overall_stats,
            regular_stats=regular_stats,
            beats_per_bar=beats_per_bar,
            tolerance_percent=tolerance_percent,
            tolerance_interval=tolerance_interval_calculated,
            min_measures=min_measures,
            start_regular_beat_idx=start_regular_beat_idx_calc,
            end_regular_beat_idx=end_regular_beat_idx_calc,
        )

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
        }

    def get_info_at_time(self, t: float) -> Tuple[int, float, int]:
        """
        Get count, time since last beat, and beat index at time t.

        Returns a tuple of (count, time_since_beat, beat_idx) where:
        - count: The beat count (0 if before first beat, irregular, or outside regular section)
        - time_since_beat: Time in seconds since the last beat (0.0 if before first beat)
        - beat_idx: Index of the beat in the beat_data array (-1 if before first beat)
        """
        if self.beat_data.shape[0] == 0 or t < self.beat_data[0, 0]:
            return 0, 0.0, -1  # Before first beat or no beats

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
        tolerance_percent: float,
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
        tolerance_percent : float
            The tolerance percentage (used for error message).
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
                f"with the given tolerance ({tolerance_percent}%)."
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


T = TypeVar("T", bound="RawBeats")

@dataclasses.dataclass(frozen=True)
class RawBeats:
    """Stores the raw timestamp/count data and the beats_per_bar used for detection."""

    timestamps: np.ndarray
    beat_counts: np.ndarray
    beats_per_bar: int
    # tolerance_percent: float # Removed
    # min_measures: int      # Removed

    def __post_init__(self):
        if self.timestamps.shape != self.beat_counts.shape:
            raise ValueError(
                "Timestamps and beat_counts must have the same shape. "
                f"Got {self.timestamps.shape} and {self.beat_counts.shape}"
            )
        if self.timestamps.ndim != 1:
            raise ValueError(
                "Timestamps must be a 1D array. "
                f"Got shape {self.timestamps.shape}"
            )
        if not isinstance(self.beats_per_bar, int) or self.beats_per_bar <= 0:
            raise ValueError(f"beats_per_bar must be a positive integer, got {self.beats_per_bar}")
        # Removed validation for tolerance_percent and min_measures

    def save_to_file(self, path: Path | str) -> None:
        """Saves the raw beat data and beats_per_bar to a JSON file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data_to_save = {
            "beats_per_bar": self.beats_per_bar,
            # Removed tolerance_percent, min_measures
            "timestamps": self.timestamps.tolist(),
            "beat_counts": self.beat_counts.tolist(),
        }
        with save_path.open("w") as f:
            json.dump(data_to_save, f, indent=2)

    @classmethod
    def load_from_file(cls: Type[T], path: Path | str) -> T:
        """Loads raw beat data and beats_per_bar from a JSON file."""
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
            "beats_per_bar",
            # Removed tolerance_percent, min_measures
            "timestamps",
            "beat_counts",
        }
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            raise ValueError(
                f"File {load_path} is missing required keys: {sorted(list(missing_keys))}."
            )

        timestamps_list = data["timestamps"]
        beat_counts_list = data["beat_counts"]

        if len(timestamps_list) != len(beat_counts_list):
            raise ValueError(
                f"Mismatched lengths in {load_path}: "
                f"{len(timestamps_list)} timestamps vs {len(beat_counts_list)} counts."
            )

        timestamps = np.array(timestamps_list, dtype=float)
        beat_counts = np.array(beat_counts_list, dtype=int)

        try:
             beats_per_bar = int(data["beats_per_bar"])
             # Removed tolerance_percent, min_measures
        except (TypeError, ValueError) as e:
             raise ValueError(f"Invalid type for beats_per_bar in {load_path}: {e}") from e

        # Rely on __post_init__ for final validation
        return cls(
            timestamps=timestamps,
            beat_counts=beat_counts,
            beats_per_bar=beats_per_bar,
            # Removed tolerance_percent, min_measures
        )

