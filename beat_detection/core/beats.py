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
    is_downbeat: bool = False
    is_irregular_interval: bool = False # Irregular based on time interval from previous beat
    is_irregular_count: bool = False    # Irregular based on count exceeding meter
    beat_count: int = 0                 # 1-based count within the measure
    
    @property
    def is_irregular(self) -> bool:
        """Returns True if the beat is irregular for any reason."""
        return self.is_irregular_interval or self.is_irregular_count
        
    def to_dict(self) -> Dict:
        """Convert BeatInfo object to a dictionary."""
        return {
            "timestamp": self.timestamp,
            "index": self.index,
            "is_downbeat": self.is_downbeat,
            "is_irregular_interval": self.is_irregular_interval,
            "is_irregular_count": self.is_irregular_count,
            "is_irregular": self.is_irregular, # Include the property
            "beat_count": self.beat_count,
        }


@dataclass(frozen=True) # Make Beats immutable after creation
class Beats:
    """Container for all calculated beat-related information."""
    
    beat_list: List[BeatInfo]
    stats: BeatStatistics
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
        stats: BeatStatistics

        if num_beats < 2: 
            stats = BeatStatistics(0, 0, 0, 0, 0, 0, 0, num_beats)
            if num_beats == 1:
                 beat_list.append(BeatInfo(
                     timestamp=timestamps[0], index=0, is_downbeat=(0 in downbeat_indices),
                     is_irregular_interval=False, is_irregular_count=False,
                     beat_count=1 if (0 in downbeat_indices) else 0
                 ))
            return cls(beat_list=beat_list, stats=stats, meter=meter, 
                       tolerance_percent=tolerance_percent, tolerance_interval=0.0,
                       min_consistent_measures=min_consistent_measures,
                       start_regular_beat_idx=start_regular_beat_idx_calc, 
                       end_regular_beat_idx=end_regular_beat_idx_calc)

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

        stats = BeatStatistics(
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
            is_downbeat = i in downbeat_indices
            is_irregular_interval = interval_irregularities[i]
            
            if is_downbeat:
                last_downbeat_idx = i
                beat_count = 1
            elif last_downbeat_idx == -1:
                 beat_count = 0 # Undetermined count
            else:
                beat_count = i - last_downbeat_idx + 1

            is_irregular_count = (meter > 0 and beat_count > meter) or beat_count == 0
            
            display_beat_count = beat_count
            if meter > 0 and beat_count > meter:
                display_beat_count = (beat_count - 1) % meter + 1
            elif beat_count == 0:
                display_beat_count = 0

            beat_list.append(BeatInfo(
                timestamp=ts,
                index=i,
                is_downbeat=is_downbeat,
                is_irregular_interval=is_irregular_interval,
                is_irregular_count=is_irregular_count,
                beat_count=display_beat_count
            ))
            
        # TODO: Add intro/ending detection logic here to update 
        # start_regular_beat_idx_calc and end_regular_beat_idx_calc
        # Also add the check for >= 4 downbeats within this regular section.

        return cls(beat_list=beat_list, 
                   stats=stats, 
                   meter=meter, 
                   tolerance_percent=tolerance_percent, 
                   tolerance_interval=tolerance_interval_calculated,
                   min_consistent_measures=min_consistent_measures,
                   start_regular_beat_idx=start_regular_beat_idx_calc, # Use calculated value
                   end_regular_beat_idx=end_regular_beat_idx_calc)     # Use calculated value

    def to_dict(self) -> Dict:
        """Convert the Beats object to a dictionary suitable for JSON serialization."""
        return {
            "meter": self.meter,
            "tolerance_percent": self.tolerance_percent,
            "tolerance_interval": self.tolerance_interval,
            "min_consistent_measures": self.min_consistent_measures,
            "start_regular_beat_idx": self.start_regular_beat_idx,
            "end_regular_beat_idx": self.end_regular_beat_idx,
            "stats": self.stats.to_dict(),
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