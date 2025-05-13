"""
Simple Beat-This detector wrapping the `beat_this` library.

This implementation mirrors the minimal example shipped with Beat-This! –
it just runs the model and hands the timestamps over to our RawBeats
container.  Any problems (missing package, bad input, GPU issues, …) are
allowed to raise naturally to keep failures obvious.
"""

from pathlib import Path
import os

import numpy as np
import torch
from beat_this.inference import File2Beats
from beat_this.model.postprocessor import Postprocessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
from pydub import AudioSegment

from beat_detection.core.beats import RawBeats, BeatCalculationError
from beat_detection.core.detector_protocol import BeatDetector  # Reuse the existing protocol

import beat_detection.utils.constants as constants

from typing import Optional, List, Tuple, Union

class CustomBeatTrackingProcessor(Postprocessor):
    """Custom Postprocessor that uses DBN with customizable parameters."""
    
    def __init__(self, 
                 fps: int,
                 beats_per_bar: Union[List[int], Tuple[int, ...]],
                 min_bpm: int,
                 max_bpm: int,
                 transition_lambda: float):
        """
        Initialize a custom DBN postprocessor with configurable parameters.
        
        Args:
            fps: The frames per second of the model predictions.
            beats_per_bar: List of possible values for beats per bar, e.g. [3, 4]
            min_bpm: Minimum BPM to consider
            max_bpm: Maximum BPM to consider
            transition_lambda: Lambda for the tempo change distribution
        """
        super().__init__(type="dbn", fps=fps)
        
        print(f"fps: {fps}")
        print(f"beats_per_bar: {beats_per_bar}")
        print(f"min_bpm: {min_bpm}")
        print(f"max_bpm: {max_bpm}")
        print(f"transition_lambda: {transition_lambda}")
        
        dbn_args = {
            "beats_per_bar": beats_per_bar,
            "fps": fps,
            "transition_lambda": transition_lambda,
        }
        if min_bpm is not None:
            dbn_args["min_bpm"] = min_bpm
        if max_bpm is not None:
            dbn_args["max_bpm"] = max_bpm
        
        self.dbn = DBNDownBeatTrackingProcessor(**dbn_args)


class BeatThisDetector:
    """Very thin wrapper around Beat-This! (`File2Beats`)."""

    def __init__(
        self,
        checkpoint: str = "final0",
        device: str | None = None,
        use_float16: bool = False,
        use_dbn: bool = True,
        beats_per_bar: Optional[Union[List[int], Tuple[int, ...]]] = constants.SUPPORTED_BEATS_PER_BAR,
        min_bpm: Optional[int] = None,
        max_bpm: Optional[int] = None,
        transition_lambda: float = 100,
        fps: int = constants.FPS,
    ) -> None:
        """Prepare the underlying Beat-This model once at construction time."""

        # Check if we need to force CPU usage via environment variable
        force_cpu = os.environ.get('BEAT_DETECTION_FORCE_CPU') == '1'
        
        # Select GPU if available unless the caller overrides the *device* string or force_cpu is set
        resolved_device = (
            torch.device("cpu") if force_cpu or device == "cpu" 
            else torch.device("cuda:0") if device is None and torch.cuda.is_available() 
            else torch.device(device or "cpu")
        )

        # Convenience wrapper provided by Beat-This!
        self._file2beats = File2Beats(
            checkpoint_path=checkpoint,
            device=resolved_device,
            float16=use_float16,
            dbn=use_dbn,
        )
        
        custom_postprocessor = CustomBeatTrackingProcessor(
            fps=fps,
            beats_per_bar=beats_per_bar,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            transition_lambda=transition_lambda
        )
        
        self._file2beats.frames2beats = custom_postprocessor

    def _get_audio_duration(self, audio_path: str | Path) -> float:
        """Get the duration of an audio file in seconds.

        Parameters:
        -----------
        audio_path : str | Path
            Path to the audio file.

        Returns:
        --------
        float
            Duration of the audio file in seconds.

        Raises:
        -------
        BeatCalculationError
            If the audio file cannot be loaded or processed.
        """
        try:
            # Load the audio file using pydub
            audio = AudioSegment.from_file(str(audio_path))
            duration = len(audio) / 1000.0  # Convert milliseconds to seconds
            if duration <= 0:
                raise BeatCalculationError(f"Invalid audio duration: {duration} seconds")
            return duration
        except Exception as e:
            raise BeatCalculationError(f"Failed to get audio duration: {e}") from e

    # ---------------------------------------------------------------------
    # Public API – part of the BeatDetector protocol
    # ---------------------------------------------------------------------
    def detect_beats(self, audio_path: str | Path) -> RawBeats:
        """Run Beat-This! on *audio_path* and return raw timestamp data."""

        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

        # Get audio duration for clip_length
        clip_length = self._get_audio_duration(audio_path)

        # The underlying processor returns (beats, downbeats)
        beats, downbeats = self._file2beats(str(audio_path))
        print(f"beats: {beats}")
        print(f"downbeats: {downbeats}")

        timestamps, counts = self._beats_to_counts(beats, downbeats)

        return RawBeats(
            timestamps=timestamps, 
            beat_counts=counts,
            clip_length=clip_length
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _beats_to_counts(
        beats_arr: np.ndarray, downbeats_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map *beats* to measure-relative counts (1 = downbeat).

        Replicates the logic used by the Beat-This! TSV writer.  Counting starts
        at 1 for every downbeat and increments until the next downbeat resets
        it.  Pick-up measures (beats before the first downbeat) are handled the
        same way.
        """

        if not np.all(np.isin(downbeats_arr, beats_arr)):
            raise ValueError("Not all downbeats are beats.")

        # Determine initial counter while accounting for pick-up beats.
        if len(downbeats_arr) >= 2:
            first_idx, second_idx = np.searchsorted(beats_arr, downbeats_arr[:2])
            beats_in_first_measure = second_idx - first_idx
            pickup_beats = first_idx
            counter = (
                beats_in_first_measure - pickup_beats
                if pickup_beats < beats_in_first_measure
                else 1
            )
        else:
            counter = 1

        counts: list[int] = []
        down_iter = iter(list(downbeats_arr) + [-1])  # Sentinel at end
        next_down = next(down_iter)

        for beat_time in beats_arr:
            if beat_time == next_down:
                counter = 1
                next_down = next(down_iter)
            else:
                counter += 1
            counts.append(counter)

        return beats_arr.copy(), np.asarray(counts, dtype=int) 