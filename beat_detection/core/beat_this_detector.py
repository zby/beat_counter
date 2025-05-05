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

from beat_detection.core.beats import RawBeats
from beat_detection.core.detector_protocol import BeatDetector  # Reuse the existing protocol


class BeatThisDetector:
    """Very thin wrapper around Beat-This! (`File2Beats`)."""

    def __init__(
        self,
        checkpoint: str = "final0",
        device: str | None = None,
        use_float16: bool = False,
        use_dbn: bool = False,
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

    # ---------------------------------------------------------------------
    # Public API – part of the BeatDetector protocol
    # ---------------------------------------------------------------------
    def detect_beats(self, audio_path: str | Path) -> RawBeats:
        """Run Beat-This! on *audio_path* and return raw timestamp data."""

        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

        # The underlying processor returns (beats, downbeats)
        beats, downbeats = self._file2beats(str(audio_path))

        timestamps, counts = self._beats_to_counts(beats, downbeats)

        return RawBeats(timestamps=timestamps, beat_counts=counts)

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