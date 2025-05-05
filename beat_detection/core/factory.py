"""
Factory for creating beat detectors.
"""
import logging # Add logging import
from typing import Dict, Type, Optional, Any, List, Tuple # Add List, Tuple
import inspect  # Add inspect import
import warnings # Add warnings import
import os # Add os import
from pathlib import Path # Add Path import

from tqdm import tqdm # Add tqdm import

from beat_detection.core.detector_protocol import BeatDetector
from beat_detection.core.madmom_detector import MadmomBeatDetector
from beat_detection.core.beat_this_detector import BeatThisDetector
from beat_detection.core.beats import Beats
from beat_detection.utils.file_utils import find_audio_files # Add find_audio_files import

# Registry of available detectors
DETECTOR_REGISTRY: Dict[str, Type[BeatDetector]] = {
    "madmom": MadmomBeatDetector,
    "beat_this": BeatThisDetector,
}

def get_beat_detector(algorithm: str = "madmom", **kwargs: Any) -> BeatDetector:
    """
    Factory function to get a beat detector instance based on the algorithm name.
    
    Parameters
    ----------
    algorithm : str
        Name of the beat detection algorithm to use.
        Currently supported: "madmom" (default), "beat_this"
    **kwargs : Any
        Additional keyword arguments to pass to the detector constructor.
        Common parameters:
        
        MadmomBeatDetector specific parameters:
        - min_bpm: Minimum BPM to consider
        - max_bpm: Maximum BPM to consider
        - fps: Frames per second for processing
        
        BeatThisDetector specific parameters:
        - file2beats_processor: Custom processor to use
        
    Returns
    -------
    BeatDetector
        An instance of the requested beat detector.
        
    Raises
    ------
    ValueError
        If the requested algorithm is not supported.
    """
    if algorithm not in DETECTOR_REGISTRY:
        supported = ", ".join(f'"{name}"' for name in DETECTOR_REGISTRY.keys())
        raise ValueError(
            f'Unsupported beat detection algorithm: "{algorithm}". '
            f'Supported algorithms are: {supported}.'
        )
    
    detector_class = DETECTOR_REGISTRY[algorithm]
    
    # Get the signature of the detector's __init__ method
    init_signature = inspect.signature(detector_class.__init__)
    valid_params = {param.name for param in init_signature.parameters.values()}
    
    # Filter kwargs to only include valid parameters for the detector's __init__
    filtered_kwargs = {
        key: value for key, value in kwargs.items() if key in valid_params
    }
    
    # Check for extraneous arguments and issue a warning
    extraneous_kwargs = {key for key in kwargs if key not in valid_params}
    if extraneous_kwargs:
        warnings.warn(
            f"Ignoring extraneous keyword arguments for {detector_class.__name__}: "
            f"{', '.join(extraneous_kwargs)}",
            UserWarning
        )

    # Debugging: Print filtered kwargs
    # print(f"Initializing {detector_class.__name__} with: {filtered_kwargs}")
    
    return detector_class(**filtered_kwargs)


def extract_beats(
    audio_file_path: str,
    output_path: Optional[str] = None,
    algorithm: str = "madmom",
    beats_args: Optional[Dict[str, Any]] = {},
    **kwargs: Any
) -> Beats:  # Update return type hint
    """
    Detects beats in an audio file using the specified algorithm and saves them.

    This function gets the appropriate detector, detects beats, creates a Beats
    object, saves the beats to the specified output file, and returns the
    Beats object.

    Parameters
    ----------
    audio_file_path : str
        Path to the audio file to process.
    output_path : Optional[str], optional
        Path where the detected beat times will be saved (one time per line).
        If None, defaults to the audio file path with the extension replaced by '.beats'.
        Defaults to None.
    algorithm : str
        Name of the beat detection algorithm to use (passed to get_beat_detector).
        Defaults to "madmom".
    beats_args : Optional[Dict[str, Any]], optional
        Arguments to pass to the Beats constructor. Defaults to {}.
    **kwargs : Any
        Additional keyword arguments to pass to the detector constructor
        (passed to get_beat_detector).

    Returns
    -------
    Beats
        The Beats object containing the detected beat times.

    Raises
    ------
    ValueError
        If the requested algorithm is not supported (raised by get_beat_detector).
    FileNotFoundError
        If the audio_file_path does not exist.
    IOError
        If there's an error writing to the output_path.
    Exception
        Catches and logs any other exceptions during processing, then re-raises.
    """
    # Check if audio file exists
    audio_path = Path(audio_file_path)
    if not audio_path.is_file():
        logging.error("Audio file not found: %s", audio_path)
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Determine the output path if not provided
    # Create parent directory if an explicit output path is given
    final_output_path = output_path
    if final_output_path is None:
        base, _ = os.path.splitext(audio_file_path)
        final_output_path = base + ".beats"
    else:
        # Ensure the parent directory exists if an explicit path was provided
        Path(final_output_path).parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting beat detection for {audio_file_path} using {algorithm}...")

    try:
        detector = get_beat_detector(algorithm=algorithm, **kwargs)
        raw_beats = detector.detect_beats(audio_file_path)

        # Create Beats object to validate raw_beats structure and infer parameters
        beats_obj = Beats(
            raw_beats, # Positional argument
            **(beats_args or {}), # Ensure beats_args is a dict
            )

        # Save raw beats to the output file
        raw_beats.save_to_file(final_output_path)

        logging.info(
            f"Successfully processed {audio_file_path}. Effective beats_per_bar: {beats_obj.beats_per_bar}. Beats saved to {final_output_path}."
        )
        return beats_obj

    except Exception as exc:
        logging.exception("Beat detection failed for %s: %s", audio_file_path, exc)
        raise # Re-raise the exception after logging


def process_batch(
    directory_path: str | Path,
    algorithm: str = "madmom",
    beats_args: Optional[Dict[str, Any]] = None,
    detector_kwargs: Optional[Dict[str, Any]] = None,
    no_progress: bool = False,
) -> List[Tuple[str, Optional[Beats]]]:
    """
    Processes all audio files in a directory tree for beat detection.

    Recursively finds audio files, runs beat detection on each, saves the
    `.beats` file alongside the original audio, and returns a summary of results.

    Parameters
    ----------
    directory_path : str | Path
        The root directory to scan for audio files.
    algorithm : str, optional
        Beat detection algorithm to use, by default "madmom".
    beats_args : Optional[Dict[str, Any]], optional
        Arguments to pass to the Beats constructor, by default None.
    detector_kwargs : Optional[Dict[str, Any]], optional
        Arguments to pass to the beat detector constructor, by default None.
    no_progress : bool, optional
        If True, disable the progress bar, by default False.

    Returns
    -------
    List[Tuple[str, Optional[Beats]]]
        A list of tuples, where each tuple contains the relative path (as str)
        of the processed file and the resulting Beats object, or None if processing failed.

    Raises
    ------
    FileNotFoundError
        If the directory_path does not exist.
    """
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        logging.error("Directory not found: %s", dir_path)
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    audio_files = find_audio_files(dir_path)
    if not audio_files:
        logging.warning("No audio files found in %s", dir_path)
        return [] # Return empty list if no files found

    logging.info(f"Found {len(audio_files)} audio files to process in {dir_path}")

    pbar: Optional[tqdm] = None
    if not no_progress:
        pbar = tqdm(audio_files, desc="Processing files", unit="file", ncols=100)
        file_iterator = pbar
    else:
        file_iterator = audio_files

    results: List[Tuple[str, Optional[Beats]]] = []
    _beats_args = beats_args or {}
    _detector_kwargs = detector_kwargs or {}

    for audio_file in file_iterator:
        # Use relative path for reporting, but full path for processing
        relative_path_str = str(audio_file.relative_to(dir_path))
        if pbar:
            pbar.set_description(f"Processing {audio_file.name}")
        else:
            # Log start only if not using progress bar and not quiet
            logging.info(f"Processing {relative_path_str}...")

        try:
            # extract_beats handles saving the .beats file next to the audio file by default
            beats_obj = extract_beats(
                audio_file_path=str(audio_file),
                output_path=None, # Let extract_beats handle default output
                algorithm=algorithm,
                beats_args=_beats_args,
                **_detector_kwargs,
            )
            results.append((relative_path_str, beats_obj))
            if pbar is None: # Log success only if not using progress bar
                logging.info(f"Successfully processed {relative_path_str}")
        except Exception as e:
            results.append((relative_path_str, None))
            # Always log errors
            logging.error(f"Failed to process {relative_path_str}: {e}")

    if pbar:
        pbar.close()

    return results
