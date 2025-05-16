"""
Pipeline functionality for beat detection.

This module provides high-level functions for processing audio files for beat detection,
both individually and in batches.
"""
import logging
import os
import json
from typing import Dict, Type, Optional, Any, List, Tuple
from pathlib import Path
from tqdm import tqdm

from beat_detection.core.beats import Beats
from beat_detection.core.registry import get as get_detector
from beat_detection.core.detector_protocol import BeatDetector
from beat_detection.utils.file_utils import find_audio_files, get_output_path
from beat_detection.genre_db import GenreDB, parse_genre_from_path


def extract_beats(
    audio_file_path: str,
    output_path: Optional[str] = None,
    algorithm: str = "madmom",
    beats_args: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Beats:
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

    # Determine the output path using the utility function
    final_output_path = get_output_path(audio_file_path, output_path, extension=".beats")

    # Determine the stats output path
    stats_output_path = os.path.splitext(final_output_path)[0] + "._beat_stats"

    logging.info(f"Starting beat detection for {audio_file_path} using {algorithm}...")

    try:
        detector = get_detector(algorithm=algorithm, **kwargs)
        raw_beats = detector.detect_beats(audio_file_path)

        # Save raw beats to the output file
        raw_beats.save_to_file(final_output_path)

        # Create Beats object to validate raw_beats structure and infer parameters
        _beats_args = beats_args or {}
        beats_obj = Beats(raw_beats, **_beats_args)
        
        # Save beat statistics to ._beat_stats file
        beat_stats = beats_obj.to_dict()
        # Remove the beat_list key to avoid duplicating the raw beat data
        if "beat_list" in beat_stats:
            del beat_stats["beat_list"]
        
        # Write the stats to the ._beat_stats file
        with open(stats_output_path, 'w') as f:
            json.dump(beat_stats, f, indent=2)

        logging.info(
            f"Successfully processed {audio_file_path}. Effective beats_per_bar: {beats_obj.beats_per_bar}. "
            f"Beats saved to {final_output_path}. Statistics saved to {stats_output_path}."
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
    use_genre_defaults: bool = False,
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
    use_genre_defaults : bool, optional
        If True, try to detect genre from file path and apply genre-specific parameters,
        by default False.

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
    
    # Pre-initialize the GenreDB if using genre defaults
    genre_db = None
    if use_genre_defaults:
        genre_db = GenreDB()
        logging.info("Genre-based defaults enabled. Will check paths for genre information.")

    for audio_file in file_iterator:
        # Use relative path for reporting, but full path for processing
        relative_path_str = str(audio_file.relative_to(dir_path))
        full_path_str = str(audio_file)
        
        if pbar:
            pbar.set_description(f"Processing {audio_file.name}")
        else:
            # Log start only if not using progress bar and not quiet
            logging.info(f"Processing {relative_path_str}...")
        
        # For each file, start with base arguments
        file_beats_args = _beats_args.copy()
        file_detector_kwargs = _detector_kwargs.copy()
        
        # Apply genre-specific parameters if enabled
        if use_genre_defaults and genre_db is not None:
            try:
                # Try to get genre from path
                genre = parse_genre_from_path(full_path_str)
                logging.info(f"Detected genre from path for {audio_file.name}: {genre}")
                
                # Apply genre defaults to detector kwargs and beats args
                file_detector_kwargs = genre_db.detector_kwargs_for_genre(genre, existing=file_detector_kwargs)
                file_beats_args = genre_db.beats_kwargs_for_genre(genre, existing=file_beats_args)
                
                logging.info(f"Applied genre defaults for '{genre}' to {audio_file.name}")
                logging.debug(f"Genre-specific detector kwargs: {file_detector_kwargs}")
                logging.debug(f"Genre-specific beats args: {file_beats_args}")
            except ValueError:
                # No genre in path, use base arguments
                logging.debug(f"No genre detected in path for {audio_file.name}, using base arguments")

        try:
            # extract_beats handles saving the .beats file next to the audio file by default
            beats_obj = extract_beats(
                audio_file_path=full_path_str,
                output_path=None, # Let extract_beats handle default output
                algorithm=algorithm,
                beats_args=file_beats_args,
                **file_detector_kwargs,
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