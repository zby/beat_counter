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

from beat_counter.core.beats import Beats
from beat_counter.core.registry import build
from beat_counter.core.detector_protocol import BeatDetector
from beat_counter.utils.file_utils import find_audio_files, get_output_path
from beat_counter.genre_db import GenreDB, parse_genre_from_path


def extract_beats(
    audio_file_path: str,
    output_path: Optional[str] = None,
    detector_name: str = "madmom",
    beats_args: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Optional[Beats]:
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
    detector_name : str
        Name of the beat detection algorithm to use (passed to registry.build).
        Defaults to "madmom".
    beats_args : Optional[Dict[str, Any]], optional
        Arguments to pass to the Beats constructor. Defaults to {}.
    **kwargs : Any
        Additional keyword arguments to pass to the detector constructor
        (passed to registry.build).

    Returns
    -------
    Optional[Beats]
        The Beats object containing the detected beat times, or None if beat validation failed
        but raw beats were successfully saved.

    Raises
    ------
    ValueError
        If the requested algorithm is not supported (raised by registry.build).
    FileNotFoundError
        If the audio_file_path does not exist.
    IOError
        If there's an error writing to the output_path.
    Exception
        Catches and logs any detector/extraction exceptions, then re-raises them.
        Note that Beats validation exceptions are caught internally and will
        result in a None return value rather than being raised.
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

    logging.info(f"Starting beat detection for {audio_file_path} using {detector_name}...")

    # Step 1: Extract raw beats - failures here are critical
    try:
        detector = build(detector_name, **kwargs)
        raw_beats = detector.detect_beats(audio_file_path)
    except Exception as exc:
        logging.exception("Beat detection failed for %s: %s", audio_file_path, exc)
        raise # Re-raise the exception after logging
    
    # Step 2: Save the raw beats immediately regardless of future steps
    raw_beats.save_to_file(final_output_path)
    logging.info(f"Saved raw beats to {final_output_path}")
    
    # Step 3: Try to create Beats object and save statistics - this may fail but won't affect saved raw beats
    _beats_args = beats_args or {}
    try:
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
        
    except Exception as beats_validation_exc:
        # Log validation error but don't re-raise - raw beats were saved successfully
        logging.error(f"Beat validation failed for {audio_file_path}: {beats_validation_exc}")
        logging.info(f"Raw beats were still saved to {final_output_path}")
        
        # Return None to indicate partial success
        return None


def process_batch(
    directory_path: str | Path,
    detector_name: str = "madmom",
    beats_args: Optional[Dict[str, Any]] = None,
    detector_kwargs: Optional[Dict[str, Any]] = None,
    no_progress: bool = False,
    genre_db: Optional[GenreDB] = None,
) -> List[Tuple[str, Optional[Beats]]]:
    """
    Processes all audio files in a directory tree for beat detection.

    Recursively finds audio files, runs beat detection on each, saves the
    `.beats` file alongside the original audio, and returns a summary of results.

    Parameters
    ----------
    directory_path : str | Path
        The root directory to scan for audio files.
    detector_name : str, optional
        Beat detection algorithm to use, by default "madmom".
    beats_args : Optional[Dict[str, Any]], optional
        Arguments to pass to the Beats constructor, by default None.
    detector_kwargs : Optional[Dict[str, Any]], optional
        Arguments to pass to the beat detector constructor, by default None.
    no_progress : bool, optional
        If True, disable the progress bar, by default False.
    genre_db : Optional[GenreDB], optional
        If provided, used for genre-specific parameter lookups, by default None.
        If None, no genre-specific parameters will be applied.

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
    
    # Log if genre DB is provided
    if genre_db is not None:
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
        
        # Apply genre-specific parameters if a GenreDB instance was provided
        if genre_db is not None:
            try:
                genre = parse_genre_from_path(full_path_str)
                # Apply genre defaults to detector and Beats kwargs
                file_detector_kwargs = genre_db.detector_kwargs_for_genre(
                    genre, existing=file_detector_kwargs
                )
                file_beats_args = genre_db.beats_kwargs_for_genre(
                    genre, existing=file_beats_args
                )
                logging.info(f"Applied genre '{genre}' defaults for {relative_path_str}")
            except ValueError:
                # No genre in path, just use defaults (no warning needed)
                pass
        
        try:
            # Process this audio file and get Beats object
            beats = extract_beats(
                audio_file_path=full_path_str,
                output_path=None, # Let extract_beats handle default output
                detector_name=detector_name,
                beats_args=file_beats_args,
                **file_detector_kwargs,
            )
            results.append((relative_path_str, beats))
        except Exception as e:
            # Log the error (extract_beats already logged the specifics)
            logging.error(f"Failed to process {relative_path_str}")
            results.append((relative_path_str, None))
    
    if pbar:
        pbar.close()
    
    # Log final summary
    success_count = sum(1 for _, beats in results if beats is not None)
    logging.info(
        f"Processed {len(results)} files: {success_count} succeeded, "
        f"{len(results) - success_count} failed."
    )
    
    return results 