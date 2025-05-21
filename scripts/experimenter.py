#!/usr/bin/env python3
"""
Experiment Orchestrator Script

This script automates running experiments for beat and video generation.
It operates on an experiment directory, which should contain:
- experiment_config.yaml: Configuration for the experiment.
- dance_music_genres.csv: Genre database (if genre defaults are used).
- by_genre/: A subdirectory with audio files structured by genre.

Processing (beat detection, video generation) is done in-place within the 'by_genre' subdirectory.
Reproducibility information is saved in the experiment directory.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json # Added for loading reproducibility info

import yaml

from beat_counter.core import process_batch, extract_beats
from beat_counter.utils.file_utils import find_audio_files
from beat_counter.core.video import generate_batch_videos
from beat_counter.core.beats import Beats
from beat_counter.genre_db import GenreDB
from beat_counter.utils.reproducibility import get_git_info, save_reproducibility_info
from tqdm import tqdm


# This function has been refactored into the enhanced process_batch function
# in beat_counter/core/pipeline.py with the use_genre_defaults parameter


def run_experiment(experiment_dir: Path, cli_use_genre_defaults: Optional[bool] = None) -> None:
    """
    Run an experiment based on the contents of the specified experiment directory.
    
    Parameters
    ----------
    experiment_dir : Path
        Path to the experiment directory. This directory should contain:
        - 'experiment_config.yaml': The configuration file.
        - 'dance_music_genres.csv': CSV for genre-specific defaults (if enabled).
        - 'by_genre/': Subdirectory with audio files.
    cli_use_genre_defaults : Optional[bool]
        Whether to use genre-based parameter inference, overriding config.
        None means use the config file's setting.
        
    Raises
    ------
    ValueError
        If the configuration is invalid or essential files/directories are missing.
    FileNotFoundError
        If required files (config, genre CSV, audio directory) are not found.
    """
    experiment_dir = experiment_dir.resolve()
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    experiment_name = experiment_dir.name
    logging.info(f"Starting experiment: {experiment_name}")

    # 1. Define paths
    config_yaml_path = experiment_dir / "experiment_config.yaml"
    audio_input_output_dir = experiment_dir / "by_genre"
    genre_csv_path = experiment_dir / "dance_music_genres.csv"
    repro_info_path = experiment_dir / "reproducibility_info.json"

    # 2. Validate essential paths
    if not config_yaml_path.is_file():
        raise FileNotFoundError(f"Experiment configuration file not found: {config_yaml_path}")
    if not audio_input_output_dir.is_dir():
        raise FileNotFoundError(f"Audio input directory 'by_genre' not found in {experiment_dir}")

    # 3. Load configuration
    try:
        with open(config_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None: # Handle empty YAML file
            config = {}
    except Exception as e:
        raise ValueError(f"Failed to load configuration file {config_yaml_path}: {e}")

    # 4. Determine final `use_genre_defaults`
    # Default is True if not specified in config or CLI
    config_use_genre_defaults = config.get("use_genre_defaults", True)
    if cli_use_genre_defaults is not None:
        final_use_genre_defaults = cli_use_genre_defaults
    else:
        final_use_genre_defaults = config_use_genre_defaults
    
    config["use_genre_defaults"] = final_use_genre_defaults # Update config for reproducibility log

    # 5. Handle GenreDB if defaults are used
    if final_use_genre_defaults:
        logging.info(f"Genre-based defaults enabled. Using genre CSV: {genre_csv_path}")
        if not genre_csv_path.is_file():
            raise FileNotFoundError(
                f"Genre database file '{genre_csv_path.name}' not found in {experiment_dir}, "
                "but 'use_genre_defaults' is enabled."
            )
        try:
            genre_db_instance = GenreDB(csv_path=genre_csv_path)
            logging.info(f"Successfully loaded GenreDB instance from {genre_csv_path}")
        except Exception as e:
            logging.error(f"Failed to initialize GenreDB from {genre_csv_path}: {e}")
            raise ValueError(f"Could not load genre database from {genre_csv_path}. Error: {e}")
    else:
        logging.info("Genre-based defaults disabled. Using explicit configuration values only.")
        genre_db_instance = None

    # 6. Check reproducibility information
    current_git_info = get_git_info()
    if repro_info_path.exists():
        logging.info(f"Found existing reproducibility information: {repro_info_path}")
        try:
            with open(repro_info_path, 'r') as f:
                saved_repro_data = json.load(f)
            saved_git_info = saved_repro_data.get("git_info")
            if saved_git_info and saved_git_info != current_git_info:
                logging.warning(
                    "Software version (Git info) has changed since this experiment was last run/setup. "
                    f"Current: {current_git_info.get('commit_hash', 'N/A')}, "
                    f"Saved: {saved_git_info.get('commit_hash', 'N/A')}"
                )
            # Could also compare saved config, etc., if desired
        except Exception as e:
            logging.warning(f"Could not read or parse existing reproducibility info: {e}")
    else:
        logging.warning(f"Reproducibility information not found in {experiment_dir}. Will be created.")

    # 7. Validate required settings in config
    if not config.get("beat_settings"):
        raise ValueError(f"Configuration file {config_yaml_path} must include 'beat_settings'")
    if not config.get("video_settings"):
        raise ValueError(f"Configuration file {config_yaml_path} must include 'video_settings'")

    # 8. Save reproducibility information (before processing)
    # This will save git information and experiment configuration
    save_reproducibility_info(
        output_dir=experiment_dir,  # Save directly in the experiment directory
        git_info=current_git_info,
        config_file=config_yaml_path,  # Path to the config file
        config=config  # Configuration dictionary
    )
    
    # 9. Step A: Beat Generation (in-place in audio_input_output_dir)
    logging.info(f"Starting beat detection in: {audio_input_output_dir}")
    beat_settings = config["beat_settings"]
    algorithm = beat_settings.get("algorithm", "madmom")
    detector_kwargs = beat_settings.get("detector_kwargs", {})
    beats_args = beat_settings.get("beats_args", {})
    
    if beats_args is None:
        logging.warning("Config 'beat_settings.beats_args' is null. Using empty dict instead.")
        beats_args = {}
    if detector_kwargs is None:
        logging.warning("Config 'beat_settings.detector_kwargs' is null. Using empty dict instead.")
        detector_kwargs = {}
    
    try:
        beat_results = process_batch(
            directory_path=audio_input_output_dir,
            algorithm=algorithm,
            detector_kwargs=detector_kwargs,
            beats_args=beats_args,
            no_progress=False, # Assuming interactive or TUI progress is fine
            genre_db=genre_db_instance # Pass the GenreDB instance itself
        )
    except Exception as e:
        logging.error(f"DEBUG: Error in process_batch: {e}")
        logging.error(f"DEBUG: detector_kwargs = {detector_kwargs!r}")
        logging.error(f"DEBUG: beats_args = {beats_args!r}")
        raise
    
    logging.info(f"Beat detection completed for {len(beat_results)} files in {audio_input_output_dir}")
    
    # 10. Step B: Video Generation (in-place in audio_input_output_dir)
    logging.info(f"Starting video generation in: {audio_input_output_dir}")
    video_settings = config["video_settings"]
    
    resolution = video_settings.get("resolution", [1280, 720])
    if isinstance(resolution, list) and len(resolution) == 2:
        resolution = (resolution[0], resolution[1])
    
    video_results = generate_batch_videos(
        input_dir=audio_input_output_dir,
        output_dir=None,  # None means save in same structure as input_dir (in-place)
        resolution=resolution,
        fps=video_settings.get("fps", 60),
        sample_beats=video_settings.get("sample_beats"),
        tolerance_percent=video_settings.get("tolerance_percent", 10.0),
        min_measures=video_settings.get("min_measures", 5)
    )
    
    logging.info("Video generation completed.")
    successful_videos = sum(result[1] for result in video_results)
    logging.info(f"Generated {successful_videos} videos successfully out of {len(video_results)} attempted in {audio_input_output_dir}")
    
    logging.info(f"Experiment '{experiment_name}' completed successfully.")
    logging.info(f"Results are stored in-place within: {audio_input_output_dir}")


def main():
    """Main entry point for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description="Run beat detection and video generation experiment from a specified directory."
    )
    parser.add_argument(
        "experiment_dir",
        help="Path to the experiment directory. It must contain 'experiment_config.yaml' "
             "and a 'by_genre/' subdirectory for audio files. "
             "If using genre defaults, 'dance_music_genres.csv' should also be present."
    )
    
    # Group for genre defaults flags
    genre_group = parser.add_mutually_exclusive_group()
    genre_group.add_argument(
        "--use-genre-defaults", 
        action="store_true", 
        dest="use_genre_defaults",
        default=None, # Important for logic: None means use config, True/False means override
        help="Enable genre-based parameter inference (overrides config if set)."
    )
    genre_group.add_argument(
        "--no-genre-defaults", 
        action="store_false", 
        dest="use_genre_defaults",
        help="Disable genre-based parameter inference (overrides config if set)."
    )
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    
    try:
        run_experiment(
            experiment_dir=experiment_dir,
            cli_use_genre_defaults=args.use_genre_defaults # Pass the CLI flag state
        )
    except Exception as e:
        logging.error(f"Experiment failed: {e}")
        # For more detailed debugging, uncomment the following line:
        # logging.exception("Full traceback:")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main() 