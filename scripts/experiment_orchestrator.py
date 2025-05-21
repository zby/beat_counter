#!/usr/bin/env python3
"""
Experiment Orchestrator Script

This script automates running experiments for beat and video generation.
It processes audio files from a directory structured by genre, applies specified settings,
and saves results with reproducibility information into a structured output directory.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

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


def run_experiment(config_file: Path, input_dir: Optional[Path] = None, 
                  output_base_dir: Optional[Path] = None, force_overwrite: bool = False,
                  experiment_name: Optional[str] = None, use_genre_defaults: Optional[bool] = None) -> None:
    """
    Run an experiment according to the provided configuration.
    
    Parameters
    ----------
    config_file : Path
        Path to the experiment configuration YAML file.
    input_dir : Optional[Path]
        Path to the root directory containing genre subdirectories. Overrides config value.
    output_base_dir : Optional[Path]
        Path to the base directory where experiment results will be stored. Overrides config value.
    force_overwrite : bool
        If True, overwrite existing experiment directory. Overrides config value.
    experiment_name : Optional[str]
        New name for the experiment. Overrides config value.
    use_genre_defaults : Optional[bool]
        Whether to use genre-based parameter inference. Overrides config value.
        
    Raises
    ------
    ValueError
        If the configuration file is invalid.
    FileNotFoundError
        If input directory does not exist.
    """
    # 1. Load configuration
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to load configuration file: {e}")
    
    # Extract experiment name with override precedence
    if experiment_name:
        # Override experiment name if provided
        config["experiment_name"] = experiment_name
    
    experiment_name = config.get("experiment_name")
    if not experiment_name:
        raise ValueError("Configuration must include 'experiment_name'")
    
    # 2. Determine directories with override precedence
    # Script defaults
    default_input_dir = Path("./data/by_genre")
    default_output_base_dir = Path("./data/experiments")
    default_force_overwrite = False
    default_use_genre_defaults = True  # Default is to use genre-based defaults if available
    
    # Apply config values if present
    config_input_dir = Path(config.get("input_dir", default_input_dir))
    config_output_base_dir = Path(config.get("output_base_dir", default_output_base_dir))
    config_force_overwrite = config.get("force_overwrite", default_force_overwrite)
    config_use_genre_defaults = config.get("use_genre_defaults", default_use_genre_defaults)
    
    # Apply CLI overrides if provided
    final_input_dir = input_dir if input_dir is not None else config_input_dir
    final_output_base_dir = output_base_dir if output_base_dir is not None else config_output_base_dir
    final_force_overwrite = force_overwrite or config_force_overwrite
    final_use_genre_defaults = use_genre_defaults if use_genre_defaults is not None else config_use_genre_defaults
    
    # Update config with final values for reproducibility
    config["input_dir"] = str(final_input_dir)
    config["output_base_dir"] = str(final_output_base_dir)
    config["force_overwrite"] = final_force_overwrite
    config["use_genre_defaults"] = final_use_genre_defaults
    
    # Validate required settings
    if not config.get("beat_settings"):
        raise ValueError("Configuration must include 'beat_settings'")
    if not config.get("video_settings"):
        raise ValueError("Configuration must include 'video_settings'")
    
    # Ensure input directory exists
    if not final_input_dir.exists() or not final_input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {final_input_dir}")
    
    # 3. Create output directory structure
    original_root_input_audio_dir = final_input_dir
    root_output_dir = final_output_base_dir / experiment_name
    experiment_data_dir = root_output_dir / original_root_input_audio_dir.name
    
    # Handle existing output directory
    if root_output_dir.exists():
        if final_force_overwrite:
            logging.warning(f"Overwriting existing experiment directory: {root_output_dir}")
            shutil.rmtree(root_output_dir)
        else:
            raise ValueError(f"Experiment directory already exists: {root_output_dir}. Use --force-overwrite to overwrite.")
    
    # Create output directory
    root_output_dir.mkdir(parents=True, exist_ok=False)
    
    # 4. Save reproducibility information
    git_info = get_git_info()
    save_reproducibility_info(
        output_dir=root_output_dir,
        git_info=git_info,
        config_file=config_file,
        config=config
    )
    
    # 5. Copy input data to experiment directory
    logging.info(f"Copying input data from {original_root_input_audio_dir} to {experiment_data_dir}")
    shutil.copytree(original_root_input_audio_dir, experiment_data_dir)
    
    # 6. Step A: Beat Generation
    logging.info("Starting beat detection...")
    beat_settings = config["beat_settings"]
    algorithm = beat_settings.get("algorithm", "madmom")
    detector_kwargs = beat_settings.get("detector_kwargs", {})
    beats_args = beat_settings.get("beats_args", {})
    
    # Handle the case where the config has beats_args: null explicitly set
    if beats_args is None:
        logging.warning("Configuration has beats_args: null. Using empty dict instead.")
        beats_args = {}
    
    # Handle the case where the config has detector_kwargs: null explicitly set
    if detector_kwargs is None:
        logging.warning("Configuration has detector_kwargs: null. Using empty dict instead.")
        detector_kwargs = {}
    
    # Log whether genre defaults will be used
    if final_use_genre_defaults:
        logging.info("Genre-based defaults enabled. Will create a GenreDB instance.")
        try:
            genre_db_instance = GenreDB()  # Use default database location
            logging.info("Successfully created GenreDB instance")
        except Exception as e:
            logging.error(f"Failed to initialize GenreDB: {e}")
            raise ValueError(f"Could not load genre database. Error: {e}")
    else:
        logging.info("Genre-based defaults disabled. No GenreDB will be used.")
        genre_db_instance = None
    
    # Use the process_batch function with the GenreDB instance if needed
    try:
        beat_results = process_batch(
            directory_path=experiment_data_dir,
            algorithm=algorithm,
            detector_kwargs=detector_kwargs,
            beats_args=beats_args,
            no_progress=False,
            genre_db=genre_db_instance
        )
    except Exception as e:
        print(f"DEBUG: Error in process_batch: {e}")
        print(f"DEBUG: detector_kwargs = {detector_kwargs!r}")
        print(f"DEBUG: beats_args = {beats_args!r}")
        raise
    
    logging.info(f"Beat detection completed for {len(beat_results)} files")
    
    # 7. Step B: Video Generation
    logging.info("Starting video generation...")
    video_settings = config["video_settings"]
    
    # Convert resolution from list to tuple if provided as list
    resolution = video_settings.get("resolution", [1280, 720])
    if isinstance(resolution, list) and len(resolution) == 2:
        resolution = (resolution[0], resolution[1])
    
    video_results = generate_batch_videos(
        input_dir=experiment_data_dir,
        output_dir=None,  # Save in the same directory structure
        resolution=resolution,
        fps=video_settings.get("fps", 60),
        sample_beats=video_settings.get("sample_beats"),
        tolerance_percent=video_settings.get("tolerance_percent", 10.0),
        min_measures=video_settings.get("min_measures", 5)
    )
    
    logging.info(f"Video generation completed")
    successful_videos = sum(result[1] for result in video_results)
    logging.info(f"Generated {successful_videos} videos successfully out of {len(video_results)} attempted")
    
    logging.info(f"Experiment '{experiment_name}' completed successfully")
    logging.info(f"Results saved to: {root_output_dir}")


def main():
    """Main entry point for the script."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run beat detection and video generation experiments")
    parser.add_argument("config_file", help="Path to experiment configuration YAML file")
    parser.add_argument("--input-dir", help="Path to root directory containing genre subdirectories")
    parser.add_argument("--output-base-dir", help="Path to base directory for experiment results")
    parser.add_argument("--force-overwrite", action="store_true", 
                        help="Overwrite existing experiment directory if it exists")
    parser.add_argument("--experiment-name", help="Override the experiment name from the config file")
    parser.add_argument("--use-genre-defaults", action="store_true", dest="use_genre_defaults",
                        help="Enable genre-based parameter inference (default behavior)")
    parser.add_argument("--no-genre-defaults", action="store_false", dest="use_genre_defaults",
                        help="Disable genre-based parameter inference")
    parser.set_defaults(use_genre_defaults=None)  # None means "use the config value"
    
    args = parser.parse_args()
    
    # Convert string paths to Path objects if provided
    input_dir = Path(args.input_dir) if args.input_dir else None
    output_base_dir = Path(args.output_base_dir) if args.output_base_dir else None
    config_file = Path(args.config_file)
    
    try:
        run_experiment(
            config_file=config_file,
            input_dir=input_dir,
            output_base_dir=output_base_dir,
            force_overwrite=args.force_overwrite,
            experiment_name=args.experiment_name,
            use_genre_defaults=args.use_genre_defaults
        )
    except Exception as e:
        logging.error(f"Experiment failed: {e}")
        # Optional: Enable for debugging
        # logging.exception("Traceback:")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main() 