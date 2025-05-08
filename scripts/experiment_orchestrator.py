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
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from beat_detection.core.factory import process_batch
from beat_detection.core.video import generate_batch_videos


def get_git_info() -> Dict[str, str]:
    """
    Get the current Git commit hash and diff.
    
    Returns
    -------
    Dict[str, str]
        Dictionary containing commit hash and diff.
        
    Raises
    ------
    RuntimeError
        If Git commands fail.
    """
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.STDOUT
        ).decode().strip()
        
        diff = subprocess.check_output(
            ["git", "diff", "HEAD"], 
            stderr=subprocess.STDOUT
        ).decode()
        
        return {
            "commit_hash": commit_hash,
            "diff": diff
        }
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to get Git information: {e.output.decode()}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)


def save_reproducibility_info(output_dir: Path, git_info: Dict[str, str], config_file: Path, config: Dict[str, Any]) -> None:
    """
    Save reproducibility information to the experiment directory.
    
    Parameters
    ----------
    output_dir : Path
        Directory to save reproducibility information.
    git_info : Dict[str, str]
        Dictionary containing Git commit hash and diff.
    config_file : Path
        Path to the experiment configuration file.
    config : Dict[str, Any]
        The configuration dictionary that will be saved (possibly modified from original).
    """
    # Save Git commit hash
    (output_dir / "git_commit.txt").write_text(git_info["commit_hash"])
    
    # Save Git diff
    (output_dir / "git_diff.patch").write_text(git_info["diff"])
    
    # Save the configuration
    with open(output_dir / "config_used.yaml", 'w') as f:
        yaml.dump(config, f)


def run_experiment(config_file: Path, input_dir: Optional[Path] = None, 
                  output_base_dir: Optional[Path] = None, force_overwrite: bool = False,
                  experiment_name: Optional[str] = None) -> None:
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
    
    # Apply config values if present
    config_input_dir = Path(config.get("input_dir", default_input_dir))
    config_output_base_dir = Path(config.get("output_base_dir", default_output_base_dir))
    config_force_overwrite = config.get("force_overwrite", default_force_overwrite)
    
    # Apply CLI overrides if provided
    final_input_dir = input_dir if input_dir is not None else config_input_dir
    final_output_base_dir = output_base_dir if output_base_dir is not None else config_output_base_dir
    final_force_overwrite = force_overwrite or config_force_overwrite
    
    # Update config with final values for reproducibility
    config["input_dir"] = str(final_input_dir)
    config["output_base_dir"] = str(final_output_base_dir)
    config["force_overwrite"] = final_force_overwrite
    
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
    save_reproducibility_info(root_output_dir, git_info, config_file, config)
    
    # 5. Copy input data to experiment directory
    logging.info(f"Copying input data from {original_root_input_audio_dir} to {experiment_data_dir}")
    shutil.copytree(original_root_input_audio_dir, experiment_data_dir)
    
    # 6. Step A: Beat Generation
    logging.info("Starting beat detection...")
    beat_settings = config["beat_settings"]
    algorithm = beat_settings.get("algorithm", "madmom")
    detector_kwargs = beat_settings.get("detector_kwargs", {})
    beats_args = beat_settings.get("beats_args", {})
    
    beat_results = process_batch(
        directory_path=experiment_data_dir,
        algorithm=algorithm,
        detector_kwargs=detector_kwargs,
        beats_args=beats_args
    )
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
            experiment_name=args.experiment_name
        )
    except Exception as e:
        logging.error(f"Experiment failed: {e}")
        # Optional: Enable for debugging
        # logging.exception("Traceback:")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main() 