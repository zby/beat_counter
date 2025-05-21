"""
Reproducibility utilities for beat detection experiments.

This module provides functions for capturing and saving reproducibility information
for experiments, ensuring that experiments can be recreated.
"""

import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional


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


def save_reproducibility_info(
    output_dir: Path, 
    git_info: Dict[str, str], 
    config_file: Optional[Path] = None, 
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save reproducibility information to the experiment directory.
    
    Parameters
    ----------
    output_dir : Path
        Directory to save reproducibility information.
    git_info : Dict[str, str]
        Dictionary containing Git commit hash and diff.
    config_file : Optional[Path], optional
        Path to the experiment configuration file.
    config : Optional[Dict[str, Any]], optional
        The configuration dictionary that will be saved (possibly modified from original).
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Git commit hash
    (output_dir / "git_commit.txt").write_text(git_info["commit_hash"])
    
    # Save Git diff
    (output_dir / "git_diff.patch").write_text(git_info["diff"])
    
    # Save the configuration if provided
    if config is not None:
        try:
            import yaml
            with open(output_dir / "config_used.yaml", 'w') as f:
                yaml.dump(config, f)
        except ImportError:
            logging.warning("PyYAML not available, skipping config saving")