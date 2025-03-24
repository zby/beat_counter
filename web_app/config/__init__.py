"""Configuration module for Beat Detection Web App.

This module provides functions to load and access application configuration.
"""

import json
import os
import pathlib
from typing import Any, Dict, Optional

def get_app_dir() -> pathlib.Path:
    """Get the application directory.
    
    First checks the BEAT_COUNTER_APP_DIR environment variable.
    If not set, assumes the app is being run from its root directory 
    and uses the parent directory of the web_app package.
    
    Returns
    -------
    Path to the application directory
    """
    # Try environment variable first
    app_dir = os.getenv('BEAT_COUNTER_APP_DIR')
    if app_dir:
        return pathlib.Path(app_dir).resolve()
    
    # Otherwise use the parent directory of web_app package
    return pathlib.Path(__file__).parent.parent.parent.resolve()

# Get application directory
APP_DIR = get_app_dir()

# Default paths for config files
CONFIG_DIR = APP_DIR / "web_app" / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.json"
DEFAULT_USERS_PATH = CONFIG_DIR / "users.json"

# Configuration cache (loaded once)
_app_config = None
_users_config = None

def load_config(config_path: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Load application configuration from JSON file.
    
    Parameters
    ----------
    config_path : Path, optional
        Path to the configuration file, by default DEFAULT_CONFIG_PATH
    
    Returns
    -------
    Dict[str, Any]
        Application configuration
    
    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist
    """
    global _app_config
    
    # If already loaded, return cached version
    if _app_config is not None:
        return _app_config
    
    # Default to DEFAULT_CONFIG_PATH if not specified
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Update paths to be absolute
    config['storage']['upload_dir'] = str(APP_DIR / config['storage']['upload_dir'])
    
    _app_config = config
    return config

def load_users(users_path: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Load user configuration from JSON file.
    
    Parameters
    ----------
    users_path : Path, optional
        Path to the users file, by default DEFAULT_USERS_PATH
    
    Returns
    -------
    Dict[str, Any]
        User configuration
    
    Raises
    ------
    FileNotFoundError
        If the users file does not exist
    """
    global _users_config
    
    # If already loaded, return cached version
    if _users_config is not None:
        return _users_config
    
    # Default to DEFAULT_USERS_PATH if not specified
    if users_path is None:
        users_path = DEFAULT_USERS_PATH
    
    # Load users
    with open(users_path, "r") as f:
        users = json.load(f)
    
    _users_config = users
    return users

def get_config() -> Dict[str, Any]:
    """Get application configuration.
    
    Returns
    -------
    Dict[str, Any]
        Application configuration
    """
    return load_config()

def get_users() -> Dict[str, Any]:
    """Get user configuration.
    
    Returns
    -------
    Dict[str, Any]
        User configuration
    """
    return load_users()

# Load configs at import time
load_config()
load_users() 