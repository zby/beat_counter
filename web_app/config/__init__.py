"""Configuration module for Beat Detection Web App.

This module provides functions to load and access application configuration.
"""

import json
import os
import pathlib
from typing import Any, Dict, Optional

# Get the config directory path
CONFIG_DIR = pathlib.Path(__file__).parent
APP_DIR = CONFIG_DIR.parent

# Default paths for config files
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.json"
DEFAULT_USERS_PATH = CONFIG_DIR / "users.json"

# Configuration cache (loaded once)
_app_config = None
_users_config = None


def load_config(config_path: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Load application configuration from the config file.
    
    Args:
        config_path: Optional path to config file. Defaults to config/config.json.
        
    Returns:
        Dict containing application configuration
    """
    global _app_config
    
    # Return cached config if available
    if _app_config is not None:
        return _app_config
        
    # Use default path if not specified
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    # Check if config file exists
    if not config_path.exists():
        # Create default config if it doesn't exist
        default_config = {
            "app": {"name": "Beat Detection Web App", "version": "0.1.0", "debug": False},
            "files": {"upload_dir": "web_app/uploads", "max_upload_size_mb": 50},
            "queue": {"max_files": 50},
            "celery": {"broker_url": "redis://localhost:6379/0"}
        }
        
        # Ensure directory exists
        config_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Write default config
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        _app_config = default_config
        return default_config
    
    # Load config from file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        _app_config = config
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}


def load_users(users_path: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Load users configuration from the users file.
    
    Args:
        users_path: Optional path to users file. Defaults to config/users.json.
        
    Returns:
        Dict containing users configuration
    """
    global _users_config
    
    # Return cached config if available
    if _users_config is not None:
        return _users_config
    
    # Use default path if not specified
    if users_path is None:
        users_path = DEFAULT_USERS_PATH
    
    # Check if users file exists
    if not users_path.exists():
        # Create default users if they don't exist
        default_users = {
            "users": [
                {
                    "username": "admin",
                    "password": "admin123",
                    "is_admin": True,
                    "created_at": "2023-10-01T12:00:00Z"
                }
            ]
        }
        
        # Ensure directory exists
        users_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Write default users
        with open(users_path, 'w') as f:
            json.dump(default_users, f, indent=4)
        
        _users_config = default_users
        return default_users
    
    # Load users from file
    try:
        with open(users_path, 'r') as f:
            users = json.load(f)
        
        _users_config = users
        return users
    except Exception as e:
        print(f"Error loading users file: {e}")
        return {"users": []}


def save_users(users_data: Dict[str, Any], users_path: Optional[pathlib.Path] = None) -> bool:
    """Save users configuration to the users file.
    
    Args:
        users_data: Users data to save
        users_path: Optional path to users file. Defaults to config/users.json.
        
    Returns:
        True if successful, False otherwise
    """
    global _users_config
    
    # Use default path if not specified
    if users_path is None:
        users_path = DEFAULT_USERS_PATH
    
    try:
        # Ensure directory exists
        users_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Write users to file
        with open(users_path, 'w') as f:
            json.dump(users_data, f, indent=4)
        
        # Update cache
        _users_config = users_data
        return True
    except Exception as e:
        print(f"Error saving users file: {e}")
        return False


def get_config() -> Dict[str, Any]:
    """Get the application configuration.
    
    Returns:
        Dict containing application configuration
    """
    return load_config()


def get_users() -> Dict[str, Any]:
    """Get the users configuration.
    
    Returns:
        Dict containing users configuration
    """
    return load_users()


# Load configs at import time
load_config()
load_users() 