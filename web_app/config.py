"""Configuration module for Beat Detection Web App.

This module provides classes to load and access application configuration.
"""

import json
import os
import pathlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class User:
    username: str
    password: str
    is_admin: bool
    created_at: datetime

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        return cls(
            username=data['username'],
            password=data['password'],
            is_admin=data['is_admin'],
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        )

@dataclass
class StorageConfig:
    upload_dir: pathlib.Path
    max_upload_size_mb: int
    allowed_extensions: List[str]
    max_duration: int

@dataclass
class CeleryConfig:
    broker_url: str
    result_backend: str
    task_serializer: str
    accept_content: List[str]

@dataclass
class AppConfig:
    name: str
    version: str
    debug: bool
    allowed_hosts: List[str]
    max_queue_files: int

@dataclass
class Config:
    app: AppConfig
    storage: StorageConfig
    celery: CeleryConfig
    users: List[User]

    @classmethod
    def from_dir(cls, app_dir: pathlib.Path) -> 'Config':
        config_path = app_dir / "etc" / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path, "r") as f:
            data = json.load(f)
        
        # Load user configuration
        users_path = app_dir / "etc" / "users.json"
        if not users_path.exists():
            raise FileNotFoundError(f"Users configuration file not found at {users_path}")
        with open(users_path, "r") as f:
            users_data = json.load(f)
            
        
        return cls(
            app=AppConfig(**data['app']),
            storage=StorageConfig(**data['storage']),
            celery=CeleryConfig(**data['celery']),
            users=[User.from_dict(user) for user in users_data['users']]
        )
    
    @classmethod
    def from_env(cls) -> 'Config':
        app_dir = os.getenv('BEAT_COUNTER_APP_DIR')
        if app_dir:
            return cls.from_dir(pathlib.Path(app_dir).resolve())
        logger.warning("BEAT_COUNTER_APP_DIR is not set, using current directory")
        return cls.from_dir(pathlib.Path.cwd())



# Convenience functions for backward compatibility
def get_config() -> Dict[str, Any]:
    """Get application configuration as a dictionary."""
    config = Config.from_env()
    return config.__dict__

def get_users() -> List[Dict[str, Any]]:
    """Get user configuration as a list of dictionaries."""
    config = Config.from_env()
    return [user.__dict__ for user in config.users] 