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


class ConfigurationError(Exception):
    """Custom exception for configuration related errors."""
    pass


@dataclass
class User:
    username: str
    password: str
    is_admin: bool
    created_at: datetime

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        return cls(
            username=data["username"],
            password=data["password"],
            is_admin=data["is_admin"],
            created_at=datetime.fromisoformat(
                data["created_at"].replace("Z", "+00:00")
            ),
        )


@dataclass
class StorageConfig:
    upload_dir: pathlib.Path
    max_upload_size_mb: int
    allowed_extensions: List[str]
    max_audio_secs: int


@dataclass
class CeleryConfig:
    name: str
    broker_url: str
    result_backend: str
    task_serializer: str
    accept_content: List[str]
    task_ignore_result: bool = False
    result_extended: bool = True
    task_track_started: bool = True


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
    def _load_json_file(cls, file_path: pathlib.Path) -> Dict[str, Any]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required configuration file not found at {file_path}"
            )
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Error decoding JSON from {file_path}: {e}"
            ) from e

    @classmethod
    def from_dir(cls, app_root_dir: pathlib.Path) -> "Config":
        if not app_root_dir.is_dir():
            raise ConfigurationError(
                f"Application root directory not found or is not a directory: {app_root_dir}"
            )

        config_file_path = app_root_dir / "etc" / "config.json"
        users_file_path = app_root_dir / "etc" / "users.json"

        main_config_data = cls._load_json_file(config_file_path)
        users_data = cls._load_json_file(users_file_path)

        # Validate top-level keys
        required_main_keys = {"app", "storage", "celery"}
        if not required_main_keys.issubset(main_config_data.keys()):
            missing_keys = required_main_keys - main_config_data.keys()
            raise ConfigurationError(
                f"Missing required keys in {config_file_path}: {missing_keys}"
            )

        required_users_keys = {"users"}
        if not required_users_keys.issubset(users_data.keys()):
            missing_keys = required_users_keys - users_data.keys()
            raise ConfigurationError(
                f"Missing required keys in {users_file_path}: {missing_keys}"
            )

        try:
            storage_data = main_config_data["storage"]
            return cls(
                app=AppConfig(**main_config_data["app"]),
                storage=StorageConfig(
                    upload_dir=pathlib.Path(storage_data["upload_dir"]),
                    max_upload_size_mb=storage_data["max_upload_size_mb"],
                    allowed_extensions=storage_data["allowed_extensions"],
                    max_audio_secs=storage_data["max_audio_secs"],
                ),
                celery=CeleryConfig(**main_config_data["celery"]),
                users=[User.from_dict(user) for user in users_data["users"]],
            )
        except KeyError as e:
            raise ConfigurationError(
                f"Missing expected key in configuration files: {e}"
            ) from e
        except TypeError as e:  # For issues with **unpacking
            raise ConfigurationError(f"Configuration structure error: {e}") from e

    @classmethod
    def get_app_dir_from_env(cls) -> pathlib.Path:
        app_dir_str = os.getenv("BEAT_COUNTER_APP_DIR")
        if not app_dir_str:
            raise ConfigurationError(
                "BEAT_COUNTER_APP_DIR environment variable is not set. "
                "It should point to the root directory of the application "
                "(the parent of the 'etc' directory)."
            )

        resolved_path = pathlib.Path(app_dir_str).resolve()
        if not resolved_path.is_dir():
            raise ConfigurationError(
                f"The path specified by BEAT_COUNTER_APP_DIR is not a valid directory: {resolved_path}"
            )
        return resolved_path
