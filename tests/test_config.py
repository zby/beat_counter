import os
import shutil
import tempfile
import json
from pathlib import Path
import pytest
from unittest.mock import patch

# Import the classes and exception to be tested
from web_app.config import Config, ConfigurationError, AppConfig, StorageConfig, CeleryConfig, User

@pytest.fixture
def temp_config_files_factory():
    """Factory to create temporary config files in a nested 'etc' directory."""
    created_dirs = []

    def _creator(config_data: dict, users_data: dict, create_config_json=True, create_users_json=True) -> Path:
        temp_root_dir = tempfile.TemporaryDirectory()
        created_dirs.append(temp_root_dir) # Keep track for cleanup
        app_root_path = Path(temp_root_dir.name)
        
        etc_dir = app_root_path / "etc"
        etc_dir.mkdir()

        if create_config_json:
            with open(etc_dir / "config.json", "w") as f:
                json.dump(config_data, f)
        
        if create_users_json:
            with open(etc_dir / "users.json", "w") as f:
                json.dump(users_data, f)
        
        return app_root_path

    yield _creator

    for d in created_dirs:
        d.cleanup()

# Sample valid data for tests
VALID_CONFIG_DATA = {
    "app": {"name": "TestApp", "version": "1.0", "debug": True, "allowed_hosts": ["*"], "max_queue_files": 10},
    "storage": {"upload_dir": "./uploads", "max_upload_size_mb": 5, "allowed_extensions": [".mp3"], "max_audio_secs": 300},
    "celery": {"name": "TestCelery", "broker_url": "memory://", "result_backend": "cache+memory://", "task_serializer": "json", "accept_content": ["json"]}
}
VALID_USERS_DATA = {
    "users": [
        {"username": "test", "password": "pass", "is_admin": False, "created_at": "2023-01-01T00:00:00Z"}
    ]
}

def test_config_from_dir_valid(temp_config_files_factory):
    app_root_path = temp_config_files_factory(VALID_CONFIG_DATA, VALID_USERS_DATA)
    config = Config.from_dir(app_root_path)
    
    assert isinstance(config, Config)
    assert config.app.name == "TestApp"
    assert config.storage.upload_dir == Path("./uploads") # Stored as Path
    assert len(config.users) == 1
    assert config.users[0].username == "test"

def test_config_from_dir_missing_config_json(temp_config_files_factory):
    app_root_path = temp_config_files_factory(VALID_CONFIG_DATA, VALID_USERS_DATA, create_config_json=False)
    with pytest.raises(FileNotFoundError, match="config.json"):
        Config.from_dir(app_root_path)

def test_config_from_dir_missing_users_json(temp_config_files_factory):
    app_root_path = temp_config_files_factory(VALID_CONFIG_DATA, VALID_USERS_DATA, create_users_json=False)
    with pytest.raises(FileNotFoundError, match="users.json"):
        Config.from_dir(app_root_path)

def test_config_from_dir_app_root_not_a_dir(temp_config_files_factory):
    # Create a file instead of a directory for app_root_path
    with tempfile.NamedTemporaryFile() as tmp_file:
        app_root_path_file = Path(tmp_file.name)
        with pytest.raises(ConfigurationError, match="not a directory"):
            Config.from_dir(app_root_path_file)

def test_config_from_dir_malformed_config_json(temp_config_files_factory):
    malformed_data = "{\"app\": \"not_a_dict\""
    etc_dir = temp_config_files_factory({}, VALID_USERS_DATA, create_config_json=False) / "etc"
    with open(etc_dir / "config.json", "w") as f:
        f.write(malformed_data)
    
    app_root_path = etc_dir.parent
    with pytest.raises(ConfigurationError, match="Error decoding JSON"):
        Config.from_dir(app_root_path)

def test_config_from_dir_missing_key_in_config(temp_config_files_factory):
    invalid_config_data = VALID_CONFIG_DATA.copy()
    del invalid_config_data["app"] # Remove a required key
    app_root_path = temp_config_files_factory(invalid_config_data, VALID_USERS_DATA)
    with pytest.raises(ConfigurationError, match="Missing required keys.*app"):
        Config.from_dir(app_root_path)

def test_get_app_dir_from_env_set_valid():
    dummy_path_str = "/tmp/dummy_app_dir_for_test"
    dummy_path = Path(dummy_path_str)
    dummy_path.mkdir(parents=True, exist_ok=True) # Ensure it exists as a dir
    
    with patch.dict(os.environ, {"BEAT_COUNTER_APP_DIR": dummy_path_str}):
        resolved_path = Config.get_app_dir_from_env()
        assert resolved_path == dummy_path.resolve()
    
    dummy_path.rmdir() # Clean up

def test_get_app_dir_from_env_not_set():
    with patch.dict(os.environ, {}, clear=True): # Ensure BEAT_COUNTER_APP_DIR is not set
        with pytest.raises(ConfigurationError, match="BEAT_COUNTER_APP_DIR environment variable is not set"):
            Config.get_app_dir_from_env()

def test_get_app_dir_from_env_path_is_file():
    with tempfile.NamedTemporaryFile() as tmp_file:
        path_to_file_str = tmp_file.name
        with patch.dict(os.environ, {"BEAT_COUNTER_APP_DIR": path_to_file_str}):
            with pytest.raises(ConfigurationError, match="not a valid directory"):
                Config.get_app_dir_from_env()
