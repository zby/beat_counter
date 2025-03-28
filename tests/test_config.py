import os
import shutil
import tempfile
from pathlib import Path
import pytest
from web_app.config import get_config

def test_get_config():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create an etc directory in the temp directory
        etc_dir = Path(temp_dir) / "etc"
        etc_dir.mkdir()
        
        # Copy the actual config files to the temp directory
        shutil.copy("etc/config.json.example", etc_dir / "config.json")
        shutil.copy("etc/users.json.example", etc_dir / "users.json")
        
        # Change to the temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Get the config
            config = get_config()
            
            # Verify the config was loaded correctly
            assert "app" in config
            assert "storage" in config
            assert "celery" in config
            
            # Verify some specific values from the actual config
            assert config["app"].name == "Beat Detection Web App"
            assert config["app"].version == "0.1.0"
            assert isinstance(config["storage"].allowed_extensions, list)
            assert ".mp3" in config["storage"].allowed_extensions
            
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

def test_get_config_missing_file():
    # Create a temporary directory without config files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Change to the temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Attempt to get config should raise FileNotFoundError
            with pytest.raises(FileNotFoundError):
                get_config()
                
        finally:
            # Restore original working directory
            os.chdir(original_cwd)


if __name__ == "__main__":
    test_get_config()
    test_get_config_missing_file()
