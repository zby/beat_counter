"""Tests for the reproducibility module."""

import os
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from beat_counter.utils.reproducibility import get_git_info, save_reproducibility_info


@patch('subprocess.check_output')
def test_get_git_info(mock_check_output):
    """Test that get_git_info extracts git information correctly."""
    # Mock subprocess.check_output to return predefined values
    mock_check_output.side_effect = [
        b"abcd1234\n",  # git rev-parse HEAD
        b"diff --git a/file.py b/file.py\n+new line\n"  # git diff HEAD
    ]
    
    result = get_git_info()
    
    assert result == {
        "commit_hash": "abcd1234",
        "diff": "diff --git a/file.py b/file.py\n+new line\n"
    }
    assert mock_check_output.call_count == 2


@patch('shutil.copy2')
@patch('yaml.dump')
def test_save_reproducibility_info(mock_yaml_dump, mock_copy2):
    """Test that save_reproducibility_info creates the expected files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        git_info = {
            "commit_hash": "abcd1234",
            "diff": "sample diff"
        }
        config = {"key": "value"}
        
        save_reproducibility_info(
            output_dir=output_dir,
            git_info=git_info,
            config=config,
            save_genre_db=True
        )
        
        # Check that files were created
        assert os.path.isfile(output_dir / "git_commit.txt")
        assert os.path.isfile(output_dir / "git_diff.patch")
        
        # Check file contents
        with open(output_dir / "git_commit.txt") as f:
            assert f.read() == "abcd1234"
        
        with open(output_dir / "git_diff.patch") as f:
            assert f.read() == "sample diff"
        
        # Check that yaml.dump was called with the config
        mock_yaml_dump.assert_called_once()
        
        # If save_genre_db is True, shutil.copy2 should be called
        assert mock_copy2.call_count > 0