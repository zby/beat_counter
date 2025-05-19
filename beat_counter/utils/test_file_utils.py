"""Test for file_utils module."""

import os
import tempfile
from pathlib import Path

import pytest

from beat_counter.utils.file_utils import get_output_path


def test_get_output_path_with_default_extension():
    """Test that get_output_path correctly handles default extension."""
    with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_file:
        input_path = temp_file.name
        expected_output = os.path.splitext(input_path)[0] + ".beats"
        
        result = get_output_path(input_path)
        
        assert result == expected_output


def test_get_output_path_with_custom_extension():
    """Test that get_output_path handles custom extensions."""
    with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_file:
        input_path = temp_file.name
        expected_output = os.path.splitext(input_path)[0] + ".custom"
        
        result = get_output_path(input_path, extension=".custom")
        
        assert result == expected_output


def test_get_output_path_with_explicit_output():
    """Test that get_output_path respects explicit output paths."""
    with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_file:
        input_path = temp_file.name
        
        with tempfile.TemporaryDirectory() as temp_dir:
            explicit_output = os.path.join(temp_dir, "explicit_output.beats")
            
            result = get_output_path(input_path, output_path=explicit_output)
            
            assert result == explicit_output
            # Check that parent directory was created
            assert os.path.isdir(os.path.dirname(explicit_output))


def test_get_output_path_creates_directories():
    """Test that get_output_path creates parent directories for explicit outputs."""
    with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_file:
        input_path = temp_file.name
        
        with tempfile.TemporaryDirectory() as temp_dir:
            deep_dir = os.path.join(temp_dir, "deep", "nested", "dir")
            explicit_output = os.path.join(deep_dir, "output.beats")
            
            # Directory shouldn't exist yet
            assert not os.path.exists(deep_dir)
            
            result = get_output_path(input_path, output_path=explicit_output)
            
            # Directory should be created
            assert os.path.isdir(deep_dir)
            assert result == explicit_output