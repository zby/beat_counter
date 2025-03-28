"""Tests for the storage module."""

import pytest
import json
import pathlib
import tempfile
from datetime import datetime
import io
from pydub import AudioSegment
import numpy as np
from web_app.storage import FileMetadataStorage
from web_app.config import StorageConfig

@pytest.fixture
def storage():
    """Create a storage instance with a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        config = StorageConfig(
            upload_dir=temp_path,
            max_upload_size_mb=100,
            max_audio_secs=60,
            allowed_extensions=[".mp3", ".wav", ".m4a"]
        )
        storage_instance = FileMetadataStorage(config)
        yield storage_instance

def test_storage_initialization(storage):
    """Test that storage is properly initialized."""
    assert storage.max_audio_secs == 60  # 60 seconds in milliseconds
    assert storage.allowed_extensions == [".mp3", ".wav", ".m4a"]
    assert storage.base_upload_dir.exists()

def test_metadata_file_path(storage):
    """Test metadata file path generation."""
    file_id = "test123"
    expected_path = storage.base_upload_dir / file_id / "metadata.json"
    assert storage.get_metadata_file_path(file_id) == expected_path

def test_job_directory_creation(storage):
    """Test job directory creation and path management."""
    file_id = "test123"
    job_dir = storage.ensure_job_directory(file_id)
    
    assert job_dir == storage.base_upload_dir / file_id
    assert job_dir.exists()
    assert job_dir.is_dir()

def test_metadata_update_and_retrieval(storage):
    """Test metadata update and retrieval functionality."""
    file_id = "test123"
    test_metadata = {
        "test_key": "test_value",
        "nested": {
            "key": "value"
        }
    }
    
    # Test metadata update
    storage.update_metadata(file_id, test_metadata)
    
    # Test metadata retrieval
    retrieved_metadata = storage.get_metadata(file_id)
    assert retrieved_metadata == test_metadata
    
    # Test nested update
    update_metadata = {
        "nested": {
            "new_key": "new_value"
        }
    }
    storage.update_metadata(file_id, update_metadata)
    
    expected_metadata = {
        "test_key": "test_value",
        "nested": {
            "key": "value",
            "new_key": "new_value"
        }
    }
    assert storage.get_metadata(file_id) == expected_metadata

def test_metadata_deletion(storage):
    """Test metadata deletion functionality."""
    file_id = "test123"
    test_metadata = {"test": "value"}
    
    # Create and verify metadata
    storage.update_metadata(file_id, test_metadata)
    assert storage.get_metadata(file_id) == test_metadata
    
    # Delete metadata
    assert storage.delete_metadata(file_id)
    assert storage.get_metadata(file_id) is None

def test_get_all_metadata(storage):
    """Test retrieving all metadata."""
    # Create multiple test files
    test_files = [
        ("file1", {"key1": "value1"}),
        ("file2", {"key2": "value2"}),
        ("file3", {"key3": "value3"})
    ]
    
    for file_id, metadata in test_files:
        storage.update_metadata(file_id, metadata)
    
    # Get all metadata
    all_metadata = storage.get_all_metadata()
    
    # Verify all files are present
    assert len(all_metadata) == len(test_files)
    for file_id, metadata in test_files:
        assert file_id in all_metadata
        assert all_metadata[file_id] == metadata

def test_file_paths(storage):
    """Test various file path generation methods."""
    file_id = "test123"
    
    # Test audio file path
    audio_path = storage.get_audio_file_path(file_id, ".mp3")
    assert audio_path == storage.base_upload_dir / file_id / "audio.mp3"
    
    # Test beats file path
    beats_path = storage.get_beats_file_path(file_id)
    assert beats_path == storage.base_upload_dir / file_id / "beats.txt"
    
    # Test beat stats file path
    stats_path = storage.get_beat_stats_file_path(file_id)
    assert stats_path == storage.base_upload_dir / file_id / "beat_stats.json"
    
    # Test video file path
    video_path = storage.get_video_file_path(file_id)
    assert video_path == storage.base_upload_dir / file_id / "visualization.mp4"

def test_audio_file_saving(storage):
    """Test saving audio files with different formats."""
    file_id = "test123"
    
    # Create a simple audio segment for testing
    audio = AudioSegment.silent(duration=500)  # 500ms of silence
    
    # Test MP3 saving
    with tempfile.NamedTemporaryFile(suffix='.mp3') as temp_file:
        audio.export(temp_file.name, format='mp3')
        with open(temp_file.name, 'rb') as f:
            path = storage.save_audio_file(file_id, '.mp3', f, "test.mp3")
            assert path.exists()
            assert path.suffix == '.mp3'
    
    # Verify metadata was created
    metadata = storage.get_metadata(file_id)
    assert metadata is not None
    assert metadata['file_extension'] == '.mp3'
    assert metadata['original_filename'] == 'test.mp3'
    assert 'upload_time' in metadata
    assert metadata['duration_limit'] == 60  # Default from config
    assert metadata['original_duration'] == 0.5  # 500ms = 0.5s

def test_audio_duration_limit(storage):
    """Test that audio files are truncated to the duration limit."""
    file_id = "test123"
    
    # Create an audio segment longer than the limit
    duration_ms = storage.max_audio_secs * 1000 + 1000  # 1 second over limit
    audio = AudioSegment.silent(duration=duration_ms)
    
    # Save the audio file
    with tempfile.NamedTemporaryFile(suffix='.mp3') as temp_file:
        audio.export(temp_file.name, format='mp3')
        with open(temp_file.name, 'rb') as f:
            path = storage.save_audio_file(file_id, '.mp3', f, "test.mp3")
    
    # Load the saved file and verify duration
    saved_audio = AudioSegment.from_file(path)
    assert len(saved_audio) <= storage.max_audio_secs * 1000  # Convert seconds to milliseconds

def test_file_extension_handling(storage):
    """Test handling of different file extensions."""
    file_id = "test123"
    audio = AudioSegment.silent(duration=500)
    
    # Test M4A handling (should convert to MP4/AAC)
    with tempfile.NamedTemporaryFile(suffix='.m4a') as temp_file:
        audio.export(temp_file.name, format='mp4')
        with open(temp_file.name, 'rb') as f:
            path = storage.save_audio_file(file_id, '.m4a', f, "test.m4a")
            assert path.exists()
    
    # Verify metadata has correct extension
    metadata = storage.get_metadata(file_id)
    assert metadata['file_extension'] in ['.m4a', '.mp4']  # Either is acceptable
    
    # Test fallback to MP3 for unsupported format
    file_id = "test456"
    with tempfile.NamedTemporaryFile(suffix='.xyz') as temp_file:
        audio.export(temp_file.name, format='mp3')  # Use MP3 format but wrong extension
        with open(temp_file.name, 'rb') as f:
            path = storage.save_audio_file(file_id, '.xyz', f, "test.xyz")
            assert path.suffix == '.mp3'  # Should fall back to MP3 