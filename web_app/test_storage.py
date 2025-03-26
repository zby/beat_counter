"""Mock implementations of storage for testing."""

import pytest
from typing import Any, Dict, Optional
from web_app.storage import MetadataStorage, FileMetadataStorage
import pathlib
from datetime import datetime
import unittest
import tempfile
import shutil
import os
from unittest.mock import MagicMock, patch
import json
import io
from pydub import AudioSegment
import numpy as np

class MockMetadataStorage(MetadataStorage):
    """In-memory implementation of metadata storage for testing."""
    
    def __init__(self, max_audio_duration: int = 60):
        """Initialize the mock storage.
        
        Args:
            max_audio_duration: Maximum audio duration in seconds (default: 60)
        """
        super().__init__(max_audio_duration)
        self.storage = {}
        self.base_upload_dir = pathlib.Path("web_app/uploads")
    
    async def get_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific file."""
        return self.storage.get(file_id)
    
    def update_metadata(self, file_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a specific file."""
        if file_id not in self.storage:
            self.storage[file_id] = {}
        self._deep_update(self.storage[file_id], metadata)
    
    async def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all files."""
        return self.storage.copy()

    def delete_metadata(self, file_id: str) -> bool:
        """Delete metadata for a specific file."""
        if file_id in self.storage:
            del self.storage[file_id]
            return True
        return False

    def _deep_update(self, target: Dict, source: Dict) -> None:
        """Helper function for deep updating dictionaries."""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    # Path management methods
    def get_job_directory(self, file_id: str) -> pathlib.Path:
        """Get the standardized job directory for a file ID."""
        return self.base_upload_dir / file_id
    
    def get_audio_file_path(self, file_id: str, file_extension: str = None) -> pathlib.Path:
        """Get the standardized path for the audio file."""
        job_dir = self.get_job_directory(file_id)
        
        # If extension not provided, try to get from metadata
        if file_extension is None:
            metadata = self.storage.get(file_id, {})
            file_extension = metadata.get("file_extension", ".mp3")  # Default to .mp3 if not found
        
        return job_dir / f"audio{file_extension}"
    
    def get_beats_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the beats file."""
        job_dir = self.get_job_directory(file_id)
        return job_dir / "beats.txt"
    
    def get_beat_stats_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the beat statistics file."""
        job_dir = self.get_job_directory(file_id)
        return job_dir / "beat_stats.json"
    
    def get_video_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the visualization video."""
        job_dir = self.get_job_directory(file_id)
        return job_dir / "visualization.mp4"
    
    def ensure_job_directory(self, file_id: str) -> pathlib.Path:
        """Ensure the job directory exists and return its path."""
        job_dir = self.get_job_directory(file_id)
        job_dir.mkdir(exist_ok=True, parents=True)
        return job_dir

    def save_audio_file(self, file_id: str, file_extension: str, file_obj, filename: str = None) -> pathlib.Path:
        """Save an uploaded audio file to the storage and return its path.
        Also creates and saves basic metadata about the file.
        
        Args:
            file_id: The unique ID for the file
            file_extension: The file extension of the audio file
            file_obj: A file-like object that supports read
            filename: Original filename (optional)
            
        Returns:
            Path to the saved audio file
        """
        # In the mock, we just need to update the metadata
        audio_file_path = self.get_audio_file_path(file_id, file_extension)
        
        # Create and save metadata if filename is provided
        if filename:
            metadata = {
                "original_filename": filename,
                "audio_file_path": str(audio_file_path),
                "file_extension": file_extension,
                "upload_time": datetime.now().isoformat()
            }
            
            self.update_metadata(file_id, metadata)
            
        return audio_file_path

    async def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """Get the file metadata and file existence information without task status computation."""
        metadata = await self.get_metadata(file_id)
        if not metadata:
            return None

        # Create response with basic file information
        status_data = {
            "file_id": file_id,
            "filename": metadata.get("original_filename"),
            "audio_file_path": metadata.get("audio_file_path")
        }

        # Add task IDs to the status data
        beat_task_id = metadata.get("beat_detection")
        video_task_id = metadata.get("video_generation")
        
        if beat_task_id:
            status_data["beat_detection"] = beat_task_id
            
        if video_task_id:
            status_data["video_generation"] = video_task_id

        # In the mock, we'll simulate file existence based on task completion in metadata
        beats_file_exists = metadata.get("beats_file_exists", False)
        video_file_exists = metadata.get("video_file_exists", False)
        
        status_data["beats_file_exists"] = beats_file_exists
        status_data["video_file_exists"] = video_file_exists
        
        # Add mock beat stats if beats file exists
        if beats_file_exists:
            status_data["beat_stats"] = {
                "bpm": 120.0,
                "beats": 100,
                "duration": 60.0,
                "time_signature": "4/4"
            }
        
        return status_data

@pytest.mark.asyncio
async def test_mock_metadata_storage():
    """Test the methods of MockMetadataStorage class."""
    storage = MockMetadataStorage()
    
    # Test initial state
    assert storage.storage == {}
    
    # Test get_metadata for non-existent file
    metadata = await storage.get_metadata("nonexistent")
    assert metadata is None
    
    # Test update_metadata for new file
    file_id = "test_file"
    initial_metadata = {
        "filename": "test.mp3",
        "file_path": "/path/to/test.mp3"
    }
    storage.update_metadata(file_id, initial_metadata)
    
    # Verify metadata was stored
    metadata = await storage.get_metadata(file_id)
    assert metadata == initial_metadata
    
    # Test deep update of metadata
    update_metadata = {
        "task_info": {
            "beat_detection": "task1",
            "status": "STARTED"
        }
    }
    storage.update_metadata(file_id, update_metadata)
    
    # Verify deep update worked
    metadata = await storage.get_metadata(file_id)
    assert metadata["filename"] == "test.mp3"  # Original data preserved
    assert metadata["task_info"]["beat_detection"] == "task1"  # New data added
    
    # Test get_all_metadata
    all_metadata = await storage.get_all_metadata()
    assert len(all_metadata) == 1
    assert all_metadata[file_id] == metadata
    
    # Test delete_metadata
    assert storage.delete_metadata(file_id) is True
    assert storage.delete_metadata("nonexistent") is False
    assert await storage.get_metadata(file_id) is None

@pytest.fixture
def temp_storage():
    """Create a temporary storage for testing."""
    temp_dir = tempfile.mkdtemp()
    storage = FileMetadataStorage(base_dir=temp_dir, max_audio_duration=1)  # 1 second max duration
    yield storage
    shutil.rmtree(temp_dir)

@pytest.mark.asyncio
async def test_truncate_audio(temp_storage):
    """Test that audio files are truncated to the specified duration."""
    # Create a 2-second audio clip with a simple sine wave
    sample_rate = 44100
    duration = 2  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    
    # Save the original audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        audio_segment.export(temp_file.name, format='wav')
        
        # Create a file ID and save the audio using storage
        file_id = "test_truncate"
        with open(temp_file.name, 'rb') as f:
            audio_path = temp_storage.save_audio_file(
                file_id=file_id,
                file_extension='.wav',
                file_obj=f
            )
        
        # Load the truncated audio
        truncated_audio = AudioSegment.from_wav(audio_path)
        
        # Verify the duration is 1 second (with some tolerance for rounding)
        assert abs(truncated_audio.duration_seconds - 1.0) < 0.1
        
        # Clean up the temporary file
        os.unlink(temp_file.name)
        
        # Verify the metadata contains the duration limit
        metadata = await temp_storage.get_metadata(file_id)
        assert metadata.get("duration_limit") == 1.0  # 1 second in seconds 