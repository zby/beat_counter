"""Storage implementations for metadata management."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import logging
import os
import pathlib
from datetime import datetime
import aiofiles
import asyncio

# Import constants from task_executor.py
from web_app.task_executor import ANALYZING, ANALYZED, ANALYZING_FAILURE, \
    GENERATING_VIDEO, COMPLETED, VIDEO_ERROR, ERROR, VALID_STATES

# Set up logger
logger = logging.getLogger(__name__)

class MetadataStorage(ABC):
    """Abstract base class for metadata storage."""
    
    @abstractmethod
    async def get_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific file."""
        pass
    
    @abstractmethod
    def update_metadata(self, file_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a specific file."""
        pass
    
    @abstractmethod
    async def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all files."""
        pass

    @abstractmethod
    def delete_metadata(self, file_id: str) -> bool:
        """Delete metadata for a specific file."""
        pass

    @abstractmethod
    async def get_file_status(self, file_id: str) -> Dict[str, Any]:
        """Get the processing status for a file."""
        pass
    
    # Path management methods - all implementations should provide these
    @abstractmethod
    def get_job_directory(self, file_id: str) -> pathlib.Path:
        """Get the standardized job directory for a file ID."""
        pass
    
    @abstractmethod
    def get_audio_file_path(self, file_id: str, file_extension: str = None) -> pathlib.Path:
        """Get the standardized path for the audio file."""
        pass
    
    @abstractmethod
    def get_beats_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the beats file."""
        pass
    
    @abstractmethod
    def get_beat_stats_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the beat statistics file."""
        pass
    
    @abstractmethod
    def get_video_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the visualization video."""
        pass
    
    @abstractmethod
    def ensure_job_directory(self, file_id: str) -> pathlib.Path:
        """Ensure the job directory exists and return its path."""
        pass

class FileMetadataStorage(MetadataStorage):
    """File-based metadata storage implementation."""
    
    def __init__(self, base_dir: str):
        """Initialize the file-based metadata storage.
        
        Args:
            base_dir: Base directory for file storage
        """
        self.base_upload_dir = pathlib.Path(base_dir)
        self.base_upload_dir.mkdir(parents=True, exist_ok=True)
        self._stats_cache = {}
    
    # Metadata management methods
    async def get_metadata(self, file_id: str) -> Dict[str, Any]:
        """Get metadata for a specific file."""
        metadata_file = self.get_job_directory(file_id) / "metadata.json"
        if not metadata_file.exists():
            return None
        
        async with aiofiles.open(metadata_file, "r") as f:
            content = await f.read()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse metadata for file {file_id}")
                return None
    
    def get_metadata_sync(self, file_id: str) -> Dict[str, Any]:
        """Synchronous version of get_metadata."""
        metadata_file = self.get_job_directory(file_id) / "metadata.json"
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.error(f"Failed to parse metadata for file {file_id}")
            return None
    
    def update_metadata(self, file_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a specific file."""
        job_dir = self.get_job_directory(file_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = job_dir / "metadata.json"
        
        # If metadata file exists, merge with existing data
        existing = self.get_metadata_sync(file_id) or {}
        
        # Update existing metadata with new data
        for key, value in metadata.items():
            existing[key] = value
        
        try:
            with open(metadata_file, "w") as f:
                json.dump(existing, f, indent=2)
        except IOError:
            logger.exception(f"Failed to write metadata for file {file_id}")
            raise
    
    async def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all files."""
        all_metadata = {}
        
        # Find all job directories in the base directory
        try:
            # List all directories in the base directory
            for job_dir in self.base_upload_dir.iterdir():
                if job_dir.is_dir():
                    file_id = job_dir.name
                    metadata = await self.get_metadata(file_id)
                    if metadata:
                        all_metadata[file_id] = metadata
        except Exception as e:
            logger.error(f"Error getting all metadata: {e}")
        
        return all_metadata
    
    def delete_metadata(self, file_id: str) -> bool:
        """Delete metadata for a specific file."""
        metadata_file = self.get_job_directory(file_id) / "metadata.json"
        if metadata_file.exists():
            try:
                metadata_file.unlink()
                return True
            except IOError:
                logger.exception(f"Failed to delete metadata for file {file_id}")
        return False
    
    async def get_file_status(self, file_id: str) -> Dict[str, Any]:
        """Get the processing status for a file."""
        # Import here to avoid circular imports
        from web_app.app import get_task_status
        
        metadata = await self.get_metadata(file_id)
        if not metadata:
            return None

        # Get task statuses
        beat_task_id = metadata.get("beat_detection")
        video_task_id = metadata.get("video_generation")

        status_data = {
            "file_id": file_id,
            "filename": metadata.get("original_filename"),
            "audio_file_path": metadata.get("audio_file_path")
        }

        # Determine overall status based on task states
        overall_status = ERROR

        if beat_task_id:
            beat_task_status = get_task_status(beat_task_id)
            # Ensure task ID is included
            beat_task_status["id"] = beat_task_id
            status_data["beat_detection_task"] = beat_task_status

            if beat_task_status["state"] == "SUCCESS":
                overall_status = ANALYZED
            elif beat_task_status["state"] == "FAILURE":
                overall_status = ANALYZING_FAILURE
            else:
                overall_status = ANALYZING

        if video_task_id:
            video_task_status = get_task_status(video_task_id)
            # Ensure task ID is included
            video_task_status["id"] = video_task_id
            status_data["video_generation_task"] = video_task_status

            if video_task_status["state"] == "SUCCESS":
                overall_status = COMPLETED
            elif video_task_status["state"] == "FAILURE":
                overall_status = VIDEO_ERROR
            else:
                overall_status = GENERATING_VIDEO

        # Use the highest priority status (video success > beat success > error states)
        status_data["status"] = overall_status

        # Check if the file exists on disk
        beats_file = self.get_beats_file_path(file_id)
        video_file = self.get_video_file_path(file_id)
        
        status_data["beats_file_exists"] = beats_file.exists()
        status_data["video_file_exists"] = video_file.exists()

        # If video file exists, it's completed regardless of task state
        if status_data["video_file_exists"]:
            status_data["status"] = COMPLETED
        # If beats file exists but no video task, it's just analyzed
        elif status_data["beats_file_exists"] and not video_task_id:
            status_data["status"] = ANALYZED

        # Attempt to get beat statistics if they exist
        if status_data["beats_file_exists"]:
            beat_stats_file = self.get_beat_stats_file_path(file_id)
            if beat_stats_file.exists():
                try:
                    beat_stats = self._parse_beat_stats_file(str(beat_stats_file))
                    status_data["beat_stats"] = beat_stats
                except Exception as e:
                    logger.error(f"Error parsing beat stats: {e}")
        
        return status_data
    
    def _parse_beat_stats_file(self, filepath: str) -> Dict[str, Any]:
        """Parse beat statistics file."""
        # Check if already cached
        if filepath in self._stats_cache:
            return self._stats_cache[filepath]
            
        try:
            with open(filepath, 'r') as f:
                stats = json.load(f)
                # Cache the result
                self._stats_cache[filepath] = stats
                return stats
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading beat stats file: {e}")
            return {}

    # Path management methods
    def get_job_directory(self, file_id: str) -> pathlib.Path:
        """Get the standardized job directory for a file ID."""
        return self.base_upload_dir / file_id
    
    def get_audio_file_path(self, file_id: str, file_extension: str = None) -> pathlib.Path:
        """Get the standardized path for the audio file."""
        # If extension provided, return direct path
        if file_extension:
            return self.get_job_directory(file_id) / f"original{file_extension}"
            
        # Otherwise, check for files with supported extensions
        from beat_detection.utils.constants import SUPPORTED_AUDIO_EXTENSIONS
        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            path = self.get_job_directory(file_id) / f"original{ext}"
            if path.exists():
                return path
                
        # Default to mp3 if no file found
        return self.get_job_directory(file_id) / "original.mp3"
    
    def get_beats_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the beats file."""
        return self.get_job_directory(file_id) / "beats.json"
    
    def get_beat_stats_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the beat statistics file."""
        return self.get_job_directory(file_id) / "beat_stats.json"
    
    def get_video_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the visualization video."""
        return self.get_job_directory(file_id) / "visualization.mp4"
    
    def ensure_job_directory(self, file_id: str) -> pathlib.Path:
        """Ensure the job directory exists and return its path."""
        job_dir = self.get_job_directory(file_id)
        job_dir.mkdir(exist_ok=True, parents=True)
        return job_dir 