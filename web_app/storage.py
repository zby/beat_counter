"""Storage implementations for metadata management."""

from typing import Any, Dict, List, Optional, BinaryIO
import json
import logging
import os
import pathlib
from datetime import datetime
import fcntl
import time
import tempfile
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from web_app.config import StorageConfig

# Set up logger
logger = logging.getLogger(__name__)

class FileMetadataStorage:
    """File-based implementation of metadata storage with standardized directory structure."""
    
    def __init__(self, config: StorageConfig):
        """Initialize the storage with a configuration object.
        
        Args:
            config: Storage configuration object containing settings
        """
        self.max_audio_secs = config.max_audio_secs # Duration in seconds
        self.base_upload_dir = config.upload_dir
        self.allowed_extensions = config.allowed_extensions
        # Ensure the base directory exists
        self.base_upload_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"File metadata storage initialized with base directory: {self.base_upload_dir}")
    
    def get_metadata_file_path(self, file_id: str) -> pathlib.Path:
        """Get the path to the metadata JSON file for a file ID."""
        job_dir = self.get_job_directory(file_id)
        return job_dir / "metadata.json"
    
    def get_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific file."""
        metadata_file = self.get_metadata_file_path(file_id)
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metadata file as JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading metadata file: {e}")
            return None
    
    def update_metadata(self, file_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a specific file with file locking for atomicity."""
        # Ensure job directory exists
        job_dir = self.ensure_job_directory(file_id)
        metadata_file = self.get_metadata_file_path(file_id)
        lock_file = metadata_file.with_suffix('.lock')
        
        # Create lock file if it doesn't exist
        if not lock_file.exists():
            try:
                with open(lock_file, 'w') as f:
                    pass  # Just create an empty file
            except Exception as e:
                logger.warning(f"Failed to create lock file, proceeding without locking: {e}")
        
        # Acquire lock for atomic update
        try:
            # Open the lock file 
            with open(lock_file, 'r+') as lock_f:
                # Try to acquire an exclusive lock with timeout
                max_wait = 10  # Maximum wait time in seconds
                start_time = time.time()
                
                while True:
                    try:
                        # Try to get an exclusive lock
                        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break  # Lock acquired
                    except IOError:
                        # Lock not acquired, wait and retry
                        elapsed = time.time() - start_time
                        if elapsed > max_wait:
                            logger.warning(f"Timeout waiting for lock, proceeding without lock: {file_id}")
                            break
                        time.sleep(0.1)
                
                try:
                    # Read the latest metadata to ensure we have the most up-to-date version
                    existing = {}
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            try:
                                existing = json.load(f)
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON in metadata file, starting with empty metadata: {file_id}")
                    
                    # Deep update the metadata
                    self._deep_update(existing, metadata)
                    
                    # Write directly to the file since we have an exclusive lock
                    with open(metadata_file, 'w') as f:
                        json.dump(existing, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    
                finally:
                    # Release the lock
                    fcntl.flock(lock_f, fcntl.LOCK_UN)
        
        except Exception as e:
            logger.error(f"Error updating metadata file: {e}")
            
            # Fall back to non-atomic update on error
            try:
                logger.warning(f"Falling back to non-atomic update for {file_id}")
                existing = self.get_metadata(file_id) or {}
                self._deep_update(existing, metadata)
                with open(metadata_file, 'w') as f:
                    json.dump(existing, f, indent=2)
            except Exception as inner_e:
                logger.error(f"Fallback update also failed: {inner_e}")
                raise inner_e  # Re-raise the inner exception
            
            raise  # Re-raise the original exception
    
    def check_ready_for_confirmation(self, file_id: str) -> bool:
        """Check if a file is ready for video generation confirmation based on metadata.
        
        Args:
            file_id: The ID of the file to check
            
        Returns:
            bool: True if ready for confirmation, False otherwise
        """
        try:
            # Get current metadata
            metadata = self.get_metadata(file_id)
            if not metadata:
                logger.info(f"No metadata found for {file_id} during confirmation check.")
                return False
                
            # Check if beat detection was marked successful in metadata
            if metadata.get("beat_detection_status") != "success":
                logger.info(f"Beat detection status in metadata is not 'success' for {file_id}: {metadata.get('beat_detection_status')}")
                return False
                
            # Check if the beats file path is recorded in metadata
            if 'beats_file' not in metadata or not metadata.get('beats_file'):
                logger.info(f"Beats file path not found in metadata for {file_id}")
                return False

            # Optionally, check if the file *actually* exists at the path stored in metadata
            beats_file_path_str = metadata.get('beats_file')
            if not pathlib.Path(beats_file_path_str).exists():
                 logger.warning(f"Beats file path found in metadata ({beats_file_path_str}), but file does not exist on disk for {file_id}")
                 return False # Treat as not ready if file is missing despite metadata entry

            return True # All checks passed
            
        except Exception as e:
            logger.error(f"Error checking if file {file_id} is ready for confirmation: {e}")
            return False
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all files."""
        all_metadata = {}
        
        # Find all job directories
        try:
            for job_dir in self.base_upload_dir.iterdir():
                if job_dir.is_dir():
                    file_id = job_dir.name
                    metadata = self.get_metadata(file_id)
                    if metadata:
                        all_metadata[file_id] = metadata
        except Exception as e:
            logger.error(f"Error getting all metadata: {e}")
        
        return all_metadata
    
    def delete_metadata(self, file_id: str) -> bool:
        """Delete metadata for a specific file."""
        try:
            metadata_file = self.get_metadata_file_path(file_id)
            if metadata_file.exists():
                metadata_file.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting metadata file: {e}")
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
    
    def get_job_directory_creation_time(self, file_id: str) -> float:
        """Get the creation time of the job directory.
        
        Returns:
            Float timestamp representing creation time, or 0 if directory doesn't exist
        """
        try:
            job_dir = self.get_job_directory(file_id)
            if job_dir.exists():
                return job_dir.stat().st_ctime
            return 0
        except Exception as e:
            logger.error(f"Error getting job directory creation time: {e}")
            return 0
    
    def get_audio_file_path(self, file_id: str, file_extension: str = None) -> pathlib.Path:
        """Get the standardized path for the audio file."""
        # If extension provided, return direct path
        if file_extension:
            return self.get_job_directory(file_id) / f"audio{file_extension}"
        
        # If no extension provided, attempt to get from metadata
        metadata = self.get_metadata(file_id)
        if metadata and "file_extension" in metadata:
            return self.get_job_directory(file_id) / f"audio{metadata['file_extension']}"
        
        # Fallback - return path without extension (caller must handle this)
        return self.get_job_directory(file_id) / "audio"
    
    def get_beats_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the beats file."""
        return self.get_job_directory(file_id) / "beats.txt"
    
    def get_video_file_path(self, file_id: str) -> pathlib.Path:
        """Get the standardized path for the visualization video."""
        return self.get_job_directory(file_id) / "visualization.mp4"
    
    def ensure_job_directory(self, file_id: str) -> pathlib.Path:
        """Ensure the job directory exists and return its path."""
        job_dir = self.get_job_directory(file_id)
        job_dir.mkdir(exist_ok=True, parents=True)
        return job_dir
        
    def save_audio_file(self, file_id: str, file_extension: str, file_obj: BinaryIO, filename: str = None) -> pathlib.Path:
        """Save an uploaded audio file to the storage and return its path.
        Also creates and saves basic metadata about the file.
        
        Args:
            file_id: The unique ID for the file
            file_extension: The file extension of the audio file
            file_obj: A binary file-like object that supports read operations
            filename: Original filename (optional)
            
        Returns:
            Path to the saved audio file
        """
        # Ensure job directory exists
        self.ensure_job_directory(file_id)
        
        # Create a temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save the uploaded file temporarily
            with open(temp_path, "wb") as f:
                import shutil
                shutil.copyfileobj(file_obj, f)
            
            # Load and truncate the audio file
            audio = AudioSegment.from_file(temp_path)
            original_duration = len(audio) / 1000  # Convert to seconds
            max_duration_ms = self.max_audio_secs * 1000  # Convert seconds to milliseconds
            truncated_audio = audio[:max_duration_ms]
            
            # Get standardized path for the audio file
            audio_file_path = self.get_audio_file_path(file_id, file_extension)
            
            try:
                # For M4A files, we need to use specific parameters
                if file_extension.lower() == '.m4a':
                    truncated_audio.export(audio_file_path, format='mp4', codec='aac')
                else:
                    # For other formats, use the original format
                    truncated_audio.export(audio_file_path, format=file_extension[1:])
            except Exception as e:
                # If original format fails, try MP3 as fallback
                logger.warning(f"Failed to save in original format {file_extension}, falling back to MP3: {e}")
                audio_file_path = self.get_audio_file_path(file_id, '.mp3')
                truncated_audio.export(audio_file_path, format='mp3')
                file_extension = '.mp3'  # Update extension to match actual format
            
            # Create and save metadata
            metadata = {
                "audio_file_path": str(audio_file_path),
                "file_extension": file_extension,  # Store the actual extension used
                "upload_time": datetime.now().isoformat(),
                "duration_limit": self.max_audio_secs,  # Store in seconds
                "original_duration": original_duration,  # Store original duration
                "duration": len(truncated_audio) / 1000,  # Store truncated duration in seconds
                "original_filename": filename
            }
            
            self.update_metadata(file_id, metadata)
            
            return audio_file_path
            
        finally:
            # Clean up temporary file
            os.remove(temp_path) 