"""
File metadata management using Redis.

This module provides functions for storing and retrieving file metadata
using Redis as a shared storage backend. This allows both the FastAPI app
and Celery tasks to access the same metadata.
"""

import json
import redis
import logging
from typing import Dict, Any, Optional
from functools import wraps

# Set up logger
logger = logging.getLogger(__name__)

# Connect to Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Test the connection
try:
    redis_client.ping()
    logger.info("Redis connection successful")
except redis.exceptions.ConnectionError as e:
    logger.error(f"Failed to connect to Redis: {e}")
    raise RuntimeError("Redis connection failed. Make sure Redis is running.")

# Prefix for file metadata keys
FILE_METADATA_PREFIX = 'file_metadata:'

# Helper function for deep updating dictionaries
def deep_update(target, source):
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            deep_update(target[key], value)
        else:
            target[key] = value


def update_file_metadata(file_id: str, updates: Dict[str, Any]) -> None:
    """
    Update metadata for a file in Redis.
    
    Parameters:
    -----------
    file_id : str
        The unique identifier for the file
    updates : Dict[str, Any]
        Dictionary containing the updates to apply
    """
    # Get existing metadata or initialize empty dict
    metadata = get_file_metadata(file_id) or {}
    
    # Deep update the metadata
    deep_update(metadata, updates)
    
    # Save updated metadata to Redis
    try:
        redis_key = f"{FILE_METADATA_PREFIX}{file_id}"
        # Use a reasonable expiration time (7 days) to prevent Redis from growing indefinitely
        redis_client.set(redis_key, json.dumps(metadata), ex=604800)  # 7 days in seconds
    except Exception as e:
        logger.error(f"Error saving metadata to Redis: {e}")
        raise


def get_file_metadata(file_id: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a file from Redis.
    
    Parameters:
    -----------
    file_id : str
        The unique identifier for the file
        
    Returns:
    --------
    Dict[str, Any] or None
        The file metadata or None if not found
    """
    try:
        redis_key = f"{FILE_METADATA_PREFIX}{file_id}"
        metadata_json = redis_client.get(redis_key)
        
        if metadata_json:
            return json.loads(metadata_json)
        return None
    except Exception as e:
        logger.error(f"Error getting metadata from Redis: {e}")
        raise


def get_all_file_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all files from Redis.
    
    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Dictionary mapping file IDs to their metadata
    """
    try:
        all_metadata = {}
        
        # Get all keys matching the file metadata prefix
        for key in redis_client.keys(f"{FILE_METADATA_PREFIX}*"):
            file_id = key.replace(FILE_METADATA_PREFIX, '')
            metadata = get_file_metadata(file_id)
            if metadata:
                all_metadata[file_id] = metadata
        
        return all_metadata
    except Exception as e:
        logger.error(f"Error getting all metadata from Redis: {e}")
        raise


def update_progress(file_id: str, status: str, percent: float) -> None:
    """
    Update progress for a file in Redis.
    
    Parameters:
    -----------
    file_id : str
        The unique identifier for the file
    status : str
        Status message
    percent : float
        Progress percentage (0-100)
    
    Note:
    -----
    DEPRECATED: This function is maintained for backward compatibility only.
    New code should update task state directly instead of using this function.
    Progress information should be stored in task metadata, not file metadata.
    """
    logger.warning("update_progress is deprecated. Use task metadata for progress tracking instead.")
    # We no longer update file metadata with progress information
    # Progress should be tracked in task metadata only
    pass


def delete_file_metadata(file_id: str) -> bool:
    """
    Delete metadata for a file from Redis.
    
    Parameters:
    -----------
    file_id : str
        The unique identifier for the file
        
    Returns:
    --------
    bool
        True if metadata was deleted, False if not found
    """
    try:
        redis_key = f"{FILE_METADATA_PREFIX}{file_id}"
        return redis_client.delete(redis_key) > 0
    except Exception as e:
        logger.error(f"Error deleting metadata from Redis: {e}")
        raise


from typing import Tuple, Optional, Dict, Any
# Import AsyncResult from the app directly instead of from celery.result
from web_app.celery_app import app


def extract_task_output(task_data: Dict[str, Any], output_key: str) -> Optional[Dict[str, str]]:
    """
    Extract stdout/stderr output from task data.
    
    Parameters:
    -----------
    task_data : Dict[str, Any]
        The task data dictionary
    output_key : str
        The key for the output data in the task dictionary
        
    Returns:
    --------
    Optional[Dict[str, str]]
        Dictionary containing stdout/stderr output or None if not available
    """
    if not task_data or not isinstance(task_data, dict):
        return None
        
    # Check if output data is available
    if output_key in task_data and isinstance(task_data[output_key], dict):
        return task_data[output_key]
    
    return None


def update_task_status_with_output(task_status: Dict[str, Any], task_data: Dict[str, Any], task_type: str) -> None:
    """
    Update task status with stdout/stderr output information.
    
    Parameters:
    -----------
    task_status : Dict[str, Any]
        The task status dictionary to update
    task_data : Dict[str, Any]
        The task data dictionary containing output information
    task_type : str
        The type of task ('beat_detection' or 'video_generation')
    """
    if not task_data or not isinstance(task_data, dict):
        return
    
    # Determine the output key based on task type
    output_key = f"{task_type}_output"
    
    # Extract and include stdout/stderr output if available
    output_data = extract_task_output(task_data, output_key)
    if output_data:
        task_status["output"] = output_data

async def get_beat_detection_status(beat_detection_task_id: str) -> Tuple[Dict[str, Any], Optional[object]]:
    """
    Get the status of a beat detection task.
    
    Parameters:
    -----------
    beat_detection_task_id : str
        The ID of the beat detection task
        
    Returns:
    --------
    Tuple[Dict[str, Any], Optional[AsyncResult]]
        A tuple containing the task status dictionary and the AsyncResult object
    """
    # Create a basic task status object with just the essential information
    task_status = {
        "id": beat_detection_task_id,
        "type": "beat_detection",
        "status": "UNKNOWN"
    }
    
    # Safely create and use the AsyncResult
    try:
        # Use the app's AsyncResult method to ensure proper backend configuration
        task_result = app.AsyncResult(beat_detection_task_id)
        
        # Safely get the task state
        try:
            # Get the state and ensure it's uppercase for consistency
            state = task_result.state
            if state:
                task_status["status"] = state.upper()
        except (AttributeError, Exception) as e:
            logger.error(f"Error getting task state: {e}")
            return task_status, None
    except Exception as e:
        logger.error(f"Error creating AsyncResult: {e}")
        return task_status, None
    
    # Handle different task states to gather task-specific information
    if task_status["status"] == 'SUCCESS':
        try:
            # Get the task result
            result = None
            try:
                result = task_result.result
            except (AttributeError, Exception) as e:
                logger.error(f"Error getting task result: {e}")
            
            if result and isinstance(result, dict):
                # For beat detection tasks, extract the beats_file path
                if "beats_file" in result:
                    task_status["beats_file"] = result["beats_file"]
                    task_status["file_path"] = result.get("file_path")
                    
                    # Add beat statistics if available
                    if "stats" in result:
                        task_status["stats"] = result["stats"]
                
                # Add any warning if present
                if "warning" in result:
                    task_status["warning"] = result["warning"]
                    
                # Extract stdout/stderr output if available
                update_task_status_with_output(task_status, result, "beat_detection")
        except Exception as e:
            logger.error(f"Error processing task result: {e}")
    
    # For in-progress tasks, include progress information
    elif task_status["status"] in ['STARTED', 'PROGRESS']:
        try:
            task_info_data = None
            try:
                task_info_data = task_result.info
            except (AttributeError, Exception) as e:
                logger.error(f"Error getting task info: {e}")
            
            if task_info_data and isinstance(task_info_data, dict):
                # Include progress information
                if "progress" in task_info_data:
                    task_status["progress"] = task_info_data["progress"]
                
                # Include file paths if available
                if "file_path" in task_info_data:
                    task_status["file_path"] = task_info_data["file_path"]
                if "beats_file" in task_info_data:
                    task_status["beats_file"] = task_info_data["beats_file"]
                    
                # Extract stdout/stderr output if available
                update_task_status_with_output(task_status, task_info_data, "beat_detection")
        except Exception as e:
            logger.error(f"Error processing task progress: {e}")
    
    # For failed tasks, include error information
    elif task_status["status"] == 'FAILURE':
        try:
            error_result = None
            try:
                error_result = task_result.result
            except (AttributeError, Exception) as e:
                logger.error(f"Error getting failure result: {e}")
            
            error_msg = str(error_result) if error_result else "Unknown error"
            task_status["error"] = error_msg
        except Exception as e:
            logger.error(f"Error processing task failure: {e}")
    
    return task_status, task_result


async def get_video_generation_status(video_task_id: str) -> Tuple[Dict[str, Any], Optional[object]]:
    """
    Get the status of a video generation task.
    
    Parameters:
    -----------
    video_task_id : str
        The ID of the video generation task
        
    Returns:
    --------
    Tuple[Dict[str, Any], Optional[AsyncResult]]
        A tuple containing the task status dictionary and the AsyncResult object
    """
    # Create a basic task status object with just the essential information
    task_status = {
        "id": video_task_id,
        "type": "video_generation",
        "status": "UNKNOWN"
    }
    
    # Safely create and use the AsyncResult
    try:
        # Use the app's AsyncResult method to ensure proper backend configuration
        task_result = app.AsyncResult(video_task_id)
        
        # Safely get the task state
        try:
            # Get the state and ensure it's uppercase for consistency
            state = task_result.state
            if state:
                task_status["status"] = state.upper()
        except (AttributeError, Exception) as e:
            logger.error(f"Error getting task state: {e}")
            return task_status, None
    except Exception as e:
        logger.error(f"Error creating AsyncResult: {e}")
        return task_status, None
    
    # Handle different task states to gather task-specific information
    if task_status["status"] == 'SUCCESS':
        try:
            # Get the task result
            result = None
            try:
                result = task_result.result
            except (AttributeError, Exception) as e:
                logger.error(f"Error getting task result: {e}")
            
            if result and isinstance(result, dict):
                # For video generation tasks, extract the video_file path
                if "video_file" in result:
                    task_status["video_file"] = result["video_file"]
                
                # Add any warning if present
                if "warning" in result:
                    task_status["warning"] = result["warning"]
                    
                # Extract stdout/stderr output if available
                update_task_status_with_output(task_status, result, "video_generation")
        except Exception as e:
            logger.error(f"Error processing task result: {e}")
    
    # For in-progress tasks, include progress information
    elif task_status["status"] in ['STARTED', 'PROGRESS']:
        try:
            task_info_data = None
            try:
                task_info_data = task_result.info
            except (AttributeError, Exception) as e:
                logger.error(f"Error getting task info: {e}")
            
            if task_info_data and isinstance(task_info_data, dict):
                # Include progress information
                if "progress" in task_info_data:
                    task_status["progress"] = task_info_data["progress"]
                
                # Include file paths if available
                if "file_path" in task_info_data:
                    task_status["file_path"] = task_info_data["file_path"]
                if "beats_file" in task_info_data:
                    task_status["beats_file"] = task_info_data["beats_file"]
                    
                # Extract stdout/stderr output if available
                update_task_status_with_output(task_status, task_info_data, "beat_detection")
        except Exception as e:
            logger.error(f"Error processing task progress: {e}")
    
    # For failed tasks, include error information
    elif task_status["status"] == 'FAILURE':
        try:
            error_result = None
            try:
                error_result = task_result.result
            except (AttributeError, Exception) as e:
                logger.error(f"Error getting failure result: {e}")
            
            error_msg = str(error_result) if error_result else "Unknown error"
            task_status["error"] = error_msg
        except Exception as e:
            logger.error(f"Error processing task failure: {e}")
    
    return task_status, task_result


async def get_status(file_id: str) -> Dict[str, Any]:
    """
    Get the processing status for a file.
    
    Parameters:
    -----------
    file_id : str
        The unique identifier for the file
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the file status information
        
    Raises:
    -------
    Exception
        If the file is not found or there's an error retrieving the status
    """
    # Get basic file info from Redis
    file_info = get_file_metadata(file_id)
    if not file_info:
        raise Exception("File not found")
    
    # Initialize status data with minimal information
    status_data = {
        "file_id": file_id,
        "filename": file_info.get("filename", ""),
        "original_filename": file_info.get("original_filename", ""),
        "upload_time": file_info.get("upload_time", ""),
        "file_path": file_info.get("file_path", "")
        # Named tasks will be added as needed
    }
    
    # Get task IDs directly by name
    # Check for both key formats for backward compatibility
    beat_detection_task_id = file_info.get("beat_detection") or file_info.get("beat_detection_task_id")
    video_task_id = file_info.get("video_generation") or file_info.get("video_generation_task_id")
    
    # Process each task and add its status to the response
    latest_beat_detection_task = None
    latest_video_generation_task = None
    
    # Process beat detection task if it exists
    if beat_detection_task_id:
        task_status, task_result = await get_beat_detection_status(beat_detection_task_id)
        latest_beat_detection_task = task_result
        status_data["beat_detection_task"] = task_status
    
    # Process video generation task if it exists
    if video_task_id:
        task_status, task_result = await get_video_generation_status(video_task_id)
        latest_video_generation_task = task_result
        status_data["video_generation_task"] = task_status
    
    # Now derive the overall file status from the tasks
    # Priority: video generation > beat detection
    if latest_video_generation_task:
        try:
            # Get the task state and ensure consistent case comparison
            task_state = latest_video_generation_task.state
            if task_state == 'SUCCESS':
                result = latest_video_generation_task.result
                if result and isinstance(result, dict):
                    status_from_result = result.get("status", "COMPLETED")
                    status_data["status"] = status_from_result.upper() if isinstance(status_from_result, str) else "COMPLETED"
                    if "video_file" in result:
                        status_data["video_file"] = result["video_file"]
                    if "file_path" in result:
                        status_data["file_path"] = result["file_path"]
                    if "beats_file" in result:
                        status_data["beats_file"] = result["beats_file"]
                    # We no longer need to copy stats or progress from either task
                    # The frontend will look for them in the respective task objects
            elif task_state in ['STARTED', 'PROGRESS']:
                status_data["status"] = "GENERATING_VIDEO"
                # We don't need to copy task info to the top level
                # The frontend will access it through video_generation_task
                # Just keep the file_path and beats_file for backward compatibility
                task_info = latest_video_generation_task.info
                if task_info and isinstance(task_info, dict):
                    if "file_path" in task_info:
                        status_data["file_path"] = task_info["file_path"]
                    if "beats_file" in task_info:
                        status_data["beats_file"] = task_info["beats_file"]
            elif task_state == 'FAILURE':
                status_data["status"] = "ERROR"
                error_result = latest_video_generation_task.result
                error_msg = str(error_result) if error_result else "Unknown error"
                status_data["error"] = error_msg
        except Exception as e:
            logger.error(f"Error deriving status from video task: {e}")
    
    # If no video task or video task is not definitive, check beat detection task
    elif latest_beat_detection_task:
        try:
            task_state = latest_beat_detection_task.state
            if task_state == 'SUCCESS':
                # Always set the status to ANALYZED when beat detection completes successfully
                status_data["status"] = "ANALYZED"
                result = latest_beat_detection_task.result
                if result and isinstance(result, dict):
                    if "beats_file" in result:
                        status_data["beats_file"] = result["beats_file"]
                    if "file_path" in result:
                        status_data["file_path"] = result["file_path"]
                    # We don't need to copy stats or progress to the top level anymore
                    # The frontend will look for them in beat_detection_task
            elif task_state in ['STARTED', 'PROGRESS']:
                status_data["status"] = "ANALYZING"
                # We don't need to copy task info to the top level
                # The frontend will access it through beat_detection_task
                # Just keep the file_path for backward compatibility
                task_info = latest_beat_detection_task.info
                if task_info and isinstance(task_info, dict) and "file_path" in task_info:
                    status_data["file_path"] = task_info["file_path"]
            elif task_state == 'FAILURE':
                status_data["status"] = "ERROR"
                error_result = latest_beat_detection_task.result
                error_msg = str(error_result) if error_result else "Unknown error"
                status_data["error"] = error_msg
        except Exception as e:
            logger.error(f"Error deriving status from beat detection task: {e}")
    else:
        # If no tasks are found, set status to UPLOADED
        status_data["status"] = "UPLOADED"
    
    return status_data
