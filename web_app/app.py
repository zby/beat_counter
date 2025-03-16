#!/usr/bin/env python3
"""
Beat Detection Web Application

This web application allows users to upload audio files, analyze them for beats,
and generate visualization videos marking each beat.
"""

import os
import tempfile
import pathlib
import shutil
import time
import io
import contextlib
import sys
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from celery.result import AsyncResult
import uvicorn

# Import Redis-based metadata storage
from web_app.metadata import (
    update_file_metadata,
    get_all_file_metadata,
    delete_file_metadata,
    get_status
)

from beat_detection.core.detector import BeatDetector
from beat_detection.utils import file_utils, reporting
from beat_detection.utils.constants import AUDIO_EXTENSIONS
from beat_detection.utils.beat_file import load_beat_data
from beat_detection.cli.generate_videos import generate_counter_video

# Create FastAPI app
app = FastAPI(
    title="Beat Detection App",
    description="Upload audio files, detect beats, and generate visualization videos",
    version="1.0.0"
)

# Add custom exception handler to log HTTP exceptions with details
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Log the exception with its details
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    # Return the default FastAPI HTTPException response
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Get the current directory
BASE_DIR = pathlib.Path(__file__).parent.absolute()

# Mount static files directory
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Set up templates
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Create temp directory for uploaded files
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Create output directory for processed files
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# We'll use Redis-based metadata storage to store basic file information and track all tasks related to each file
# This allows us to maintain a mapping between file_id and all its associated tasks
# The metadata will be persistent across application restarts and shared between instances


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_audio(request: Request, file: UploadFile = File(...), analyze: Optional[bool] = Form(False), generate_video: Optional[bool] = Form(False)):
    """
    Upload an audio file for processing.
    
    Returns the file ID for subsequent API calls or redirects to the analysis page.
    Beat detection is automatically started after upload.
    """
    # Validate file extension
    filename = file.filename
    file_extension = pathlib.Path(filename).suffix.lower()
    
    if file_extension not in AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported formats: {', '.join(AUDIO_EXTENSIONS)}"
        )
    
    # Generate a unique ID for this upload
    file_id = str(uuid.uuid4())
    
    # Create a directory for this upload
    upload_path = UPLOAD_DIR / file_id
    upload_path.mkdir(exist_ok=True)
    
    # Save the uploaded file
    file_path = upload_path / filename
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Store only minimal required file information in Redis
    # Status is now derived from task metadata, so we don't store it here
    update_file_metadata(file_id, {
        "filename": filename,
        "original_filename": filename,  # Store original filename for reference
        "file_path": str(file_path),
        "upload_time": datetime.now().isoformat()
        # Task IDs will be stored by name (beat_detection, video_generation) when tasks are created
    })
    
    # Automatically start beat detection after upload
    from web_app.tasks import detect_beats_task
    task = detect_beats_task.delay(file_id, str(file_path))
    
    # Update metadata in Redis with the beat detection task ID
    update_file_metadata(file_id, {
        "beat_detection": task.id  # Store task ID by name
    })
    
    logger.info(f"Automatically started beat detection for file {file_id}, task ID: {task.id}")
    
    # Check if this is an AJAX request or a traditional form submission
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
    
    if is_ajax:
        # For AJAX requests, return JSON response with task information
        return {"file_id": file_id, "filename": filename, "task_id": task.id, "task_type": "beat_detection"}
    else:
        # For traditional form submissions, redirect to the file page
        return RedirectResponse(url=f"/file/{file_id}", status_code=303)


# The analyze_audio endpoint has been removed since beat detection now starts automatically after upload

@app.get("/status/{file_id}")
async def get_file_status(file_id: str):
    """Get the processing status for a file."""
    try:
        # Get the status directly from the metadata module
        # This now uses Celery's state field for task status
        status_data = await get_status(file_id)
        return status_data
    except Exception as e:
        logger.error(f"Error getting file status: {e}")
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/processing_queue", response_class=HTMLResponse)
async def get_processing_queue(request: Request):
    """
    Get the list of files currently in processing.
    
    Returns an HTML page with links to each file's status page.
    """
    # Get all file metadata from Redis
    all_metadata = get_all_file_metadata()
    
    # Process each file's metadata to create a list of files with their status
    files_with_status = []
    for file_id, file_info in all_metadata.items():
        # Get filename from file metadata
        filename = file_info.get("filename", "Unknown file")
        
        status_data = await get_status(file_id)
        status = status_data.get("status", "unknown") if status_data else "unknown"
        
        # Add to the list of files
        files_with_status.append({
            "file_id": file_id,
            "filename": filename,
            "status": status,
            "link": f"/file/{file_id}"
        })
    
    # Sort files by status (to group similar statuses together)
    files_with_status.sort(key=lambda x: x["status"])
    
    # Render the template with the file list
    return templates.TemplateResponse(
        "processing_queue.html", 
        {"request": request, "files": files_with_status}
    )




@app.post("/confirm/{file_id}")
async def confirm_analysis(request: Request, file_id: str):
    """
    Confirm the beat analysis and generate the video using Celery.
    
    This is an asynchronous operation. The client should poll the /status endpoint
    to check when processing is complete.
    """
    # Get the current file status
    current_status = await get_status(file_id)
    if not current_status:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Check if the file has been analyzed or if beat detection was successful
    beat_task = current_status.get("beat_detection_task", {})
    beat_task_success = beat_task and beat_task.get("state") == "SUCCESS"
    
    if not beat_task_success:
        raise HTTPException(status_code=400, detail="File has not been analyzed yet")
    
    # Get required file paths
    file_path = current_status.get("file_path")
    beats_file = beat_task.get("beats_file")

    if not file_path or not beats_file:
        raise HTTPException(status_code=400, detail="Required file paths not found in state")
    
    # Check if there's an existing video generation task
    existing_video_task_id = current_status.get("video_generation_task", {}).get("id")
    if existing_video_task_id:
        logger.info(f"Found existing video generation task: {existing_video_task_id}. Creating a new one.")
    
    # Start a new Celery task for video generation
    from web_app.tasks import generate_video_task
    task = generate_video_task.delay(file_id, file_path, beats_file)
    logger.info(f"Created new video generation task with ID: {task.id}")
    
    # Update metadata in Redis with the new video generation task ID
    # This will replace any existing task ID
    update_file_metadata(file_id, {
        "video_generation": task.id,  # Store task ID by name
        "status": "GENERATING_VIDEO"  # Update overall status
    })
    
    # Check if this is an AJAX request or a traditional form submission
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
    
    if is_ajax:
        # For AJAX requests, return JSON response with task ID and type
        return {"status": "generating_video", "file_id": file_id, "task_id": task.id, "task_type": "video_generation"}
    else:
        # For traditional form submissions, redirect back to the file view page
        return RedirectResponse(url=f"/file/{file_id}", status_code=303)


# The /beats/{file_id} endpoint has been removed as it's no longer needed
# It was previously used for chart visualization which has been removed

@app.get("/file/{file_id}")
async def file_page(request: Request, file_id: str):
    """
    Render the file view page for a specific file at any stage of processing.
    
    This single route handles all file states (analyzing, analyzed, video_generation, completed).
    The frontend JavaScript will determine what to display based on the file's status.
    """
    # Get the current status of the file from task metadata
    try:
        file_status = await get_status(file_id)
        if not file_status:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Error getting file status: {e}")
        # Return Internal Server Error instead of using fallback
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
    # Render the dedicated file view template with file information
    return templates.TemplateResponse(
        "file_view.html", 
        {
            "request": request,
            "file_id": file_id,
            "file_status": file_status
        }
    )

@app.get("/download/{file_id}")
async def download_video(file_id: str):
    """Download the generated video."""
    # Get the current file status
    current_status = await get_status(file_id)
    if not current_status:
        raise HTTPException(status_code=404, detail="File not found")
    
    if current_status.get("status") != "COMPLETED":
        raise HTTPException(status_code=400, detail="Video generation not completed")
    
    video_file = current_status.get("video_file")
    
    if not os.path.exists(video_file):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=video_file, 
        filename=os.path.basename(video_file),
        media_type="video/mp4"
    )


# These functions have been moved to the Celery task system where output capturing is now handled

# This function has been moved to the Celery task system (detect_beats_task in tasks.py)


# This function has been moved to the Celery task system (generate_video_task in tasks.py)





def main():
    """Entry point for the web application."""
    uvicorn.run("web_app.app:app", host="0.0.0.0", port=8000, reload=True)

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the status of a Celery task.
    """
    from celery.result import AsyncResult
    task_result = AsyncResult(task_id)
    
    # Get task status
    # Get the task state and ensure consistent case comparison
    task_state = task_result.state
    
    if task_state == 'PENDING':
        response = {
            'status': 'pending',
            'info': 'Task is pending'
        }
    elif task_state == 'STARTED':
        response = {
            'status': 'started',
            'info': task_result.info
        }
    elif task_state == 'SUCCESS':
        # For successful tasks, include the full result
        result = task_result.result
        response = {
            'status': 'success',
            'result': result
        }
        # If the result has its own status field, use it for consistency
        if isinstance(result, dict) and 'status' in result:
            response['status'] = result['status']
        
        # If this was a beat detection task and video generation is requested,
        # start the video generation task
        if task_result.result.get('status') == 'analyzed':
            file_id = task_result.result.get('file_id')
            
            # Get the file info using get_status
            file_info = await get_status(file_id) or {}
            
            # Check if we need to automatically generate a video
            if file_info.get('generate_video_after_analysis', False):
                # Get the file path and beats file from the task result
                file_path = task_result.result.get('file_path')
                beats_file = task_result.result.get('beats_file')
                
                if file_path and beats_file:
                    # Start video generation task
                    from web_app.tasks import generate_video_task
                    video_task = generate_video_task.delay(
                        file_id,
                        file_path,
                        beats_file
                    )
                    
                    # Update metadata in Redis with the video generation task ID
                    update_file_metadata(file_id, {
                        "video_generation": video_task.id,  # Store task ID by name
                        "generate_video_after_analysis": False  # Reset the flag
                    })
                    
                    # Add video task info to response
                    response['video_task'] = {
                        'id': video_task.id,
                        'status': 'started'
                    }
    elif task_state == 'FAILURE':
        response = {
            'status': 'failure',
            'error': str(task_result.result)
        }
    else:
        response = {
            'status': task_state.lower(),
            'info': str(task_result.info)
        }
    
    return response


if __name__ == "__main__":
    main()
