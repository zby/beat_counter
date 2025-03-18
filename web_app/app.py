#!/usr/bin/env python3
"""Beat Detection Web Application.

This web application allows users to upload audio files, analyze them for beats,
and generate visualization videos marking each beat.
"""

# Standard library imports
import logging
import os
import pathlib
import uuid
import json
from datetime import datetime
from typing import Optional, Dict, Any, Callable, Protocol

# Third-party imports
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Local imports
from beat_detection.utils.constants import AUDIO_EXTENSIONS
from web_app.celery_app import app as celery_app
from web_app.storage import MetadataStorage, FileMetadataStorage
from web_app.tasks import detect_beats_task, generate_video_task

# Constants for task states
ANALYZING = "ANALYZING"
ANALYZED = "ANALYZED"
ANALYZING_FAILURE = "ANALYZING_FAILURE"
GENERATING_VIDEO = "GENERATING_VIDEO"
COMPLETED = "COMPLETED"
VIDEO_ERROR = "VIDEO_ERROR"
ERROR = "ERROR"

# Set of all valid states for validation
VALID_STATES = {
    ANALYZING, ANALYZED, ANALYZING_FAILURE,
    GENERATING_VIDEO, COMPLETED, VIDEO_ERROR, ERROR
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the current directory
BASE_DIR = pathlib.Path(__file__).parent.absolute()

# Create temp directory for uploaded files
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Create output directory for processed files
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Define task service provider interfaces
class TaskStatus(Protocol):
    """Protocol for task status retrieval."""
    def __call__(self, task_id: str) -> Dict[str, Any]: ...

class TaskRunner(Protocol):
    """Protocol for task execution."""
    def __call__(self, file_id: str) -> Any: ...

class TaskServiceProvider:
    """Provider for task-related services with dependency injection support."""
    
    def __init__(
        self, 
        get_task_status_fn: TaskStatus = None,
        detect_beats_fn: TaskRunner = None,
        generate_video_fn: TaskRunner = None
    ):
        """Initialize the service provider with the given implementations."""
        self.get_task_status = get_task_status_fn or self._default_get_task_status
        self.detect_beats = detect_beats_fn or self._default_detect_beats
        self.generate_video = generate_video_fn or self._default_generate_video
    
    def _default_get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Default implementation for getting task status from Celery."""
        # Create initial response with task ID
        result_dict = {"id": task_id, "state": ERROR}
        
        try:
            # Get AsyncResult object
            async_result = celery_app.AsyncResult(task_id)
            
            # Get state
            result_dict["state"] = async_result.state
            
            # For completed tasks, get result details
            if async_result.state == "SUCCESS":
                result = async_result.result
                if isinstance(result, dict):
                    result_dict["result"] = result
            
            # For failed tasks, get error information
            elif async_result.state == "FAILURE":
                result_dict["error"] = str(async_result.result)
                
        except Exception as e:
            logger.error(f"Error getting task status for {task_id}: {e}")
            result_dict["error"] = str(e)
            
        return result_dict
    
    def _default_detect_beats(self, file_id: str) -> Any:
        """Default implementation for detecting beats using Celery task."""
        return detect_beats_task.delay(file_id)
    
    def _default_generate_video(self, file_id: str) -> Any:
        """Default implementation for generating video using Celery task."""
        return generate_video_task.delay(file_id)

# Global instance with default implementations
task_service = TaskServiceProvider()

# For backward compatibility
get_task_status = task_service.get_task_status

def create_app(
    metadata_storage: Optional[MetadataStorage] = None,
    task_provider: Optional[TaskServiceProvider] = None
) -> FastAPI:
    """Create and configure the FastAPI application.
    
    Args:
        metadata_storage: Optional MetadataStorage implementation to use
        task_provider: Optional TaskServiceProvider to use for task operations
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Beat Detection Web App",
        description="Web interface for audio beat detection and visualization",
        version="0.1.0"
    )
    
    # Configure services
    storage = metadata_storage or FileMetadataStorage(base_dir=str(UPLOAD_DIR))
    service = task_provider or task_service
    
    # Mount static files directory
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    
    # Set up Jinja templates
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    
    # Dependency for getting storage
    def get_storage():
        return storage

    # Dependency for getting task service provider
    def get_task_service():
        return service

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Render the main page."""
        return templates.TemplateResponse("index.html", {"request": request})

    @app.post("/upload")
    async def upload_audio(
        request: Request,
        file: UploadFile = File(...),
        analyze: Optional[bool] = Form(False),
        generate_video: Optional[bool] = Form(False),
        storage: MetadataStorage = Depends(get_storage),
        service: TaskServiceProvider = Depends(get_task_service)
    ):
        """Upload an audio file for processing."""
        # Validate file extension
        filename = file.filename
        file_extension = pathlib.Path(filename).suffix.lower()
        
        if file_extension not in AUDIO_EXTENSIONS:
            supported_formats = ", ".join(AUDIO_EXTENSIONS)
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {supported_formats}"
            )
        
        # Generate a unique ID for this upload
        file_id = str(uuid.uuid4())
        
        # Save the uploaded file using the storage, which also saves the basic metadata
        audio_file_path = storage.save_audio_file(file_id, file_extension, file.file, filename=filename)
        
        # Start beat detection task directly with file_id
        task = service.detect_beats(file_id)
        
        # Update metadata with task ID
        task_metadata = {
            "beat_detection": task.id
        }
        storage.update_metadata(file_id, task_metadata)
        
        # Check if this is an AJAX request (XMLHttpRequest)
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        
        if is_ajax:
            # For AJAX requests, return JSON with file_id
            return {"file_id": file_id}
        else:
            # For regular form submissions, redirect to file status page
            return RedirectResponse(url=f"/file/{file_id}", status_code=303)

    @app.get("/status/{file_id}")
    async def get_file_status(
        file_id: str,
        storage: MetadataStorage = Depends(get_storage),
        service: TaskServiceProvider = Depends(get_task_service)
    ):
        """Get the processing status for a file."""
        # Get file metadata from storage
        metadata = await storage.get_file_metadata(file_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Create a copy to avoid modifying the original metadata
        status_data = dict(metadata)
        
        # Add filename to the status data
        status_data["filename"] = metadata.get("original_filename", "Unknown file")

        # Get task statuses
        beat_task_id = metadata.get("beat_detection")
        video_task_id = metadata.get("video_generation")
        
        # Determine overall status based on task states
        overall_status = ERROR

        if beat_task_id:
            beat_task_status = service.get_task_status(beat_task_id)
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
            video_task_status = service.get_task_status(video_task_id)
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

        # If video file exists, it's completed regardless of task state
        if 'video_file' in status_data:
            status_data["status"] = COMPLETED
        # If beats file exists but no video task, it's just analyzed
        elif 'beats_file' in status_data and not video_task_id:
            status_data["status"] = ANALYZED
        
        return status_data

    @app.get("/processing_queue", response_class=HTMLResponse)
    async def get_processing_queue(
        request: Request,
        storage: MetadataStorage = Depends(get_storage),
        service: TaskServiceProvider = Depends(get_task_service)
    ):
        """Get the list of files currently in processing."""
        # Get all file metadata
        all_metadata = await storage.get_all_metadata()
        
        # Process each file's metadata to create a list of files with their status
        files_with_status = []
        for file_id, file_info in all_metadata.items():
            # Get detailed metadata for this file
            file_metadata = await storage.get_file_metadata(file_id)
            if not file_metadata:
                continue
                
            # Get filename from file metadata
            filename = file_metadata.get("filename", "Unknown file")
            
            # Get task statuses
            beat_task_id = file_metadata.get("beat_detection")
            video_task_id = file_metadata.get("video_generation")
            
            status = ERROR.lower()
            if beat_task_id:
                beat_status = service.get_task_status(beat_task_id)
                if beat_status["state"] == "SUCCESS":
                    status = ANALYZED.lower()
                elif beat_status["state"] == "FAILURE":
                    status = ERROR.lower()
                else:
                    status = ANALYZING.lower()
            
            if video_task_id:
                video_status = service.get_task_status(video_task_id)
                if video_status["state"] == "SUCCESS":
                    status = COMPLETED.lower()
                elif video_status["state"] == "FAILURE":
                    status = VIDEO_ERROR.lower()
                else:
                    status = GENERATING_VIDEO.lower()
            
            # Override status if files exist
            if 'video_file' in file_metadata:
                status = COMPLETED.lower()
            elif 'beats_file' in file_metadata and not video_task_id:
                status = ANALYZED.lower()
            
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
    async def confirm_analysis(
        file_id: str, 
        storage: MetadataStorage = Depends(get_storage),
        service: TaskServiceProvider = Depends(get_task_service)
    ):
        """Confirm successful analysis and generate visualization video."""
        # Get file metadata
        metadata = await storage.get_file_metadata(file_id)
        
        if not metadata:
            logger.error(f"File {file_id} not found")
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")
        
        # Log metadata for debugging
        #logger.info(f"Metadata for file {file_id}: {metadata}")
        
        # Get task statuses to determine overall status
        beat_task_id = metadata.get("beat_detection")
        if not beat_task_id:
            logger.error(f"No beat detection task found for file {file_id}")
            raise HTTPException(
                status_code=400, 
                detail=f"No beat detection task found for file {file_id}"
            )
        
        # Get beat task status
        beat_task_status = service.get_task_status(beat_task_id)
        logger.info(f"Beat task status for {file_id}: {beat_task_status}")
        
        # Check if the file is ready for confirmation
        if isinstance(storage, FileMetadataStorage):
            is_ready = storage.check_ready_for_confirmation(file_id, beat_task_status)
        else:
            # Fallback for other storage implementations
            is_ready = (beat_task_status.get("state") == "SUCCESS" and 'beats_file' in metadata)
        
        if not is_ready:
            logger.error(f"File {file_id} not ready for confirmation")
            raise HTTPException(
                status_code=400, 
                detail=f"File {file_id} not ready for confirmation. Beat detection must be completed successfully."
            )
        
        try:
            # Start video generation task
            task = service.generate_video(file_id)
            
            # Update metadata with video generation task
            storage.update_metadata(file_id, {"video_generation": task.id})
            
            return {"status": "ok", "message": "Video generation initiated", "task_id": task.id}
        except Exception as e:
            logger.exception(f"Error starting video generation for {file_id}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error starting video generation: {str(e)}"
            )

    @app.get("/file/{file_id}")
    async def file_page(
        request: Request,
        file_id: str,
        storage: MetadataStorage = Depends(get_storage),
        service: TaskServiceProvider = Depends(get_task_service)
    ):
        """Render the file view page."""
        # Get file metadata
        metadata = await storage.get_file_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")
            
        # Calculate full file status for the template
        file_status = await get_file_status(file_id, storage, service)
        
        # Just add the task IDs directly to the template data
        # The frontend will poll for task status directly from the /task endpoint
        template_data = {
            "request": request,
            "file_id": file_id,
            "file_status": file_status
        }
        
        return templates.TemplateResponse("file_view.html", template_data)

    @app.get("/download/{file_id}")
    async def download_video(
        file_id: str,
        storage: MetadataStorage = Depends(get_storage)
    ):
        """Download the generated video."""
        # Get file metadata
        metadata = await storage.get_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")
        
        # Get standardized video file path
        video_path = storage.get_video_file_path(file_id)
        
        # Check if the video file exists
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Generate download filename based on the original filename
        original_filename = metadata.get("original_filename", "audio")
        base_name = os.path.splitext(original_filename)[0]
        download_name = f"{base_name}_with_beats.mp4"
        
        # Serve the file
        return FileResponse(
            path=str(video_path),
            filename=download_name,
            media_type="video/mp4"
        )

    @app.get("/task/{task_id}")
    async def get_task_status_endpoint(
        task_id: str,
        service: TaskServiceProvider = Depends(get_task_service)
    ):
        """Get the raw status of a Celery task.
        
        Returns the direct Celery task state and result without modification.
        """
        task_status = service.get_task_status(task_id)
        
        return task_status

    return app

# Create default app instance
app = create_app()

def main():
    """Entry point for the web application."""
    uvicorn.run("web_app.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
