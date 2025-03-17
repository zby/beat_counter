#!/usr/bin/env python3
"""Beat Detection Web Application.

This web application allows users to upload audio files, analyze them for beats,
and generate visualization videos marking each beat.
"""

# Standard library imports
import logging
import os
import pathlib
import shutil
import uuid
from datetime import datetime
from typing import Optional
import json

# Third-party imports
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Local imports
from beat_detection.utils.constants import AUDIO_EXTENSIONS
from web_app.storage import MetadataStorage, TaskExecutor, FileMetadataStorage, CeleryTaskExecutor, ANALYZING, ANALYZED, ANALYZING_FAILURE, COMPLETED, ERROR, GENERATING_VIDEO, VIDEO_ERROR

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

def create_app(
    metadata_storage: MetadataStorage = FileMetadataStorage(base_dir=str(UPLOAD_DIR)),
    task_executor: TaskExecutor = CeleryTaskExecutor()
) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Beat Detection App",
        description="Upload audio files, detect beats, and generate visualization videos",
        version="1.0.0"
    )

    # Add custom exception handler to log HTTP exceptions with details
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions and log their details."""
        logger.error(f"HTTP {exc.status_code}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )

    # Mount static files directory
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    # Set up templates
    templates = Jinja2Templates(directory=BASE_DIR / "templates")

    # Dependency to get storage and task executor
    def get_storage() -> MetadataStorage:
        return metadata_storage

    def get_task_executor() -> TaskExecutor:
        return task_executor

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
        executor: TaskExecutor = Depends(get_task_executor),
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
        
        # Use storage to get standardized paths and ensure directories exist
        job_dir = storage.ensure_job_directory(file_id)
        audio_file_path = storage.get_audio_file_path(file_id, file_extension)
        
        # Save the uploaded file with standardized name
        with open(audio_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Create metadata
        metadata = {
            "original_filename": filename,
            "audio_file_path": str(audio_file_path),
            "file_extension": file_extension,
            "upload_time": datetime.now().isoformat()
        }
        
        # Store metadata in the central storage
        storage.update_metadata(file_id, metadata)
        
        # Start beat detection task (now only needs file_id)
        task = executor.execute_beat_detection(file_id)
        
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
        executor: TaskExecutor = Depends(get_task_executor)
    ):
        """Get the processing status for a file."""
        # Use the new get_file_status method from storage
        status_data = await storage.get_file_status(file_id, executor)
        if not status_data:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Add debugging information
        if "beat_detection_task" not in status_data and "beat_detection" in status_data:
            task_id = status_data["beat_detection"]
            task_status = executor.get_task_status(task_id)
            status_data["beat_detection_task"] = {
                "id": task_id,
                "state": task_status.get("state", "UNKNOWN"),
                "result": task_status.get("result", None)
            }
        
        if "video_generation_task" not in status_data and "video_generation" in status_data:
            task_id = status_data["video_generation"]
            task_status = executor.get_task_status(task_id)
            status_data["video_generation_task"] = {
                "id": task_id,
                "state": task_status.get("state", "UNKNOWN"),
                "result": task_status.get("result", None)
            }
        
        return status_data

    @app.get("/processing_queue", response_class=HTMLResponse)
    async def get_processing_queue(
        request: Request,
        storage: MetadataStorage = Depends(get_storage),
        executor: TaskExecutor = Depends(get_task_executor)
    ):
        """Get the list of files currently in processing."""
        # Get all file metadata
        all_metadata = await storage.get_all_metadata()
        
        # Process each file's metadata to create a list of files with their status
        files_with_status = []
        for file_id, file_info in all_metadata.items():
            # Get filename from file metadata
            filename = file_info.get("original_filename", "Unknown file")
            
            # Get task statuses
            beat_task_id = file_info.get("beat_detection")
            video_task_id = file_info.get("video_generation")
            
            status = ERROR.lower()
            if beat_task_id:
                beat_status = executor.get_task_status(beat_task_id)
                if beat_status["state"] == "SUCCESS":
                    status = ANALYZED.lower()
                elif beat_status["state"] == "FAILURE":
                    status = ERROR.lower()
                else:
                    status = ANALYZING.lower()
            
            if video_task_id:
                video_status = executor.get_task_status(video_task_id)
                if video_status["state"] == "SUCCESS":
                    status = COMPLETED.lower()
                elif video_status["state"] == "FAILURE":
                    status = VIDEO_ERROR.lower()
                else:
                    status = GENERATING_VIDEO.lower()
            
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
        request: Request,
        file_id: str,
        storage: MetadataStorage = Depends(get_storage),
        executor: TaskExecutor = Depends(get_task_executor)
    ):
        """Confirm analysis and generate visualization video."""
        # Get file metadata
        metadata = await storage.get_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if file is in the right state
        status = await storage.get_file_status(file_id, executor)
        if status["status"] != ANALYZED:
            raise HTTPException(
                status_code=400,
                detail=f"File is not ready for confirmation. Current status: {status['status']}"
            )
        
        # Start video generation task
        try:
            # Start generation using just the file_id
            task = executor.execute_video_generation(file_id)
            
            # Update metadata
            storage.update_metadata(file_id, {
                "video_generation": task.id,
                "status": GENERATING_VIDEO
            })
            
            # Return success response
            return {"task_id": task.id}
            
        except Exception as e:
            logger.error(f"Error starting video generation: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error starting video generation: {str(e)}"
            )

    @app.get("/file/{file_id}")
    async def file_page(
        request: Request,
        file_id: str,
        storage: MetadataStorage = Depends(get_storage),
        executor: TaskExecutor = Depends(get_task_executor)
    ):
        """Render the file view page."""
        # Get file metadata
        metadata = await storage.get_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")
            
        # Get simplified file status - we don't need detailed task status in the initial page load
        file_status = await storage.get_file_status(file_id, executor)
        
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
        storage: MetadataStorage = Depends(get_storage),
        executor: TaskExecutor = Depends(get_task_executor)
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
    async def get_task_status(
        task_id: str,
        executor: TaskExecutor = Depends(get_task_executor)
    ):
        """Get the raw status of a Celery task.
        
        Returns the direct Celery task state and result without modification.
        """
        task_status = executor.get_task_status(task_id)
        
        # Add the task ID to the response
        task_status["id"] = task_id
        return task_status

    return app

# Create default app instance
app = create_app()

def main():
    """Entry point for the web application."""
    uvicorn.run("web_app.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
