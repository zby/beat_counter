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

# Third-party imports
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Local imports
from beat_detection.utils.constants import AUDIO_EXTENSIONS
from web_app.storage import MetadataStorage, TaskExecutor, RedisMetadataStorage, CeleryTaskExecutor

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
    metadata_storage: MetadataStorage = RedisMetadataStorage(),
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
        
        # Create a directory for this upload
        upload_path = UPLOAD_DIR / file_id
        upload_path.mkdir(exist_ok=True)
        
        # Save the uploaded file
        file_path = upload_path / filename
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Store metadata
        metadata = {
            "filename": filename,
            "original_filename": filename,
            "file_path": str(file_path),
            "upload_time": datetime.now().isoformat()
        }
        storage.update_metadata(file_id, metadata)
        
        # Start beat detection task
        task = executor.execute_beat_detection(file_id, str(file_path))
        
        # Update metadata with task ID
        storage.update_metadata(file_id, {
            "beat_detection": task.id
        })
        
        logger.info(f"Started beat detection for file {file_id}, task ID: {task.id}")
        
        # Handle response based on request type
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        
        if is_ajax:
            return {
                "file_id": file_id,
                "filename": filename,
                "task_id": task.id,
                "task_type": "beat_detection"
            }
        
        # For non-AJAX requests, always return a redirect
        return RedirectResponse(url=f"/file/{file_id}", status_code=303)

    @app.get("/status/{file_id}")
    async def get_file_status(
        file_id: str,
        storage: MetadataStorage = Depends(get_storage),
        executor: TaskExecutor = Depends(get_task_executor)
    ):
        """Get the processing status for a file."""
        try:
            metadata = await storage.get_metadata(file_id)
            if not metadata:
                raise HTTPException(status_code=404, detail="File not found")
            
            # Get task statuses
            beat_task_id = metadata.get("beat_detection")
            video_task_id = metadata.get("video_generation")
            
            status_data = {
                "file_id": file_id,
                "filename": metadata.get("filename"),
                "file_path": metadata.get("file_path")
            }
            
            if beat_task_id:
                beat_task_status = executor.get_task_status(beat_task_id)
                status_data["beat_detection_task"] = beat_task_status
            
            if video_task_id:
                video_task_status = executor.get_task_status(video_task_id)
                status_data["video_generation_task"] = video_task_status
            
            return status_data
        except Exception as e:
            logger.error(f"Error getting file status: {e}")
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/processing_queue", response_class=HTMLResponse)
    async def get_processing_queue(
        request: Request,
        storage: MetadataStorage = Depends(get_storage),
        executor: TaskExecutor = Depends(get_task_executor)
    ):
        """Get the list of files currently in processing."""
        # Get all file metadata
        all_metadata = storage.get_all_metadata()
        
        # Process each file's metadata to create a list of files with their status
        files_with_status = []
        for file_id, file_info in all_metadata.items():
            # Get filename from file metadata
            filename = file_info.get("filename", "Unknown file")
            
            # Get task statuses
            beat_task_id = file_info.get("beat_detection")
            video_task_id = file_info.get("video_generation")
            
            status = "unknown"
            if beat_task_id:
                beat_status = executor.get_task_status(beat_task_id)
                if beat_status["state"] == "SUCCESS":
                    status = "analyzed"
                elif beat_status["state"] == "FAILURE":
                    status = "failed"
                else:
                    status = "processing"
            
            if video_task_id:
                video_status = executor.get_task_status(video_task_id)
                if video_status["state"] == "SUCCESS":
                    status = "completed"
                elif video_status["state"] == "FAILURE":
                    status = "failed"
                else:
                    status = "generating_video"
            
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
        """Confirm the beat analysis and generate the video."""
        # Get the current file status
        metadata = await storage.get_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get task statuses
        beat_task_id = metadata.get("beat_detection")
        if not beat_task_id:
            raise HTTPException(status_code=400, detail="No beat detection task found")
        
        beat_task_status = executor.get_task_status(beat_task_id)
        if beat_task_status["state"] != "SUCCESS":
            raise HTTPException(status_code=400, detail="File has not been analyzed yet")
        
        # Get required file paths
        file_path = metadata.get("file_path")
        beats_file = beat_task_status.get("result", {}).get("beats_file")
        
        if not file_path or not beats_file:
            raise HTTPException(status_code=400, detail="Required file paths not found")
        
        # Start video generation task
        task = executor.execute_video_generation(file_id, file_path, beats_file)
        
        # Update metadata
        storage.update_metadata(file_id, {
            "video_generation": task.id,
            "status": "GENERATING_VIDEO"
        })
        
        logger.info(f"Started video generation for file {file_id}, task ID: {task.id}")
        
        # Handle response based on request type
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        
        if is_ajax:
            return {
                "status": "generating_video",
                "file_id": file_id,
                "task_id": task.id,
                "task_type": "video_generation"
            }
        
        # For non-AJAX requests, always return a redirect
        return RedirectResponse(url=f"/file/{file_id}", status_code=303)

    @app.get("/file/{file_id}")
    async def file_page(
        request: Request,
        file_id: str,
        storage: MetadataStorage = Depends(get_storage),
        executor: TaskExecutor = Depends(get_task_executor)
    ):
        """Render the file view page."""
        try:
            metadata = await storage.get_metadata(file_id)
            if not metadata:
                raise HTTPException(status_code=404, detail="File not found")
            
            # Get task statuses
            beat_task_id = metadata.get("beat_detection")
            video_task_id = metadata.get("video_generation")
            
            file_status = {
                "file_id": file_id,
                "filename": metadata.get("filename"),
                "file_path": metadata.get("file_path")
            }
            
            if beat_task_id:
                beat_task_status = executor.get_task_status(beat_task_id)
                file_status["beat_detection_task"] = beat_task_status
            
            if video_task_id:
                video_task_status = executor.get_task_status(video_task_id)
                file_status["video_generation_task"] = video_task_status
            
            return templates.TemplateResponse(
                "file_view.html",
                {
                    "request": request,
                    "file_id": file_id,
                    "file_status": file_status
                }
            )
        except Exception as e:
            logger.error(f"Error getting file status: {e}")
            raise HTTPException(status_code=500, detail="Internal Server Error")

    @app.get("/download/{file_id}")
    async def download_video(
        file_id: str,
        storage: MetadataStorage = Depends(get_storage),
        executor: TaskExecutor = Depends(get_task_executor)
    ):
        """Download the generated video."""
        metadata = await storage.get_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        video_task_id = metadata.get("video_generation")
        if not video_task_id:
            raise HTTPException(status_code=404, detail="No video generation task found")
        
        video_task_status = executor.get_task_status(video_task_id)
        if video_task_status["state"] != "SUCCESS":
            raise HTTPException(status_code=400, detail="Video generation not completed")
        
        video_file = video_task_status.get("result", {}).get("video_file")
        if not video_file or not os.path.exists(video_file):
            raise HTTPException(status_code=404, detail="Video file not found")
        
        return FileResponse(
            path=video_file,
            filename=os.path.basename(video_file),
            media_type="video/mp4"
        )

    @app.get("/task/{task_id}")
    async def get_task_status(
        task_id: str,
        executor: TaskExecutor = Depends(get_task_executor)
    ):
        """Get the status of a task."""
        task_status = executor.get_task_status(task_id)
        
        # Convert task status to response format
        state = task_status.get("state", "UNKNOWN").upper()
        result = task_status.get("result")
        
        if state == "PENDING":
            response = {
                "status": "pending",
                "info": "Task is pending"
            }
        elif state == "STARTED":
            response = {
                "status": "started",
                "info": str(result) if result else "Task is running"
            }
        elif state == "SUCCESS":
            response = {
                "status": "success",
                "result": result
            }
            # If the result has its own status field, use it
            if isinstance(result, dict) and "status" in result:
                response["status"] = result["status"]
        elif state == "FAILURE":
            response = {
                "status": "failure",
                "error": str(result) if result else "Task failed"
            }
        else:
            response = {
                "status": state.lower(),
                "info": str(result) if result else "Unknown status"
            }
        
        return response

    return app

# Create default app instance
app = create_app()

def main():
    """Entry point for the web application."""
    uvicorn.run("web_app.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
