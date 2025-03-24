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
from typing import Optional, Dict, Any, Callable, Protocol, List

# Third-party imports
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, Cookie, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import status

# Local imports
from beat_detection.utils.constants import AUDIO_EXTENSIONS
from web_app.celery_app import app as celery_app
from web_app.storage import MetadataStorage, FileMetadataStorage
from web_app.tasks import detect_beats_task, generate_video_task
from web_app.auth import UserManager
from web_app.config import get_config, get_users

# Constants for task states
ANALYZING = "ANALYZING"
ANALYZED = "ANALYZED"
ANALYZING_FAILURE = "ANALYZING_FAILURE"
GENERATING_VIDEO = "GENERATING_VIDEO"
COMPLETED = "COMPLETED"
VIDEO_ERROR = "VIDEO_ERROR"
ERROR = "ERROR"

# Number of files to show in processing queue - from config
config = get_config()
MAX_QUEUE_FILES = config.get("queue", {}).get("max_files", 50)

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
    task_provider: Optional[TaskServiceProvider] = None,
    user_manager: Optional[UserManager] = None,
) -> FastAPI:
    """Create and configure the FastAPI application.
    
    Args:
        metadata_storage: Optional storage implementation
        task_provider: Optional task service implementation
        user_manager: Optional user manager implementation
        
    Returns:
        FastAPI application instance
    """
    # Initialize app
    app = FastAPI()
    
    # Initialize services
    storage = metadata_storage or FileMetadataStorage(base_dir=str(UPLOAD_DIR))
    service = task_provider or task_service
    auth = user_manager or UserManager()
    
    # Mount static files directory
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    
    # Initialize templates
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    
    # Dependency for getting storage
    def get_storage():
        return storage

    # Dependency for getting task service provider
    def get_task_service():
        return service
        
    # Dependency for getting auth manager
    def get_auth_manager():
        return auth
        
    # Update auth dependencies to use app-specific auth manager
    async def get_current_user_from_cookie(request: Request) -> Optional[Dict[str, Any]]:
        """Get the current user from the session cookie."""
        return auth.get_current_user(request)
        
    async def require_auth(request: Request) -> Dict[str, Any]:
        """Dependency that requires a valid authentication."""
        user = auth.get_current_user(request)
        
        if not user:
            # Check if this is an AJAX request
            is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
            
            if is_ajax:
                # For AJAX requests, always return 401 Unauthorized
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated"
                )
            else:
                # For web requests, redirect to login page
                redirect_url = f"/login?next={request.url.path}"
                raise HTTPException(
                    status_code=status.HTTP_307_TEMPORARY_REDIRECT,
                    headers={"Location": redirect_url},
                    detail="Not authenticated"
                )
        
        return user

    @app.get("/", response_class=HTMLResponse)
    async def index(
        request: Request,
        user: Dict[str, Any] = Depends(get_current_user_from_cookie)
    ):
        """Render the main page."""
        # Redirect to login if not authenticated
        if not user:
            return RedirectResponse(url="/login", status_code=303)
            
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "user": user}
        )

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(
        request: Request,
        next: Optional[str] = None
    ):
        """Render the login page."""
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "next_url": next or "/"}
        )

    @app.post("/login", response_class=HTMLResponse)
    async def login(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
        next_url: Optional[str] = Form(None)
    ):
        """Process login form submission."""
        # Create user manager
        user_manager = UserManager()
        
        # Authenticate user
        user = user_manager.authenticate(username, password)
        
        if not user:
            # Authentication failed
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "Invalid username or password", "next": next_url or "/"}
            )
        
        # Authentication successful
        # Create JWT token
        token_data = {
            "sub": user["username"],
            "is_admin": user.get("is_admin", False)
        }
        access_token = user_manager.create_access_token(token_data)
        
        # Create the response with a redirect
        response = RedirectResponse(url=next_url or "/", status_code=status.HTTP_303_SEE_OTHER)
        
        # Set the cookie
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            max_age=86400  # 24 hours
        )
        
        return response

    @app.get("/logout")
    async def logout(response: Response):
        """Log out the current user."""
        # Clear the cookie
        response = RedirectResponse(url="/login", status_code=303)
        response.delete_cookie(key="access_token", path="/")
        
        return response

    @app.post("/upload")
    async def upload_audio(
        request: Request,
        file: UploadFile = File(...),
        analyze: Optional[bool] = Form(False),
        generate_video: Optional[bool] = Form(False),
        storage: MetadataStorage = Depends(get_storage),
        service: TaskServiceProvider = Depends(get_task_service),
        user: Dict[str, Any] = Depends(require_auth)
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
        
        # Get user's IP address
        client_host = request.client.host if request.client else "unknown"
        
        # Save the uploaded file using the storage, which also saves the basic metadata
        audio_file_path = storage.save_audio_file(file_id, file_extension, file.file, filename=filename)
        
        # Start beat detection task directly with file_id
        task = service.detect_beats(file_id)
        
        # Update metadata with task ID, user IP, and username
        task_metadata = {
            "beat_detection": task.id,
            "user_ip": client_host,
            "upload_timestamp": datetime.now().isoformat(),
            "uploaded_by": user["username"]
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
        request: Request,
        storage: MetadataStorage = Depends(get_storage),
        task_service: TaskServiceProvider = Depends(get_task_service),
        user: Dict[str, Any] = Depends(require_auth)
    ):
        """Get status of a specific file."""
        # User authentication is now required via the require_auth dependency
        
        # Fetch metadata from storage
        metadata = await storage.get_file_metadata(file_id)
        
        if not metadata:
            if "X-Requested-With" in request.headers and request.headers["X-Requested-With"] == "XMLHttpRequest":
                # For AJAX requests, return 404 with JSON
                raise HTTPException(status_code=404, detail="File not found")
            else:
                # For browser requests, render a nice 404 page
                return templates.TemplateResponse(
                    "404.html",
                    {"request": request, "message": "The requested file was not found"},
                    status_code=404
                )
        
        # Initialize response data with basic file info
        response_data = {
            "file_id": file_id,
            "original_filename": metadata.get("original_filename"),
            "upload_timestamp": metadata.get("upload_timestamp"),
            "user_ip": metadata.get("user_ip"),
            "uploaded_by": metadata.get("uploaded_by"),
            "status": ERROR  # Default to ERROR
        }
        
        # Fetch task statuses if task IDs are present in metadata
        beat_detection_task_id = metadata.get("beat_detection")
        video_generation_task_id = metadata.get("video_generation")
        
        # Process beat detection task
        if beat_detection_task_id:
            beat_detection_status = task_service.get_task_status(beat_detection_task_id)
            response_data["beat_detection_task"] = beat_detection_status
            
            # Include beat statistics if available
            if "beat_stats" in metadata:
                response_data["beat_stats"] = metadata["beat_stats"]
        
        # Process video generation task
        if video_generation_task_id:
            video_generation_status = task_service.get_task_status(video_generation_task_id)
            response_data["video_generation_task"] = video_generation_status
        
        # Add file paths if they exist
        if "beats_file" in metadata:
            response_data["beats_file"] = metadata["beats_file"]
        
        if "video_file" in metadata:
            response_data["video_file"] = metadata["video_file"]
        
        # Determine overall status
        if beat_detection_task_id and video_generation_task_id:
            beat_state = response_data["beat_detection_task"]["state"]
            video_state = response_data["video_generation_task"]["state"]
            
            if beat_state == "SUCCESS" and video_state == "SUCCESS":
                response_data["status"] = COMPLETED
            elif beat_state == "SUCCESS" and video_state == "FAILURE":
                response_data["status"] = VIDEO_ERROR
            elif beat_state == "SUCCESS" and video_state in ["PENDING", "STARTED", "PROGRESS"]:
                response_data["status"] = GENERATING_VIDEO
            elif beat_state == "FAILURE":
                response_data["status"] = ANALYZING_FAILURE
            else:
                response_data["status"] = ERROR
        elif beat_detection_task_id:
            beat_state = response_data["beat_detection_task"]["state"]
            
            if beat_state == "SUCCESS":
                response_data["status"] = ANALYZED
            elif beat_state in ["PENDING", "STARTED", "PROGRESS"]:
                response_data["status"] = ANALYZING
            elif beat_state == "FAILURE":
                response_data["status"] = ANALYZING_FAILURE
            else:
                response_data["status"] = ERROR
        
        return response_data

    @app.get("/processing_queue", response_class=HTMLResponse)
    async def get_processing_queue(
        request: Request,
        storage: MetadataStorage = Depends(get_storage),
        service: TaskServiceProvider = Depends(get_task_service),
        user: Dict[str, Any] = Depends(require_auth)
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
            filename = file_metadata.get("original_filename", "Unknown file")
            
            # Get task statuses
            beat_task_id = file_metadata.get("beat_detection")
            video_task_id = file_metadata.get("video_generation")
            
            # Get upload timestamp (default to current time if not available)
            upload_time = file_info.get("upload_timestamp", datetime.now().isoformat())
            
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
                "link": f"/file/{file_id}",
                "upload_time": upload_time,
                "uploaded_by": file_info.get("uploaded_by", "Unknown")
            })
        
        # Sort files by upload time (newest first)
        files_with_status.sort(key=lambda x: x.get("upload_time", ""), reverse=True)
        
        # Limit to MAX_QUEUE_FILES most recent files
        files_with_status = files_with_status[:MAX_QUEUE_FILES]
        
        # Render the template with the file list
        return templates.TemplateResponse(
            "processing_queue.html", 
            {"request": request, "files": files_with_status, "user": user}
        )

    @app.post("/confirm/{file_id}")
    async def confirm_analysis(
        file_id: str, 
        storage: MetadataStorage = Depends(get_storage),
        service: TaskServiceProvider = Depends(get_task_service),
        user: Dict[str, Any] = Depends(require_auth)
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
        service: TaskServiceProvider = Depends(get_task_service),
        user: Dict[str, Any] = Depends(require_auth)
    ):
        """Render the file view page."""
        # Get file metadata
        metadata = await storage.get_file_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")
            
        # Calculate full file status for the template
        file_status = await get_file_status(file_id, request, storage, service)
        
        # Just add the task IDs directly to the template data
        # The frontend will poll for task status directly from the /task endpoint
        template_data = {
            "request": request,
            "file_id": file_id,
            "file_status": file_status,
            "user": user
        }
        
        return templates.TemplateResponse("file_view.html", template_data)

    @app.get("/download/{file_id}")
    async def download_video(
        file_id: str,
        storage: MetadataStorage = Depends(get_storage),
        user: Dict[str, Any] = Depends(require_auth)
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
        service: TaskServiceProvider = Depends(get_task_service),
        user: Dict[str, Any] = Depends(require_auth)
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
