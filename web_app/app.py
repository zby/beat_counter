#!/usr/bin/env python3
"""Beat Detection Web Application.

This web application allows users to upload audio files, analyze them for beats,
and generate visualization videos marking each beat.
"""

# Standard library imports
import logging
import pathlib
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

# Third-party imports
import uvicorn
from fastapi import (
    Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, Cookie,
    Response
)
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, RedirectResponse
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import status
from celery.result import AsyncResult

# Local imports
from web_app.config import Config
from web_app.storage import FileMetadataStorage
from web_app.auth import UserManager
# Import tasks directly from celery_app
from web_app.celery_app import app as celery_app, detect_beats_task, generate_video_task

# Constants for task states (consider moving to a shared constants module if used elsewhere)
ANALYZING = "ANALYZING"
ANALYZED = "ANALYZED"
ANALYZING_FAILURE = "ANALYZING_FAILURE"
GENERATING_VIDEO = "GENERATING_VIDEO"
COMPLETED = "COMPLETED"
VIDEO_ERROR = "VIDEO_ERROR"
ERROR = "ERROR" # General error state

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialization ---

# Load configuration using the new Config class
try:
    config = Config.from_env()
except FileNotFoundError as e:
    logger.error(f"Configuration error: {e}")
    # Fail fast during startup if config is missing
    raise SystemExit(f"Error: Configuration file not found. {e}") from e

# Get base directory
BASE_DIR = pathlib.Path(__file__).parent.absolute()

# --- FastAPI App Creation & Dependency Injection ---

def create_app(
    app_config: Optional[Config] = None,
    storage_impl: Optional[FileMetadataStorage] = None,
    auth_manager_impl: Optional[UserManager] = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        app_config: Optional configuration object.
        storage_impl: Optional storage implementation for dependency injection.
        auth_manager_impl: Optional user manager implementation for DI.

    Returns:
        FastAPI application instance.
    """
    # Use provided or default config
    current_config = app_config or config

    # Initialize services with explicit config
    storage = storage_impl or FileMetadataStorage(current_config.storage)
    auth_manager = auth_manager_impl or UserManager(users={"users": [user.__dict__ for user in current_config.users]})

    # Create FastAPI app instance
    app = FastAPI(debug=current_config.app.debug)

    # Mount static files directory
    static_dir = BASE_DIR / "static"
    if not static_dir.is_dir():
         logger.warning(f"Static directory not found at {static_dir}, creating it.")
         static_dir.mkdir(parents=True, exist_ok=True) # Attempt to create if missing
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Initialize templates
    templates_dir = BASE_DIR / "templates"
    if not templates_dir.is_dir():
         # Fail fast if templates are essential and missing
         raise SystemExit(f"Error: Templates directory not found at {templates_dir}")
    templates = Jinja2Templates(directory=str(templates_dir))

    # --- Dependencies ---
    def get_storage() -> FileMetadataStorage:
        """Dependency to get the storage instance."""
        return storage

    def get_auth_manager() -> UserManager:
        """Dependency to get the authentication manager instance."""
        return auth_manager

    async def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
        """Dependency to get the current user from the session cookie."""
        return auth_manager.get_current_user(request)

    async def require_auth(request: Request) -> Dict[str, Any]:
        """Dependency that requires valid authentication. Fails fast."""
        user = await get_current_user(request)
        if not user:
            # Check if this is an AJAX request
            is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

            if is_ajax:
                # For AJAX requests, return 401 (Fail Fast for API)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated"
                )
            else:
                # For web requests, redirect to login (User-friendly failure)
                redirect_url = request.url_for('login_page').include_query_params(next=request.url.path)
                # Use 303 See Other for POST->GET redirect pattern if needed,
                # but 307 keeps the method for general redirection.
                # Let's use 303 as it's likely after a failed action.
                raise HTTPException(
                    status_code=status.HTTP_303_SEE_OTHER,
                    headers={"Location": str(redirect_url)},
                    detail="Not authenticated"
                )
        return user

    # --- Helper Function ---
    def get_task_status_direct(task_id: str) -> Dict[str, Any]:
        """Gets Celery task status directly without the TaskServiceProvider abstraction."""
        # No try/except here - let errors surface (Fail Fast)
        # Errors like connection issues should be handled by Celery/infrastructure
        async_result = AsyncResult(task_id, app=celery_app)
        response = {"id": task_id, "state": async_result.state}
        if async_result.ready():
            if async_result.successful():
                # Only include result if successful and result exists
                if async_result.result is not None:
                     response["result"] = async_result.result
            else:
                # Include error info for failures
                 try:
                     # Attempt to get traceback for more context
                     response["error"] = str(async_result.traceback)
                 except Exception:
                     # Fallback to result which might contain error message
                     response["error"] = str(async_result.result)
        elif async_result.state == 'PROGRESS' or async_result.state == 'STARTED':
             # Include task metadata (info) if available during progress
             if isinstance(async_result.info, dict):
                  response['progress'] = async_result.info.get('progress') # Only get progress part

        return response


    # --- Routes ---

    @app.get("/", response_class=HTMLResponse, name="index")
    async def index_route(
        request: Request,
        user: Optional[Dict[str, Any]] = Depends(get_current_user)
    ):
        """Render the main page (upload form)."""
        if not user:
            # Redirect to login if not authenticated
            login_url = request.url_for('login_page')
            return RedirectResponse(url=str(login_url), status_code=status.HTTP_303_SEE_OTHER)

        return templates.TemplateResponse(
            request,
            "index.html",
            {"user": user}
        )

    @app.get("/login", response_class=HTMLResponse, name="login_page")
    async def login_page_route(
        request: Request,
        next: Optional[str] = None
    ):
        """Render the login page."""
        return templates.TemplateResponse(
            request,
            "login.html",
            {"next_url": next or request.url_for('index')}
        )

    @app.post("/login", response_class=RedirectResponse, name="login_action")
    async def login_action_route(
        request: Request,
        response: Response, # Inject response to set cookie
        username: str = Form(...),
        password: str = Form(...),
        next_url: Optional[str] = Form(None),
        auth: UserManager = Depends(get_auth_manager)
    ):
        """Process login form submission."""
        user = auth.authenticate(username, password)

        if not user:
            # Authentication failed - re-render login with error
            # Use flash messages or query params for errors in real apps
            # For simplicity, redirecting back with an error flag (not ideal)
            login_url = request.url_for('login_page').include_query_params(error="invalid", next=next_url or '/')
            # Don't raise HTTPException, redirect back to form
            return RedirectResponse(url=str(login_url), status_code=status.HTTP_303_SEE_OTHER)


        # Authentication successful - Create JWT token
        token_data = {
            "sub": user["username"],
            "is_admin": user.get("is_admin", False) # Ensure is_admin is included
        }
        access_token = auth.create_access_token(token_data)

        # Set cookie in the response
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True, # Important for security
            samesite="Lax", # Good practice
            secure=request.url.scheme == "https", # Set Secure flag if using HTTPS
            max_age=int(timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES).total_seconds()) # Use configured expiration
        )

        # Redirect to the intended page or home
        final_redirect_url = next_url or request.url_for('index')
        # Use status 303 See Other after successful POST
        response.status_code = status.HTTP_303_SEE_OTHER
        response.headers["Location"] = str(final_redirect_url)
        return response # Return the response object directly

    @app.get("/logout", response_class=RedirectResponse, name="logout")
    async def logout_route(request: Request, response: Response):
        """Log out the current user by clearing the cookie."""
        response.delete_cookie(key="access_token", path="/")
        # Redirect to login page after logout
        login_url = request.url_for('login_page')
        response.status_code = status.HTTP_303_SEE_OTHER
        response.headers["Location"] = str(login_url)
        return response # Return the response object


    @app.post("/upload", name="upload_audio")
    async def upload_audio_route(
        request: Request,
        file: UploadFile = File(...),
        storage: FileMetadataStorage = Depends(get_storage),
        user: Dict[str, Any] = Depends(require_auth) # Require authentication
    ):
        """Upload an audio file, save it, and start beat detection task."""
        filename = file.filename
        file_extension = pathlib.Path(filename).suffix.lower()

        # Validate file extension against allowed types from config
        if file_extension not in current_config.storage.allowed_extensions:
            supported_formats = ", ".join(current_config.storage.allowed_extensions)
            # Fail Fast: Use specific HTTP error for invalid input
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file format. Supported formats: {supported_formats}"
            )

        # Generate a unique ID for this upload
        file_id = str(uuid.uuid4())

        # Save the uploaded file using the storage service.
        # Let storage handle truncation and potential file saving errors (Fail Fast)
        # No try/except here - if saving fails, FastAPI returns 500.
        audio_file_path = storage.save_audio_file(
            file_id,
            file_extension,
            file.file, # Pass the file-like object directly
            filename=filename
        )

        # Start beat detection task using the imported task function
        # Let Celery handle task queueing errors (Fail Fast principle for infra issues)
        task = detect_beats_task.delay(file_id)
        if not task or not task.id:
             # If task queuing fails immediately (e.g., broker down), raise 500
             raise HTTPException(status_code=500, detail="Failed to queue beat detection task.")


        # Update metadata: User info, task ID. Let storage handle errors.
        storage.update_metadata(file_id, {
            "beat_detection": task.id,
            "user_ip": request.client.host if request.client else "unknown",
            "upload_timestamp": datetime.now().isoformat(),
            "uploaded_by": user["username"] # Get username from authenticated user
        })

        # Get truncation info from metadata after saving
        metadata = storage.get_metadata(file_id) # Read back the potentially updated metadata
        truncation_info = {
             # Use config for limit, fallback if somehow missing in saved metadata
            "duration_limit": metadata.get("duration_limit", current_config.storage.max_audio_secs),
            "original_duration": metadata.get("original_duration") # Will be None if not saved
        }


        # Respond differently for AJAX vs. form submission
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        if is_ajax:
            # Return JSON for AJAX clients
            return JSONResponse(content={
                "file_id": file_id,
                 "task_id": task.id,
                 "truncation_info": truncation_info
            })
        else:
            # Redirect to the file view page for form submissions
            file_view_url = request.url_for('file_page_route', file_id=file_id)
            # Use 303 See Other to redirect after POST
            return RedirectResponse(url=str(file_view_url), status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/status/{file_id}", name="get_file_status")
    async def get_file_status_route(
        file_id: str,
        request: Request, # Keep request for AJAX check if needed later
        storage: FileMetadataStorage = Depends(get_storage),
        user: Dict[str, Any] = Depends(require_auth) # Require authentication
    ):
        """Get the processing status of a specific file, including task states."""
        metadata = storage.get_file_metadata(file_id)
        if not metadata:
            # Fail Fast: File not found is a client error (404)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

        # Start building response data from metadata
        response_data = {
            "file_id": file_id,
            "original_filename": metadata.get("original_filename"),
            "upload_timestamp": metadata.get("upload_timestamp"),
            "uploaded_by": metadata.get("uploaded_by"),
            "user_ip": metadata.get("user_ip"), # Include IP if stored
            "duration_limit": metadata.get("duration_limit", current_config.storage.max_audio_secs),
            "original_duration": metadata.get("original_duration"),
            "beat_stats": metadata.get("beat_stats"), # Include if present
            "beats_file_exists": storage.get_beats_file_path(file_id).exists(),
            "video_file_exists": storage.get_video_file_path(file_id).exists(),
            "beat_detection_task": None,
            "video_generation_task": None,
            "status": ERROR, # Default status
        }

        # Fetch task statuses if IDs exist
        beat_task_id = metadata.get("beat_detection")
        video_task_id = metadata.get("video_generation")

        beat_state = None
        video_state = None

        if beat_task_id:
            response_data["beat_detection_task"] = get_task_status_direct(beat_task_id)
            beat_state = response_data["beat_detection_task"]["state"]
        if video_task_id:
            response_data["video_generation_task"] = get_task_status_direct(video_task_id)
            video_state = response_data["video_generation_task"]["state"]

        # Determine overall status (Simplified logic)
        if response_data["video_file_exists"]:
             response_data["status"] = COMPLETED
        elif video_state and video_state not in ["SUCCESS", "FAILURE"]:
             response_data["status"] = GENERATING_VIDEO
        elif video_state == "FAILURE":
             response_data["status"] = VIDEO_ERROR
        elif response_data["beats_file_exists"]:
             response_data["status"] = ANALYZED # Ready for confirmation
        elif beat_state and beat_state not in ["SUCCESS", "FAILURE"]:
             response_data["status"] = ANALYZING
        elif beat_state == "FAILURE":
             response_data["status"] = ANALYZING_FAILURE
        # else remains ERROR (e.g., no tasks started, or unexpected state)


        return response_data # Return directly as JSON

    @app.get("/processing_queue", response_class=HTMLResponse, name="processing_queue")
    async def get_processing_queue_route(
        request: Request,
        storage: FileMetadataStorage = Depends(get_storage),
        user: Dict[str, Any] = Depends(require_auth) # Require authentication
    ):
        """Display a list of recently processed files."""
        all_metadata = storage.get_all_metadata() # Get basic metadata for all jobs

        files_with_status = []
        # Sort by job directory creation time (approx upload time), newest first
        # This avoids loading full status for every file just for sorting
        sorted_file_ids = sorted(
            all_metadata.keys(),
            key=lambda fid: storage.get_job_directory_creation_time(fid),
            reverse=True
        )

        # Limit the number of files to display based on config
        for file_id in sorted_file_ids[:current_config.app.max_queue_files]:
            try:
                 # Fetch detailed status only for the files being displayed
                file_status_data = await get_file_status_route(file_id, request, storage, user)
                files_with_status.append({
                    "file_id": file_id,
                    # Use get() with fallback for robustness
                    "filename": file_status_data.get("original_filename", "Unknown"),
                    "status": file_status_data.get("status", ERROR).lower(),
                    "link": request.url_for('file_page_route', file_id=file_id),
                    "upload_time": file_status_data.get("upload_timestamp", "Unknown"),
                     "uploaded_by": file_status_data.get("uploaded_by", "Unknown")
                })
            except HTTPException as e:
                 # Log if fetching status for a file fails (e.g., 404) but continue
                 logger.warning(f"Could not retrieve status for file {file_id} in queue: {e.detail}")
            except Exception as e:
                 logger.error(f"Unexpected error retrieving status for file {file_id} in queue: {e}", exc_info=True)


        return templates.TemplateResponse(
            request,
            "processing_queue.html",
            {"files": files_with_status, "user": user}
        )

    @app.post("/confirm/{file_id}", name="confirm_analysis")
    async def confirm_analysis_route(
        file_id: str,
        request: Request, # Keep request for potential future use
        storage: FileMetadataStorage = Depends(get_storage),
        user: Dict[str, Any] = Depends(require_auth) # Require authentication
    ):
        """Confirm successful analysis and start video generation task."""
        # Fetch current detailed status to check readiness
        try:
             current_status = await get_file_status_route(file_id, request, storage, user)
        except HTTPException as e:
             # Re-raise if file not found etc.
             raise e

        # Check if beat detection was successful and beats file exists
        # Rely on the overall status derived in get_file_status_route
        if current_status.get("status") != ANALYZED:
             logger.warning(f"Confirmation attempt for file {file_id} failed. Status was {current_status.get('status')}, expected {ANALYZED}.")
             # Fail Fast: Client error if preconditions not met
             raise HTTPException(
                 status_code=status.HTTP_400_BAD_REQUEST,
                 detail=f"File not ready for confirmation. Status: {current_status.get('status')}. Beat detection must be completed successfully."
             )

        # Start video generation task
        # Let Celery/infra handle queueing errors
        task = generate_video_task.delay(file_id)
        if not task or not task.id:
             # If task queuing fails immediately, raise 500
             raise HTTPException(status_code=500, detail="Failed to queue video generation task.")


        # Update metadata with video generation task ID. Let storage handle errors.
        storage.update_metadata(file_id, {"video_generation": task.id})

        return JSONResponse(content={
            "status": "ok",
            "message": "Video generation initiated",
            "task_id": task.id
        })

    @app.get("/file/{file_id}", response_class=HTMLResponse, name="file_page_route")
    async def file_page_route(
        request: Request,
        file_id: str,
        storage: FileMetadataStorage = Depends(get_storage), # Inject storage directly
        user: Dict[str, Any] = Depends(require_auth) # Require authentication
    ):
        """Render the file view page, showing current status and results."""
        # Fetch the latest status first. This also handles the 404 case.
        try:
            file_status_data = await get_file_status_route(file_id, request, storage, user)
        except HTTPException as e:
             if e.status_code == 404:
                  return templates.TemplateResponse(
                       request,
                       "404.html",
                       {"message": f"File with ID {file_id} not found."},
                       status_code=404
                  )
             else:
                  # Re-raise other HTTP exceptions
                  raise e
        except Exception as e:
             logger.error(f"Error getting status for file page {file_id}: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail="Error retrieving file status.")

        # Pass the fetched status data directly to the template
        return templates.TemplateResponse(
            request,
            "file_view.html",
            {
                "request": request, # Template needs request context
                "file_id": file_id,
                "file_status": file_status_data, # Pass the whole status dict
                "user": user,
                # Pass app_dir if needed by template (though maybe less relevant now)
                "app_dir": str(BASE_DIR)
            }
        )

    @app.get("/download/{file_id}", name="download_video")
    async def download_video_route(
        file_id: str,
        storage: FileMetadataStorage = Depends(get_storage),
        user: Dict[str, Any] = Depends(require_auth) # Require authentication
    ):
        """Download the generated video file."""
        # Get video file path using storage service
        video_path = storage.get_video_file_path(file_id)

        if not video_path.exists():
             # Fail Fast: File not found is a client error (404)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video file not found or not generated yet."
            )

        # Get original filename from metadata to create a nice download name
        metadata = storage.get_metadata(file_id) # Basic metadata is enough here
        original_filename = metadata.get("original_filename", f"audio_{file_id}")
        base_name = pathlib.Path(original_filename).stem
        download_name = f"{base_name}_with_beats.mp4"

        # Serve the file using FileResponse
        # Let FileResponse handle potential read errors (maps to 500)
        return FileResponse(
            path=str(video_path),
            filename=download_name,
            media_type="video/mp4"
        )

    @app.get("/task/{task_id}", name="get_task_status_endpoint")
    async def get_task_status_endpoint_route(
        task_id: str,
        user: Dict[str, Any] = Depends(require_auth) # Require authentication
    ):
        """API endpoint to get the raw status of a Celery task."""
        # Use the direct helper function
        # Let errors from get_task_status_direct propagate (Fail Fast for infra issues)
        task_status = get_task_status_direct(task_id)
        return task_status # Return directly as JSON

    return app

# --- Main Application Instance & Entry Point ---

# Create the default app instance using the global config
app = create_app(app_config=config)

def main():
    """Entry point for running the web application directly."""
    # Use host/port from config if available, otherwise default
    # Note: Uvicorn args override config settings if both are present.
    # Prefer command-line args or environment variables for host/port in production.
    host = "0.0.0.0"
    port = 8000
    reload = config.app.debug # Enable reload only if app debug is true

    logger.info(f"Starting FastAPI application on {host}:{port} (Reload: {reload})")
    uvicorn.run(
        "web_app.app:app", # Point to the app instance
        host=host,
        port=port,
        reload=reload,
        # Consider adding log_level based on config.app.debug
        log_level="debug" if reload else "info"
    )

if __name__ == "__main__":
    main()
