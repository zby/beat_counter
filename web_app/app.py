#!/usr/bin/env python3
"""Beat Detection Web Application.

Refactored to use APIRouter for route organization and dependency injection
for configuration and templates.
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
    APIRouter,  # <-- Import APIRouter
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    Cookie,
    Response,
    status,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates  # <-- Keep Jinja2Templates import
from celery.result import AsyncResult
from celery import states

# Local imports
from web_app.config import Config
from web_app.storage import FileMetadataStorage
from web_app.auth import UserManager

# Import tasks directly from celery_app
from web_app.celery_app import app as celery_app, detect_beats_task, generate_video_task

# Constants for task states
ANALYZING = "ANALYZING"
ANALYZED = "ANALYZED"
ANALYZING_FAILURE = "ANALYZING_FAILURE"
GENERATING_VIDEO = "GENERATING_VIDEO"
COMPLETED = "COMPLETED"
VIDEO_ERROR = "VIDEO_ERROR"
ERROR = "ERROR"  # General error state
UPLOADED = "UPLOADED"  # Added for the new status logic

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Configuration Loading ---
try:
    _global_config = Config.from_env()
except FileNotFoundError as e:
    logger.error(f"Configuration error: {e}")
    raise SystemExit(f"Error: Configuration file not found. {e}") from e

# --- Global Initialization ---
BASE_DIR = pathlib.Path(__file__).parent.absolute()

# Initialize Templates globally
TEMPLATES_DIR = BASE_DIR / "templates"
if not TEMPLATES_DIR.is_dir():
    raise SystemExit(f"Error: Templates directory not found at {TEMPLATES_DIR}")
_global_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))  # <-- Initialize here

# Application Context Holder
_app_context: Dict[str, Any] = {}

# --- Dependency Functions ---


def get_config() -> Config:
    """Dependency function to get the application configuration."""
    return _global_config


def get_templates() -> Jinja2Templates:  # <-- New dependency function
    """Dependency function to get the Jinja2Templates instance."""
    return _global_templates


def get_storage() -> FileMetadataStorage:
    """Dependency to get the storage instance."""
    storage = _app_context.get("storage")
    if not storage:
        raise RuntimeError("Storage service not initialized")
    return storage


def get_auth_manager() -> UserManager:
    """Dependency to get the authentication manager instance."""
    auth_manager = _app_context.get("auth_manager")
    if not auth_manager:
        raise RuntimeError("Authentication manager not initialized")
    return auth_manager


async def get_current_user(
    request: Request, auth: UserManager = Depends(get_auth_manager)
) -> Optional[Dict[str, Any]]:
    """Dependency to get the current user from the session cookie."""
    return auth.get_current_user(request)


async def require_auth(
    request: Request, user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Dependency that requires valid authentication. Fails fast."""
    if not user:
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        if is_ajax:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
            )
        else:
            redirect_url = request.url_for("login_page").include_query_params(
                next=request.url.path
            )
            raise HTTPException(
                status_code=status.HTTP_303_SEE_OTHER,
                headers={"Location": str(redirect_url)},
                detail="Not authenticated",
            )
    return user


def get_task_status_direct(task_id: str) -> Dict[str, Any]:
    """Helper: Gets Celery task status directly."""
    # Get the async result object
    async_result = AsyncResult(task_id, app=celery_app)
    response = {"id": task_id, "state": async_result.state}

    # Debug logging to help with test troubleshooting
    logger.info(
        f"DEBUG [get_task_status_direct]: Task {task_id} state: {async_result.state}"
    )
    if hasattr(async_result, "status"):
        logger.info(
            f"DEBUG [get_task_status_direct]: Task {task_id} status attr: {async_result.status}"
        )
    if hasattr(async_result, "info"):
        logger.info(
            f"DEBUG [get_task_status_direct]: Task {task_id} info: {async_result.info}"
        )

    # Handle completed tasks
    if async_result.ready():
        if async_result.successful():
            if async_result.result is not None:
                response["result"] = async_result.result
        else:
            try:
                response["error"] = str(async_result.traceback)
            except Exception:
                response["error"] = str(async_result.result)
    # Handle in-progress tasks
    elif (
        async_result.state == "PROGRESS"
        or getattr(async_result, "status", None) == "PROGRESS"
        or async_result.state == "STARTED"
    ):
        # Check if we have progress info in the task
        if isinstance(async_result.info, dict):
            # Look for a nested 'progress' key first
            progress = async_result.info.get("progress")
            if progress and isinstance(progress, dict):
                logger.info(
                    f"DEBUG [get_task_status_direct]: Found nested progress: {progress}"
                )
                response["progress"] = progress
            # If no nested progress, check if info itself is a progress dict
            elif "percent" in async_result.info or "status" in async_result.info:
                logger.info(
                    f"DEBUG [get_task_status_direct]: Using info directly as progress: {async_result.info}"
                )
                response["progress"] = async_result.info

    # Log the final response being returned
    logger.info(f"DEBUG [get_task_status_direct]: Returning response: {response}")
    return response


# --- Routers ---
main_router = APIRouter()
auth_router = APIRouter(tags=["Authentication"])

# --- Authentication Routes (on auth_router) ---


@auth_router.get("/login", response_class=HTMLResponse, name="login_page")
async def login_page_route(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),  # <-- Inject templates
    next: Optional[str] = None,
):
    """Render the login page."""
    # Use injected templates
    return templates.TemplateResponse(
        request, "login.html", {"next_url": next or request.url_for("index")}
    )


@auth_router.post("/login", response_class=RedirectResponse, name="login_action")
async def login_action_route(
    request: Request,
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
    next_url: Optional[str] = Form(None),
    auth: UserManager = Depends(get_auth_manager),
):
    """Process login form submission."""
    # (Keep existing logic)
    user = auth.authenticate(username, password)
    if not user:
        login_url = request.url_for("login_page").include_query_params(
            error="invalid", next=next_url or "/"
        )
        return RedirectResponse(
            url=str(login_url), status_code=status.HTTP_303_SEE_OTHER
        )

    token_data = {"sub": user["username"], "is_admin": user.get("is_admin", False)}
    access_token = auth.create_access_token(token_data)

    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        samesite="Lax",
        secure=request.url.scheme == "https",
        max_age=int(
            timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES).total_seconds()
        ),
    )

    final_redirect_url = next_url or request.url_for("index")
    response.status_code = status.HTTP_303_SEE_OTHER
    response.headers["Location"] = str(final_redirect_url)
    return response


@auth_router.get("/logout", response_class=RedirectResponse, name="logout")
async def logout_route(request: Request, response: Response):
    """Log out the current user by clearing the cookie."""
    # (Keep existing logic)
    response.delete_cookie(key="access_token", path="/")
    login_url = request.url_for("login_page")
    response.status_code = status.HTTP_303_SEE_OTHER
    response.headers["Location"] = str(login_url)
    return response


# --- Main Application Routes (on main_router) ---


@main_router.get("/", response_class=HTMLResponse, name="index")
async def index_route(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),  # <-- Inject templates
    user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """Render the main page (upload form). Redirects to login if needed."""
    if not user:
        login_url = request.url_for("login_page")
        return RedirectResponse(
            url=str(login_url), status_code=status.HTTP_303_SEE_OTHER
        )
    # Use injected templates
    return templates.TemplateResponse(request, "index.html", {"user": user})


@main_router.post("/upload", name="upload_audio")
async def upload_audio_route(
    request: Request,
    file: UploadFile = File(...),
    algorithm: str = Form("madmom"),  # Default to madmom
    beats_per_bar: Optional[int] = Form(None),  # Optional override
    storage: FileMetadataStorage = Depends(get_storage),
    config: Config = Depends(get_config),
    user: Dict[str, Any] = Depends(require_auth),
):
    """Upload an audio file, save it, and start beat detection task."""
    # (Keep existing logic, uses injected config)
    filename = file.filename
    file_extension = pathlib.Path(filename).suffix.lower()

    if file_extension not in config.storage.allowed_extensions:
        supported_formats = ", ".join(config.storage.allowed_extensions)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format. Supported formats: {supported_formats}",
        )

    file_id = str(uuid.uuid4())
    try:
        audio_file_path = storage.save_audio_file(
            file_id, file_extension, file.file, filename=filename
        )
    except Exception as e:
        logger.error(f"Error saving file {file_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saving uploaded file: {e}")

    try:
        # Pass algorithm and beats_per_bar to the task
        task = detect_beats_task.delay(
            file_id,
            algorithm=algorithm,
            beats_per_bar=beats_per_bar if beats_per_bar else None
        )
        if not task or not task.id:
            raise RuntimeError("Failed to get task ID after submission.")
    except Exception as e:
        logger.error(
            f"Error queuing beat detection task for {file_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Failed to queue beat detection task."
        )

    try:
        storage.update_metadata(
            file_id,
            {
                "beat_detection": task.id,
                "user_ip": request.client.host if request.client else "unknown",
                "upload_timestamp": datetime.now().isoformat(),
                "uploaded_by": user["username"],
            },
        )
    except Exception as e:
        logger.error(
            f"Error updating metadata for {file_id} after task submission: {e}",
            exc_info=True,
        )

    metadata = storage.get_metadata(file_id)
    truncation_info = {
        "duration_limit": metadata.get("duration_limit", config.storage.max_audio_secs),
        "original_duration": metadata.get("original_duration"),
    }

    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
    if is_ajax:
        return JSONResponse(
            content={
                "file_id": file_id,
                "task_id": task.id,
                "truncation_info": truncation_info,
            }
        )
    else:
        file_view_url = request.url_for("file_page_route", file_id=file_id)
        return RedirectResponse(
            url=str(file_view_url), status_code=status.HTTP_303_SEE_OTHER
        )


@main_router.get("/status/{file_id}", name="get_file_status")
async def get_file_status_route(
    file_id: str,
    request: Request,  # Keep request for potential future use (e.g., user checks)
    storage: FileMetadataStorage = Depends(get_storage),
    # config: Config = Depends(get_config), # Config potentially not needed here anymore
    user: Dict[str, Any] = Depends(require_auth),
):
    """Get the processing status of a specific file based on metadata.json."""
    metadata = storage.get_metadata(file_id)
    logger.info(f"DEBUG [metadata]: {metadata}")
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="File metadata not found"
        )

    # Initialize response with basic info from metadata
    response_data = {
        "file_id": file_id,
        "original_filename": metadata.get("original_filename"),
        "audio_file_path": metadata.get("audio_file_path"),
        "user_ip": metadata.get("user_ip"),
        "upload_timestamp": metadata.get("upload_timestamp"),
        "uploaded_by": metadata.get("uploaded_by"),
        "original_duration": metadata.get("original_duration"),
        "duration": metadata.get("duration"),  # Add the truncated duration
        "duration_limit": metadata.get("duration_limit"),
        "beat_detection": metadata.get("beat_detection"),  # Beat Task ID
        "video_generation": metadata.get("video_generation"),  # Video Task ID
    }

    # --- Construct beat_stats from metadata fields --- #
    beat_status = metadata.get("beat_detection_status")
    if beat_status == "success":
        response_data["beat_stats"] = {
            "tempo_bpm": metadata.get("detected_tempo_bpm"),
            "total_beats": metadata.get("total_beats"),
            "detected_beats_per_bar": metadata.get("detected_beats_per_bar"),
            "irregularity_percent": metadata.get("irregularity_percent"),
            "irregular_beats_count": metadata.get("irregular_beats_count"),
            "clip_length": metadata.get("clip_length"),  # Add clip_length to beat_stats
            "status": beat_status,
            "error": None,
        }
    elif beat_status == "error":
        response_data["beat_stats"] = {
            "status": beat_status,
            "error": metadata.get("beat_detection_error"),
        }
        response_data["beat_error"] = metadata.get("beat_detection_error")
    else:
        response_data["beat_stats"] = None

    # --- Check File Existence --- #
    beats_file_path_str = metadata.get("beats_file")
    video_file_path_str = metadata.get("video_file")

    # --- Add temporary debugging for exists check --- #
    beats_exists = False
    if beats_file_path_str:
        beats_path_obj = pathlib.Path(beats_file_path_str)
        beats_exists = beats_path_obj.exists()
        logger.info(
            f"DEBUG [Status Route]: Checking existence for beats_file: '{beats_file_path_str}' -> Exists? {beats_exists}"
        )
        # --- Add explicit exception if exists() fails unexpectedly --- #
        if not beats_exists:
            # This should not happen in the test case after the explicit check/creation
            logger.error(
                f"CRITICAL [Status Route]: File path '{beats_file_path_str}' exists in metadata but pathlib.Path.exists() returned False!"
            )
            # Raise an error to make the test failure clear
            # raise RuntimeError(f"File existence check failed within status route for path: {beats_file_path_str}")
            # Let's just log the error for now, maybe raising hides other issues
            pass  # Continue but log the error
    else:
        logger.info("DEBUG [Status Route]: No beats_file path found in metadata.")
    response_data["beats_file_exists"] = beats_exists
    # --- End explicit check/debug block --- #

    response_data["video_file_exists"] = bool(
        video_file_path_str and pathlib.Path(video_file_path_str).exists()
    )

    # --- Determine Overall Status based on Metadata (Primary) and Celery Task State (Secondary) --- #
    beat_task_id = response_data["beat_detection"]
    video_task_id = response_data["video_generation"]
    video_metadata_status = metadata.get("video_generation_status")

    overall_status = UPLOADED  # Default initial state

    # --- Determine if we need to fetch active task progress information ---
    beat_task_progress = None
    video_task_progress = None

    if video_metadata_status == "success" and response_data["video_file_exists"]:
        overall_status = COMPLETED
    elif video_metadata_status == "error":
        overall_status = VIDEO_ERROR
        response_data["video_error"] = metadata.get("video_generation_error")
    elif video_task_id:
        # If video task exists but hasn't succeeded/failed according to metadata,
        # check its transient Celery state.
        try:
            video_task = AsyncResult(video_task_id, app=celery_app)
            video_task_state = video_task.state

            # Add task progress information to response
            task_info = get_task_status_direct(video_task_id)
            logger.info(f"DEBUG [get_file_status_route]: Video task info: {task_info}")
            video_task_progress = task_info.get("progress")

            if video_task_state in [
                states.PENDING,
                states.STARTED,
                states.RETRY,
                states.RECEIVED,
                "PROGRESS",
            ]:
                overall_status = GENERATING_VIDEO
            elif video_task_state == states.FAILURE:
                # Should ideally be caught by video_metadata_status == "error", but as fallback:
                overall_status = VIDEO_ERROR
                response_data["video_error"] = str(
                    video_task.traceback or "Video generation failed"
                )
                logger.warning(
                    f"Celery state for video task {video_task_id} is FAILURE, but metadata status is {video_metadata_status}"
                )
        except Exception as e:
            logger.error(f"Error checking video task status: {e}")

    # If not completed or generating video, check beat detection status
    if overall_status not in [COMPLETED, GENERATING_VIDEO, VIDEO_ERROR]:
        if beat_status == "success" and response_data["beats_file_exists"]:
            overall_status = ANALYZED  # Ready for confirmation
        elif beat_status == "error":
            overall_status = ANALYZING_FAILURE
        elif beat_task_id:
            # If beat task exists but hasn't succeeded/failed according to metadata,
            # check its transient Celery state.
            try:
                beat_task = AsyncResult(beat_task_id, app=celery_app)
                beat_task_state = beat_task.state

                # Add task progress information to response
                task_info = get_task_status_direct(beat_task_id)
                logger.info(
                    f"DEBUG [get_file_status_route]: Beat task info: {task_info}"
                )
                beat_task_progress = task_info.get("progress")

                if beat_task_state in [
                    states.PENDING,
                    states.STARTED,
                    states.RETRY,
                    states.RECEIVED,
                    "PROGRESS",
                ]:
                    overall_status = ANALYZING
                elif beat_task_state == states.FAILURE:
                    # Should ideally be caught by beat_status == "error", but as fallback:
                    overall_status = ANALYZING_FAILURE
                    response_data["beat_error"] = str(
                        beat_task.traceback or "Beat analysis failed"
                    )
                    logger.warning(
                        f"Celery state for beat task {beat_task_id} is FAILURE, but metadata status is {beat_status}"
                    )
            except Exception as e:
                logger.error(f"Error checking beat task status: {e}")

    response_data["status"] = overall_status

    # Add current task progress information if available
    if overall_status == ANALYZING and beat_task_progress:
        logger.info(
            f"DEBUG [get_file_status_route]: Adding beat task progress to task_progress: {beat_task_progress}"
        )
        response_data["task_progress"] = beat_task_progress
    elif overall_status == GENERATING_VIDEO and video_task_progress:
        logger.info(
            f"DEBUG [get_file_status_route]: Adding video task progress to task_progress: {video_task_progress}"
        )
        response_data["task_progress"] = video_task_progress

    # Also include raw task progress data for completeness
    response_data["beat_task_progress"] = beat_task_progress
    response_data["video_task_progress"] = video_task_progress

    return response_data


@main_router.get(
    "/processing_queue", response_class=HTMLResponse, name="processing_queue"
)
async def get_processing_queue_route(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),  # <-- Inject templates
    storage: FileMetadataStorage = Depends(get_storage),
    config: Config = Depends(get_config),
    user: Dict[str, Any] = Depends(require_auth),
):
    """Display a list of recently processed files."""
    # (Keep existing logic, uses injected config)
    all_metadata = storage.get_all_metadata()
    files_with_status = []
    sorted_file_ids = sorted(
        all_metadata.keys(),
        key=lambda fid: storage.get_job_directory_creation_time(fid),
        reverse=True,
    )

    for file_id in sorted_file_ids[: config.app.max_queue_files]:
        try:
            # Correctly call get_file_status_route, passing dependencies explicitly
            file_status_data = await get_file_status_route(
                file_id=file_id,
                request=request,  # Pass the request object
                storage=storage,  # Pass the storage dependency
                user=user,  # Pass the user dependency
            )
            files_with_status.append(
                {
                    "file_id": file_id,
                    "filename": file_status_data.get("original_filename", "Unknown"),
                    "status": file_status_data.get("status", ERROR).lower(),
                    "link": request.url_for("file_page_route", file_id=file_id),
                    "upload_time": file_status_data.get("upload_timestamp", "Unknown"),
                    "uploaded_by": file_status_data.get("uploaded_by", "Unknown"),
                }
            )
        except HTTPException as e:
            logger.warning(
                f"Could not retrieve status for file {file_id} in queue: {e.detail}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error retrieving status for file {file_id} in queue: {e}",
                exc_info=True,
            )

    # Use injected templates
    return templates.TemplateResponse(
        request, "processing_queue.html", {"files": files_with_status, "user": user}
    )


@main_router.post("/confirm/{file_id}", name="confirm_analysis")
async def confirm_analysis_route(
    file_id: str,
    request: Request,
    storage: FileMetadataStorage = Depends(get_storage),
    # config: Config = Depends(get_config), # Config no longer needed for this check
    user: Dict[str, Any] = Depends(require_auth),
):
    """Confirm successful analysis and start video generation task."""
    # Use the dedicated storage method to check readiness based purely on metadata
    if not storage.check_ready_for_confirmation(file_id):
        # Attempt to get status for a more informative error message
        try:
            current_status_data = await get_file_status_route(
                file_id, request, storage, user
            )
            current_status_str = current_status_data.get("status", "unknown")
        except HTTPException:
            current_status_str = "not found or error retrieving status"
        except Exception:
            current_status_str = "unexpected error retrieving status"

        logger.warning(
            f"Confirmation attempt for file {file_id} failed readiness check. Current derived status: {current_status_str}."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File not ready for confirmation. Metadata indicates beat detection did not succeed or beats file is missing. Current status: {current_status_str}.",
        )

    try:
        task = generate_video_task.delay(file_id)
        if not task or not task.id:
            raise RuntimeError("Failed to get task ID after submission.")
    except Exception as e:
        logger.error(
            f"Error queuing video generation task for {file_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Failed to queue video generation task."
        )

    try:
        storage.update_metadata(file_id, {"video_generation": task.id})
    except Exception as e:
        logger.error(
            f"Error updating metadata for {file_id} after video task submission: {e}",
            exc_info=True,
        )

    return JSONResponse(
        content={
            "status": "ok",
            "message": "Video generation initiated",
            "task_id": task.id,
        }
    )


@main_router.get("/file/{file_id}", response_class=HTMLResponse, name="file_page_route")
async def file_page_route(
    request: Request,
    file_id: str,
    templates: Jinja2Templates = Depends(get_templates),  # <-- Inject templates
    storage: FileMetadataStorage = Depends(get_storage),
    user: Dict[str, Any] = Depends(require_auth),
):
    """Render the file view page, showing current status and results."""
    # (Keep existing logic)
    try:
        # Correctly call get_file_status_route, passing dependencies explicitly
        file_status_data = await get_file_status_route(
            file_id=file_id,
            request=request,  # Pass the request object
            storage=storage,  # Pass the storage dependency
            user=user,  # Pass the user dependency
        )
    except HTTPException as e:
        if e.status_code == 404:
            # Use injected templates
            return templates.TemplateResponse(
                request,
                "404.html",
                {"message": f"File with ID {file_id} not found."},
                status_code=404,
            )
        else:
            raise e
    except Exception as e:
        logger.error(
            f"Error getting status for file page {file_id}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="Error retrieving file status.")

    # Use injected templates
    return templates.TemplateResponse(
        request,
        "file_view.html",
        {
            "request": request,
            "file_id": file_id,
            "file_status": file_status_data,
            "user": user,
            "app_dir": str(BASE_DIR),
        },
    )


@main_router.get("/download/{file_id}", name="download_video")
async def download_video_route(
    file_id: str,
    storage: FileMetadataStorage = Depends(get_storage),
    user: Dict[str, Any] = Depends(require_auth),
):
    """Download the generated video file."""
    # (Keep existing logic)
    video_path = storage.get_video_file_path(file_id)
    if not video_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Video file not found."
        )

    metadata = storage.get_metadata(file_id)
    original_filename = metadata.get("original_filename", f"audio_{file_id}")
    base_name = pathlib.Path(original_filename).stem
    download_name = f"{base_name}_with_beats.mp4"

    return FileResponse(
        path=str(video_path), filename=download_name, media_type="video/mp4"
    )


# --- FastAPI App Creation Function ---


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    current_config = _global_config

    # Initialize services required by dependencies and store in context
    # Use existing components from context if available (e.g., patched by tests)
    if "storage" not in _app_context:
        _app_context["storage"] = FileMetadataStorage(current_config.storage)
    if "auth_manager" not in _app_context:
        _app_context["auth_manager"] = UserManager(
            users={"users": [user.__dict__ for user in current_config.users]}
        )
    # Note: Templates instance (_global_templates) is already created at module level

    app = FastAPI(
        title=current_config.app.name,
        version=current_config.app.version,
        debug=current_config.app.debug,
    )

    # Mount static files
    static_dir = BASE_DIR / "static"
    if not static_dir.is_dir():
        logger.warning(f"Static directory not found at {static_dir}, creating it.")
        static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Include the routers
    app.include_router(auth_router)
    app.include_router(main_router)

    logger.info(f"FastAPI application '{current_config.app.name}' configured.")
    return app


# --- Main Application Instance & Entry Point ---
app = create_app()


def main():
    """Entry point for running the web application directly using uvicorn."""
    host = "0.0.0.0"
    port = 8000
    reload = _global_config.app.debug

    logger.info(f"Starting Uvicorn server on {host}:{port} (Reload: {reload})")
    uvicorn.run(
        "web_app.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="debug" if reload else "info",
    )


if __name__ == "__main__":
    main()
