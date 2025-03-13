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
from typing import List, Optional, Dict, Any
import uuid

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from beat_detection.core.detector import BeatDetector
from beat_detection.utils import file_utils, reporting
from beat_detection.utils.constants import AUDIO_EXTENSIONS
from beat_detection.cli.generate_videos import generate_counter_video, load_beat_data

# Import the state manager
from state_manager import StateManager

# Create FastAPI app
app = FastAPI(
    title="Beat Detection App",
    description="Upload audio files, detect beats, and generate visualization videos",
    version="1.0.0"
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

# Create a state manager for processing status
STATE_DIR = pathlib.Path("./state")
state_manager = StateManager(STATE_DIR)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


# Routes for debugging with URL paths that match processing stages
@app.get("/analyzing/{file_id}", response_class=HTMLResponse)
async def analyzing_page(request: Request, file_id: str):
    """Render the main page with file_id in analyzing state."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/analyzed/{file_id}", response_class=HTMLResponse)
async def analyzed_page(request: Request, file_id: str):
    """Render the main page with file_id in analyzed state."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_generation/{file_id}", response_class=HTMLResponse)
async def video_generation_page(request: Request, file_id: str):
    """Render the main page with file_id in video generation state."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/completed/{file_id}", response_class=HTMLResponse)
async def completed_page(request: Request, file_id: str):
    """Render the main page with file_id in completed state."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_audio(request: Request, file: UploadFile = File(...)):
    """
    Upload an audio file for processing.
    
    Returns the file ID for subsequent API calls or redirects to the analysis page.
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
    
    # Initialize processing status
    state_manager.update_state(file_id, {
        "status": "uploaded",
        "filename": filename,
        "file_path": str(file_path),
        "beats_file": None,
        "video_file": None,
        "stats": None,
        "progress": {
            "status": "File uploaded successfully",
            "percent": 100
        }
    })
    
    # Check if this is an AJAX request or a traditional form submission
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
    
    if is_ajax:
        # For AJAX requests, return JSON response
        return {"file_id": file_id, "filename": filename}
    else:
        # For traditional form submissions, redirect to the analysis page
        return RedirectResponse(url=f"/analyzing/{file_id}", status_code=303)


@app.post("/analyze/{file_id}")
async def analyze_audio(request: Request, file_id: str, background_tasks: BackgroundTasks):
    """
    Analyze the uploaded audio file to detect beats.
    
    This is an asynchronous operation. The client should poll the /status endpoint
    to check when processing is complete.
    """
    # Check if the file exists
    file_state = state_manager.get_state(file_id)
    if not file_state:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Update status with initial progress
    state_manager.update_state(file_id, {
        "status": "analyzing",
        "progress": {
            "status": "Starting analysis",
            "percent": 0
        }
    })
    
    # Run beat detection in the background
    background_tasks.add_task(detect_beats, file_id)
    
    # Check if this is an AJAX request or a traditional form submission
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
    
    if is_ajax:
        # For AJAX requests, return JSON response
        return {"status": "analyzing", "file_id": file_id}
    else:
        # For traditional form submissions, redirect to the analyzing page
        return RedirectResponse(url=f"/analyzing/{file_id}", status_code=303)


@app.get("/status/{file_id}")
async def get_status(file_id: str):
    """Get the processing status for a file."""
    status_data = state_manager.get_state(file_id)
    if not status_data:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Debug log the status
    print(f"STATUS ENDPOINT: Current status for {file_id}: {status_data.get('status')}")
    
    # More detailed debugging for video generation
    if status_data.get("status") == "generating_video":
        progress_data = status_data.get('progress', {})
        print(f"STATUS ENDPOINT: Video progress details: {progress_data}")
        print(f"STATUS ENDPOINT: Progress percent: {progress_data.get('percent')}")
        print(f"STATUS ENDPOINT: Progress status: {progress_data.get('status')}")
    
    return status_data

@app.get("/processing_queue")
async def get_processing_queue():
    """
    Get the list of files currently in processing.
    
    Returns the current processing status for all files.
    """
    return state_manager.get_all_states()

@app.post("/confirm/{file_id}")
async def confirm_analysis(request: Request, file_id: str, background_tasks: BackgroundTasks):
    """
    Confirm the beat analysis and generate the video.
    
    This is an asynchronous operation. The client should poll the /status endpoint
    to check when processing is complete.
    """
    # Check if the file exists and has been analyzed
    file_state = state_manager.get_state(file_id)
    if not file_state:
        raise HTTPException(status_code=404, detail="File not found")
    
    if file_state.get("status") != "analyzed":
        raise HTTPException(status_code=400, detail="File has not been analyzed yet")
    
    # Update status with initial progress for video generation
    state_manager.update_state(file_id, {
        "status": "generating_video",
        "progress": {
            "status": "Starting video generation",
            "percent": 0
        }
    })
    
    # Generate video in the background
    background_tasks.add_task(generate_video, file_id)
    
    # Check if this is an AJAX request or a traditional form submission
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
    
    if is_ajax:
        # For AJAX requests, return JSON response
        return {"status": "generating_video", "file_id": file_id}
    else:
        # For traditional form submissions, redirect to the video generation page
        return RedirectResponse(url=f"/video_generation/{file_id}", status_code=303)


@app.get("/download/{file_id}")
async def download_video(file_id: str):
    """Download the generated video."""
    # Check if the file exists and video has been generated
    file_state = state_manager.get_state(file_id)
    if not file_state:
        raise HTTPException(status_code=404, detail="File not found")
    
    if file_state.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Video generation not completed")
    
    video_file = file_state.get("video_file")
    
    if not os.path.exists(video_file):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=video_file, 
        filename=os.path.basename(video_file),
        media_type="video/mp4"
    )


async def detect_beats(file_id: str):
    """
    Detect beats in the uploaded audio file.
    
    This function runs in the background and updates the processing status.
    """
    try:
        # Get file information
        file_info = state_manager.get_state(file_id)
        if not file_info:
            raise Exception(f"File state not found for ID: {file_id}")
            
        file_path = file_info["file_path"]
        
        # Create output directory
        output_path = OUTPUT_DIR / file_id
        output_path.mkdir(exist_ok=True)
        
        # Create a synchronous wrapper for the progress callback
        def sync_update_progress(status, progress):
            # Update progress in the state manager
            percent = progress * 100
            state_manager.update_progress(file_id, status, percent)
        
        # Initialize beat detector with progress callback
        detector = BeatDetector(
            min_bpm=60,
            max_bpm=240,
            tolerance_percent=10.0,
            progress_callback=sync_update_progress
        )
        
        # Detect beats
        beat_timestamps, stats, irregular_beats, downbeats, intro_end_idx, ending_start_idx, detected_meter = detector.detect_beats(
            file_path, skip_intro=True, skip_ending=True
        )
        
        # Generate output file paths
        input_path = pathlib.Path(file_path)
        beats_file = output_path / f"{input_path.stem}_beats.txt"
        stats_file = output_path / f"{input_path.stem}_beat_stats.txt"
        
        # Save beat timestamps and statistics
        reporting.save_beat_timestamps(
            beat_timestamps, beats_file, downbeats, 
            intro_end_idx=intro_end_idx, ending_start_idx=ending_start_idx,
            detected_meter=detected_meter
        )
        
        reporting.save_beat_statistics(
            stats, irregular_beats, stats_file, 
            filename=input_path.name
        )
        
        # Update processing status
        state_manager.update_state(file_id, {
            "status": "analyzed",
            "beats_file": str(beats_file),
            "stats": {
                "bpm": stats.tempo_bpm,
                "total_beats": len(beat_timestamps),
                "duration": beat_timestamps[-1] if len(beat_timestamps) > 0 else 0,
                "irregularity_percent": stats.irregularity_percent,
                "detected_meter": detected_meter
            },
            "progress": {
                "status": "Beat detection complete",
                "percent": 100
            }
        })
        
    except Exception as e:
        # Update status with error
        state_manager.update_state(file_id, {
            "status": "error",
            "error": str(e)
        })


async def generate_video(file_id: str):
    """
    Generate a beat visualization video.
    
    This function runs in the background and updates the processing status.
    """
    try:
        # Get file information
        file_info = state_manager.get_state(file_id)
        if not file_info:
            raise Exception(f"File state not found for ID: {file_id}")
            
        file_path = file_info["file_path"]
        beats_file = file_info["beats_file"]
        
        # Create output directory
        output_path = OUTPUT_DIR / file_id
        output_path.mkdir(exist_ok=True)
        
        # Update progress
        state_manager.update_progress(file_id, "Loading beat data", 10)
        
        # Generate output video path
        input_path = pathlib.Path(file_path)
        video_file = output_path / f"{input_path.stem}_counter.mp4"
        
        # Load beat data
        beat_timestamps, downbeats, intro_end_idx, ending_start_idx, detected_meter = load_beat_data(beats_file)
        
        # Update progress
        state_manager.update_progress(file_id, "Preparing video generation", 30)
        
        # Update progress
        state_manager.update_progress(file_id, "Generating video", 50)
        
        # Create a synchronous wrapper for the video progress callback
        def sync_video_progress_callback(status, progress):
            # Update progress in the state manager
            percent = progress * 100
            print(f"VIDEO PROGRESS: {status} - {percent:.1f}%")
            state_manager.update_progress(file_id, status, percent)
            
            # Make sure the status is set to generating_video
            current_state = state_manager.get_state(file_id)
            if "status" not in current_state or current_state["status"] != "generating_video":
                state_manager.update_state(file_id, {"status": "generating_video"})
        
        # Generate video
        try:
            success = generate_counter_video(
                audio_path=input_path,
                output_file=video_file,
                beat_timestamps=beat_timestamps,
                downbeats=downbeats,
                intro_end_idx=intro_end_idx,
                ending_start_idx=ending_start_idx,
                meter=detected_meter,
                verbose=True,
                progress_callback=sync_video_progress_callback
            )
            
            # Update progress
            state_manager.update_progress(file_id, "Finalizing video", 90)
            
            # Check if the video file was actually created despite potential warnings
            if success or video_file.exists():
                # Update processing status
                state_manager.update_state(file_id, {
                    "status": "completed",
                    "video_file": str(video_file),
                    "progress": {
                        "status": "Video generation complete",
                        "percent": 100
                    }
                })
            else:
                raise Exception("Failed to generate video")
        except Exception as video_error:
            # Check if the video file was created despite the error
            if video_file.exists() and video_file.stat().st_size > 0:
                print(f"Warning during video generation: {video_error}, but video file was created successfully")
                state_manager.update_state(file_id, {
                    "status": "completed",
                    "video_file": str(video_file),
                    "warning": str(video_error),
                    "progress": {
                        "status": "Video generation complete (with warnings)",
                        "percent": 100
                    }
                })
            else:
                # Re-raise the exception if no video was created
                raise
        
    except Exception as e:
        # Update status with error
        state_manager.update_state(file_id, {
            "status": "error",
            "error": str(e)
        })


# Test endpoint for progress bar
@app.get("/test-progress")
async def test_progress():
    """Test endpoint that returns a simple HTML page with a progress bar test."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Progress Bar Test</title>
        <style>
            :root {
                --primary-color: #4a6fa5;
                --primary-dark: #3a5a8c;
                --secondary-color: #6c757d;
                --light-color: #f8f9fa;
                --dark-color: #343a40;
                --success-color: #28a745;
                --danger-color: #dc3545;
                --warning-color: #ffc107;
                --border-radius: 8px;
                --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                --transition: all 0.3s ease;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: var(--dark-color);
                background-color: #f5f7fa;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .card {
                background-color: white;
                border-radius: var(--border-radius);
                box-shadow: var(--box-shadow);
                padding: 30px;
                margin-bottom: 30px;
            }
            
            h1 {
                color: var(--primary-color);
                margin-bottom: 20px;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }
            
            /* Progress Bar Styles - Matching the main app */
            .progress-container {
                margin: 20px 0;
            }
            
            .progress-bar {
                height: 10px;
                background-color: #e9ecef;
                border-radius: 5px;
                overflow: hidden;
                margin-bottom: 10px;
            }
            
            .progress-fill {
                height: 100%;
                background-color: var(--primary-color);
                width: 0%;
                transition: width 0.3s ease;
            }
            
            .progress-status {
                margin-top: 10px;
                font-size: 14px;
            }
            
            /* Button Styles */
            button {
                padding: 12px 24px;
                background-color: var(--primary-color);
                color: white;
                border: none;
                border-radius: var(--border-radius);
                cursor: pointer;
                font-weight: bold;
                transition: var(--transition);
            }
            
            button:hover {
                background-color: var(--primary-dark);
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Progress Bar Test</h1>
            <p>This page tests the progress bar functionality by simulating 100 updates at 0.5 second intervals.</p>
            
            <div class="progress-container">
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill"></div>
                </div>
                <div class="progress-status" id="progress-status">Ready to start</div>
            </div>
            
            <button id="start-test">Start Test</button>
        </div>
        
        <script>
            document.getElementById('start-test').addEventListener('click', function() {
                // Reset progress
                const progressFill = document.getElementById('progress-fill');
                const progressStatus = document.getElementById('progress-status');
                progressFill.style.width = '0%';
                progressStatus.textContent = 'Starting test...';
                
                // Disable button during test
                this.disabled = true;
                
                // Simulate 100 progress updates
                let count = 0;
                const totalUpdates = 100;
                
                const updateProgress = function() {
                    count++;
                    const percent = (count / totalUpdates) * 100;
                    
                    // Update progress bar
                    progressFill.style.width = `${percent}%`;
                    progressStatus.textContent = `Update ${count}/${totalUpdates} - ${percent.toFixed(1)}%`;
                    
                    console.log(`Progress update: ${count}/${totalUpdates} - ${percent.toFixed(1)}%`);
                    
                    // Continue until we reach 100 updates
                    if (count < totalUpdates) {
                        setTimeout(updateProgress, 500);
                    } else {
                        progressStatus.textContent = 'Test completed!';
                        document.getElementById('start-test').disabled = false;
                    }
                };
                
                // Start the updates
                setTimeout(updateProgress, 500);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# Test endpoint for progress polling
@app.get("/test-polling")
async def test_polling():
    """Test endpoint that uses the main app.js file for progress polling."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Progress Polling Test</title>
        <style>
            :root {
                --primary-color: #4a6fa5;
                --primary-dark: #3a5a8c;
                --secondary-color: #6c757d;
                --light-color: #f8f9fa;
                --dark-color: #343a40;
                --success-color: #28a745;
                --danger-color: #dc3545;
                --warning-color: #ffc107;
                --border-radius: 8px;
                --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                --transition: all 0.3s ease;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: var(--dark-color);
                background-color: #f5f7fa;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .card {
                background-color: white;
                border-radius: var(--border-radius);
                box-shadow: var(--box-shadow);
                padding: 30px;
                margin-bottom: 30px;
            }
            
            h1 {
                color: var(--primary-color);
                margin-bottom: 20px;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }
            
            /* Progress Bar Styles - Matching the main app */
            .progress-container {
                margin: 20px 0;
            }
            
            .progress-bar {
                height: 10px;
                background-color: #e9ecef;
                border-radius: 5px;
                overflow: hidden;
                margin-bottom: 10px;
            }
            
            .progress-fill {
                height: 100%;
                background-color: var(--primary-color);
                width: 0%;
                transition: width 0.3s ease;
            }
            
            .progress-status {
                margin-top: 10px;
                font-size: 14px;
            }
            
            /* Button Styles */
            button {
                padding: 12px 24px;
                background-color: var(--primary-color);
                color: white;
                border: none;
                border-radius: var(--border-radius);
                cursor: pointer;
                font-weight: bold;
                transition: var(--transition);
            }
            
            button:hover {
                background-color: var(--primary-dark);
            }
            
            /* Debug Panel */
            .debug-panel {
                margin-top: 30px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: var(--border-radius);
                padding: 15px;
            }
            
            .debug-panel h3 {
                margin-top: 0;
                color: var(--secondary-color);
            }
            
            #debug-log {
                max-height: 200px;
                overflow-y: auto;
                background-color: #343a40;
                color: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
                font-family: monospace;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Progress Polling Test</h1>
            <p>This page tests the polling mechanism used in the main app to update the progress bar.</p>
            
            <div class="progress-container">
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill"></div>
                </div>
                <div class="progress-status" id="progress-status">Ready to start</div>
            </div>
            
            <button id="start-test">Start Test</button>
            
            <div class="debug-panel">
                <h3>Debug Information</h3>
                <div id="debug-log"></div>
            </div>
        </div>
        
        <script src="/static/js/app.js"></script>
        
        <script>
            // Set up the debug log
            const debugLog = document.getElementById('debug-log');
            
            // Debug logging function
            function logMessage(message) {
                const timestamp = new Date().toLocaleTimeString();
                const entry = document.createElement('div');
                entry.textContent = '[' + timestamp + '] ' + message;
                debugLog.appendChild(entry);
                debugLog.scrollTop = debugLog.scrollHeight;
                console.log('[DEBUG] ' + message);
            }
            
            // Start button click handler
            document.getElementById('start-test').addEventListener('click', function() {
                // Reset progress
                const progressContainer = document.querySelector('.progress-container');
                const progressFill = document.getElementById('progress-fill');
                
                // Use the app.js functions
                setProgress(progressContainer, 0);
                updateProgressStatus(progressContainer, 'Starting test...');
                
                // Disable button during test
                this.disabled = true;
                logMessage('Test started');
                
                // Generate a test ID
                const testId = 'test-' + Math.random().toString(36).substring(2, 15);
                logMessage('Generated test ID: ' + testId);
                
                // Start the test process on the server
                fetch('/start-test-process/' + testId)
                    .then(response => response.json())
                    .then(data => {
                        logMessage('Server response: ' + JSON.stringify(data));
                        
                        // Start polling for status
                        const statusCheckInterval = setInterval(() => {
                            fetch('/test-status/' + testId)
                                .then(response => response.json())
                                .then(statusData => {
                                    logMessage('Status update: ' + JSON.stringify(statusData));
                                    
                                    if (statusData.progress) {
                                        // Log the current progress bar state
                                        const currentWidth = progressFill.style.width;
                                        logMessage('Current progress bar width: ' + currentWidth);
                                        
                                        // Update progress
                                        setProgress(progressContainer, statusData.progress.percent);
                                        updateProgressStatus(progressContainer, statusData.progress.status);
                                        
                                        // Log the new progress bar state
                                        setTimeout(() => {
                                            const newWidth = progressFill.style.width;
                                            logMessage('New progress bar width: ' + newWidth);
                                        }, 50);
                                    }
                                    
                                    if (statusData.status === 'completed') {
                                        clearInterval(statusCheckInterval);
                                        logMessage('Test completed');
                                        document.getElementById('start-test').disabled = false;
                                    }
                                })
                                .catch(error => {
                                    logMessage('Error polling status: ' + error);
                                    clearInterval(statusCheckInterval);
                                    document.getElementById('start-test').disabled = false;
                                });
                        }, 300);
                    })
                    .catch(error => {
                        logMessage('Error starting test: ' + error);
                        this.disabled = false;
                    });
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# Test process data storage
test_processes = {}


# Start a test process
@app.get("/start-test-process/{test_id}")
async def start_test_process(test_id: str, background_tasks: BackgroundTasks):
    """Start a simulated process with progress updates."""
    # Initialize the test process data
    test_processes[test_id] = {
        "status": "running",
        "progress": {
            "status": "Initializing test",
            "percent": 0
        },
        "start_time": time.time()
    }
    
    # Start the background task
    background_tasks.add_task(run_test_process, test_id)
    
    return {"status": "started", "test_id": test_id}


# Get test process status
@app.get("/test-status/{test_id}")
async def get_test_status(test_id: str):
    """Get the status of a test process."""
    if test_id not in test_processes:
        raise HTTPException(status_code=404, detail="Test process not found")
    
    return test_processes[test_id]


# Run the test process in the background
async def run_test_process(test_id: str):
    """Simulate a process with progress updates."""
    import asyncio
    import random
    
    # Simulate 20 progress updates
    for i in range(1, 21):
        # Update progress
        percent = i * 5  # 5% increments
        test_processes[test_id]["progress"] = {
            "status": f"Processing step {i}/20",
            "percent": percent
        }
        
        # Simulate variable processing time
        await asyncio.sleep(random.uniform(0.5, 1.5))
    
    # Mark as completed
    test_processes[test_id]["status"] = "completed"
    test_processes[test_id]["progress"] = {
        "status": "Test completed",
        "percent": 100
    }


def main():
    """Entry point for the web application."""
    uvicorn.run("web_app.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
