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
from typing import List, Optional
import uuid

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from beat_detection.core.detector import BeatDetector
from beat_detection.utils import file_utils, reporting
from beat_detection.utils.constants import AUDIO_EXTENSIONS
from beat_detection.cli.generate_videos import generate_counter_video, load_beat_data

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

# Create a dictionary to store processing status
processing_status = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload an audio file for processing.
    
    Returns the file ID for subsequent API calls.
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
    processing_status[file_id] = {
        "status": "uploaded",
        "filename": filename,
        "file_path": str(file_path),
        "beats_file": None,
        "video_file": None,
        "stats": None
    }
    
    return {"file_id": file_id, "filename": filename}


@app.post("/analyze/{file_id}")
async def analyze_audio(file_id: str, background_tasks: BackgroundTasks):
    """
    Analyze the uploaded audio file to detect beats.
    
    This is an asynchronous operation. The client should poll the /status endpoint
    to check when processing is complete.
    """
    # Check if the file exists
    if file_id not in processing_status:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Update status
    processing_status[file_id]["status"] = "analyzing"
    
    # Run beat detection in the background
    background_tasks.add_task(detect_beats, file_id)
    
    return {"status": "analyzing", "file_id": file_id}


@app.get("/status/{file_id}")
async def get_status(file_id: str):
    """Get the processing status for a file."""
    if file_id not in processing_status:
        raise HTTPException(status_code=404, detail="File not found")
    
    return processing_status[file_id]


@app.post("/confirm/{file_id}")
async def confirm_analysis(file_id: str, background_tasks: BackgroundTasks):
    """
    Confirm the beat analysis and generate the video.
    
    This is an asynchronous operation. The client should poll the /status endpoint
    to check when processing is complete.
    """
    # Check if the file exists and has been analyzed
    if file_id not in processing_status:
        raise HTTPException(status_code=404, detail="File not found")
    
    if processing_status[file_id]["status"] != "analyzed":
        raise HTTPException(status_code=400, detail="File has not been analyzed yet")
    
    # Update status
    processing_status[file_id]["status"] = "generating_video"
    
    # Generate video in the background
    background_tasks.add_task(generate_video, file_id)
    
    return {"status": "generating_video", "file_id": file_id}


@app.get("/download/{file_id}")
async def download_video(file_id: str):
    """Download the generated video."""
    # Check if the file exists and video has been generated
    if file_id not in processing_status:
        raise HTTPException(status_code=404, detail="File not found")
    
    if processing_status[file_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Video generation not completed")
    
    video_file = processing_status[file_id]["video_file"]
    
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
        file_info = processing_status[file_id]
        file_path = file_info["file_path"]
        
        # Create output directory
        output_path = OUTPUT_DIR / file_id
        output_path.mkdir(exist_ok=True)
        
        # Initialize beat detector
        detector = BeatDetector(
            min_bpm=60,
            max_bpm=240,
            tolerance_percent=10.0
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
        processing_status[file_id].update({
            "status": "analyzed",
            "beats_file": str(beats_file),
            "stats": {
                "bpm": stats.tempo_bpm,
                "total_beats": len(beat_timestamps),
                "duration": beat_timestamps[-1] if len(beat_timestamps) > 0 else 0,
                "irregularity_percent": stats.irregularity_percent,
                "detected_meter": detected_meter
            }
        })
        
    except Exception as e:
        # Update status with error
        processing_status[file_id].update({
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
        file_info = processing_status[file_id]
        file_path = file_info["file_path"]
        beats_file = file_info["beats_file"]
        
        # Create output directory
        output_path = OUTPUT_DIR / file_id
        output_path.mkdir(exist_ok=True)
        
        # Generate output video path
        input_path = pathlib.Path(file_path)
        video_file = output_path / f"{input_path.stem}_counter.mp4"
        
        # Load beat data
        beat_timestamps, downbeats, intro_end_idx, ending_start_idx, detected_meter = load_beat_data(beats_file)
        
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
                verbose=True
            )
            
            # Check if the video file was actually created despite potential warnings
            if success or video_file.exists():
                # Update processing status
                processing_status[file_id].update({
                    "status": "completed",
                    "video_file": str(video_file)
                })
            else:
                raise Exception("Failed to generate video")
        except Exception as video_error:
            # Check if the video file was created despite the error
            if video_file.exists() and video_file.stat().st_size > 0:
                print(f"Warning during video generation: {video_error}, but video file was created successfully")
                processing_status[file_id].update({
                    "status": "completed",
                    "video_file": str(video_file),
                    "warning": str(video_error)
                })
            else:
                # Re-raise the exception if no video was created
                raise
        
    except Exception as e:
        # Update status with error
        processing_status[file_id].update({
            "status": "error",
            "error": str(e)
        })


def main():
    """Entry point for the web application."""
    uvicorn.run("web_app.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
