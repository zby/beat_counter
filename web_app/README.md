# Beat Detection Web Application

This web application allows users to upload audio files, analyze them for beats, and generate visualization videos that mark each beat. It's built with FastAPI and integrates with the existing beat detection and video generation code.

## Features

- Audio file upload (MP3, WAV, FLAC, M4A, OGG)
- Beat detection and analysis
- Statistics display (BPM, total beats, duration, time signature)
- User verification of analysis results
- Video generation with beat visualization
- Video download

## Installation

1. Make sure you have Python 3.8+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

### Using the Starter Scripts

The application now uses Celery for handling long-running tasks. You'll need to run both the web application and the Celery worker.

1. Start the web application:

```bash
cd web_app
./start_app.py
```

2. In a separate terminal, start the Celery worker:

```bash
cd web_app
./start_worker.py
```

The application will be available at http://localhost:8000

### Docker Setup

Alternatively, you can use Docker Compose to run Redis:

```bash
docker-compose up redis -d
```

Then run the application and worker as described above.

## Integration with Existing Code

This web application integrates with the existing beat detection and video generation code:

- **Beat Detection**: Uses the `BeatDetector` class from `beat_detection.core.detector` to analyze audio files and detect beats.
- **Video Generation**: Uses the `generate_counter_video` function from `beat_detection.cli.generate_videos` to create visualization videos.

## Project Structure

- `app.py`: Main FastAPI application
- `templates/`: HTML templates
  - `index.html`: Main page template
- `static/`: Static files
  - `css/styles.css`: Stylesheet
  - `js/app.js`: Client-side JavaScript
- `uploads/`: Temporary directory for uploaded files
- `output/`: Directory for processed files and generated videos

## API Endpoints

- `GET /`: Main page
- `POST /upload`: Upload an audio file
- `POST /analyze/{file_id}`: Analyze an uploaded audio file (starts a Celery task)
- `GET /status/{file_id}`: Get processing status
- `POST /confirm/{file_id}`: Confirm analysis and generate video (starts a Celery task)
- `GET /download/{file_id}`: Download the generated video
- `GET /task/{task_id}`: Get the status of a Celery task
