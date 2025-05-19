# Beat Detection Web Application

This web application allows users to upload audio files, analyze them for beats, and generate visualization videos that mark each beat. It's built with FastAPI and integrates with the existing beat detection and video generation code.

## Features

- Asynchronous beat detection and analysis using Celery and Redis
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

The application uses Celery with Redis for handling long-running tasks. You'll need to run both the web application and the Celery worker.

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

### Restarting After Code Changes

When you make changes to the code, you need to restart the Celery workers to ensure they use the updated code:

1. Stop all existing Celery workers:

```bash
pkill -f "celery -A celery_app.app worker"
```

2. Flush the Redis cache to remove any stored task data:

```bash
redis-cli flushall
```

3. Start a fresh Celery worker:

```bash
cd web_app
./start_worker.py
```

This prevents issues with Celery running old code from its cache.

## Troubleshooting

### Confirm Endpoint Issues

If you're experiencing a 400 Bad Request error when using the `/confirm/{file_id}` endpoint, it could be due to one of the following reasons:

1. **Beat detection task not completed successfully**: The beat detection task must have a "SUCCESS" state before video generation can be initiated.

2. **Missing beat file**: The beats file (containing the detected beat timestamps) is required for video generation. The application now includes better checking and error handling to validate that beat files exist.

3. **Metadata inconsistency**: In some cases, the metadata might not be properly updated. The application now includes automatic metadata validation and repair, which should fix most issues.

If you still experience issues, check the application logs for more detailed error messages:

```bash
tail -f web_app/celery.log
```

## Integration with Existing Code

This web application integrates with the existing beat detection and video generation code:

- **Beat Detection**: Uses the `BeatDetector` class from `beat_detection.core.detector` to analyze audio files and detect beats.
- **Video Generation**: Uses the `generate_counter_video` function from `beat_detection.cli.generate_videos` to create visualization videos.

## Architecture

- **FastAPI**: Web framework for handling HTTP requests
- **Celery**: Task queue for handling long-running operations with integrated Redis configuration
- **Redis**: Used as a message broker for Celery and as a backend for storing task results
- **File-Based Storage Design**: 
  - `FileMetadataStorage` class handles persistent storage of file metadata
  - `StateManager` class manages task state persistence using the filesystem
  - Both provide consistent error handling and standardized file paths

## Project Structure

- `app.py`: Main FastAPI application
- `storage.py`: File-based metadata storage implementation
- `state_manager.py`: File-based state management
- `tasks.py`: Celery task definitions
- `celery_app.py`: Celery application initialization
- `celery_config.py`: Celery and Redis settings (consolidated)
- `task_executor.py`: Task state constants
- `templates/`: HTML templates
  - `index.html`: Main page template
- `static/`: Static files
  - `css/styles.css`: Stylesheet
  - `js/app.js`: Client-side JavaScript
- `uploads/`: Temporary directory for uploaded files
- `output/`: Directory for processed files and generated videos

## Redis Configuration

The application uses Redis for Celery's task queue and result backend. All Redis configuration is now consolidated in the `celery_config.py` module.

### Environment Variables

You can configure Redis connection through environment variables:

- `REDIS_HOST`: Redis host (default: 'localhost')
- `REDIS_PORT`: Redis port (default: 6379)
- `REDIS_DB`: Redis database number (default: 0)
- `REDIS_PASSWORD`: Redis password (default: None)

### Docker Configuration

When using Docker, set these variables in the docker-compose.yml file:

```yaml
environment:
  - REDIS_HOST=redis
  - REDIS_PORT=6379
  - REDIS_DB=0
  # - REDIS_PASSWORD=your_password_if_needed
```

## API Endpoints

- `GET /`: Main page
- `POST /upload`: Upload an audio file
- `POST /analyze/{file_id}`: Analyze an uploaded audio file (starts a Celery task)
- `GET /status/{file_id}`: Get processing status
- `POST /confirm/{file_id}`: Confirm analysis and generate video (starts a Celery task)
- `GET /download/{file_id}`: Download the generated video
- `GET /task/{task_id}`: Get the status of a Celery task
