# Beat Detection Web Application

A web application for audio beat detection and visualization. This application allows users to:

1. Upload audio files
2. Detect beats automatically
3. Generate visualization videos with beat markers

## Features

- User authentication system
- File upload and management
- Beat detection for various audio formats
- Video generation with beat markers
- Processing queue for multiple files

## Installation

First, install [uv](https://github.com/astral-sh/uv), a fast Python package installer and resolver:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### System Requirements

```bash
# Install ffmpeg with additional codecs (required for video generation and M4A support)
sudo apt-get install ffmpeg libavcodec-extra58  # On Ubuntu/Debian
# OR
brew install ffmpeg  # On macOS
```

The application supports the following audio formats:
- MP3
- WAV
- FLAC
- M4A (requires libavcodec-extra58 package)
- OGG

Note: For M4A support on Ubuntu/Debian, you need to install the `libavcodec-extra58` package which provides additional codecs including AAC encoding support.

### Setting up a virtual environment

```bash
# Create and activate a virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Installing dependencies

```bash
# Install dependencies from pyproject.toml
uv pip install .

# For development, install with dev dependencies
uv pip install ".[dev]"
```

## Running the Application

### Development Server

```bash
python -m web_app.app
```

### Configuration

The application uses a config directory structure for storing configuration files:

```bash
# Copy example configuration files
cp web_app/config/config.json.example web_app/config/config.json
cp web_app/config/users.json.example web_app/config/users.json

# Edit the configuration files with your settings
nano web_app/config/config.json  # Application settings
nano web_app/config/users.json   # User credentials
```

The configuration files contain:

- `config.json` - Application settings and parameters:
  - App settings (debug mode, host, port, workers)
  - Celery configuration (Redis connection, serialization)
  - Storage settings (upload/output directories, file limits)
  - Video generation parameters (resolution, codecs)

- `users.json` - User credentials and information:
  - Username and password hashes
  - Admin privileges
  - Account creation timestamps

These files are gitignored to prevent committing sensitive information. The example files (`config.json.example` and `users.json.example`) are included in the repository as templates.

### User Management

The application includes a user management script to manage user accounts from the command line:

```bash
# List all users
python tools/manage_users.py list

# Add a new user (password will be generated if not provided)
python tools/manage_users.py add username [--password PASSWORD] [--admin]

# Delete a user
python tools/manage_users.py delete username

# Change a user's password
python tools/manage_users.py password username new_password
```

### Command-line Tools

The application provides command-line tools for processing files directly:

#### Beat Detection

```bash
# Run beat detection on a file by ID
python tools/run_beat_detection.py [file_id] --wait

# Run beat detection on a file by path
python tools/run_beat_detection.py /path/to/file.mp3 --wait

# Specify a custom upload directory
python tools/run_beat_detection.py [file_id] --upload-dir /custom/path/to/uploads
```

#### Video Generation

```bash
# Generate video for a file by ID
python tools/run_video_generation.py [file_id] --wait

# Generate video for a file by path
python tools/run_video_generation.py /path/to/file.mp3 --wait

# Run immediately after beat detection (waits for beat file to be created)
python tools/run_video_generation.py [file_id] --after-beat-detection --wait
```

#### Batch Processing

```bash
# Process all files in the uploads directory (both beat detection and video generation)
python tools/process_batch.py --wait-beats --wait-video

# Skip files that already have beat or video files
python tools/process_batch.py --skip-existing

# Process only a limited number of files
python tools/process_batch.py --limit 5

# Specify a custom upload directory
python tools/process_batch.py --upload-dir /custom/path/to/uploads
```

### Production Deployment

For detailed deployment instructions, see the [Deployment Guide](DEPLOYMENT.md).

## Dependencies

- FastAPI - Web framework
- Uvicorn - ASGI server
- Celery - Task queue for background processing
- Redis - Message broker for Celery
- PyJWT - JWT tokens for authentication
- Pillow - Python Imaging Library for image processing
- MoviePy - Video editing library for Python

## License

MIT