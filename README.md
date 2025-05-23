# Beat Counter Web Application

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
pip install -U uv
```

### System Requirements

```bash
# Install ffmpeg with additional codecs (required for video generation and M4A support)
sudo apt-get install ffmpeg  # On Ubuntu/Debian
# OR
brew install ffmpeg  # On macOS
```

The application supports the following audio formats:
- MP3
- WAV
- FLAC
- M4A
- OGG

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

# For installing with beat_this_algo support
uv pip install ".[beat_this_algo]"

# For development with beat_this_algo support
uv pip install ".[dev,beat_this_algo]"
```

## Running the Application

### Development Server

```bash
python -m beat_counter.web_app.app
```

### Configuration

The application uses configuration files for storing settings:

```bash
# Copy example configuration files
cp etc/config.json.example etc/config.json
cp etc/users.json.example etc/users.json
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

The project provides command-line scripts for direct file processing, installed via `uv pip install .` (or `uv pip install ".[dev]"`). These scripts operate independently of the web application and are useful for batch processing or integration into other workflows.

#### Beat Detection (Single File)

Detects beats in a single audio file and saves the raw beat timestamps, counts, and detected beats-per-bar to a `.beats.json` file in `RawBeats` format. Note that statistical analysis (like tempo or irregularity) is *not* saved in this file.

```bash
# Detect beats and save output next to the audio file (e.g., audio.beats.json)
detect-beats path/to/your/audio.mp3

# Specify an output file path
detect-beats path/to/your/audio.mp3 -o path/to/output/beats_data.beats.json

# Adjust detection parameters (BPM range, tolerance for interval analysis)
# Note: min_measures is not currently configurable via this tool.
detect-beats path/to/your/audio.mp3 --min-bpm 70 --max-bpm 160 --tolerance 15.0 --beats-per-bar 4
```

#### Beat Detection (Batch)

Detects beats for multiple audio files within a directory.

```bash
# Detect beats for all audio files in a directory (outputs .beats files alongside originals)
detect-beats-batch path/to/audio/directory/
```

#### Video Generation (Single File)

Generates a beat visualization video for a single audio file. It requires:
1. A corresponding `.beats` file (containing raw beat data)  
2. A corresponding `._beat_stats` file (containing validated beat statistics)

The `._beat_stats` file is generated when the beat analysis succeeds. If beat validation fails, the raw `.beats` file will still be created, but the `._beat_stats` file will not, and video generation will fail with a clear error message.

```bash
# Generate video for an audio file (requires both .beats and ._beat_stats files to exist)
# Uses default reconstruction parameters (tolerance=10.0, min_measures=5)
generate-video path/to/your/audio.mp3

# Specify an output video path
generate-video path/to/your/audio.mp3 -o path/to/output/video.mp4

# Specify reconstruction parameters (needed if non-defaults were used or desired)
generate-video path/to/your/audio.mp3 --tolerance-percent 15.0 --min-measures 2

# Change video resolution and FPS
generate-video path/to/your/audio.mp3 --resolution 1920x1080 --fps 60
```

#### Video Generation (Batch)

Generates beat visualization videos for multiple audio files in a directory, using their corresponding `.beats` and `._beat_stats` files. Files without valid `._beat_stats` files will be skipped with error messages.

```bash
# Generate videos for all audio files in a directory (requires corresponding .beats and ._beat_stats files)
generate-video-batch path/to/audio/directory/
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

## API Usage

The core beat detection API is organized in a modular structure:

```python
# Get a beat detector (main entry point)
from beat_counter.core.registry import build
from beat_counter.core.detectors.base import DetectorConfig

# Create a detector instance using kwargs (automatically converts to config)
detector = build("madmom", min_bpm=80, max_bpm=160)  # or "beat_this"

# Create a detector with explicit config (preferred approach)
config = DetectorConfig(min_bpm=80, max_bpm=160, fps=100)
detector = build("madmom", config=config)

# Detect beats in an audio file
raw_beats = detector.detect_beats("path/to/audio.mp3")

# Process a batch of files
from beat_counter.core import process_batch
results = process_batch("path/to/directory", detector_name="madmom")

# Extract beats from a single file (high-level API)
# Returns Beats object if validation succeeds, None if only raw beats were saved
from beat_counter.core import extract_beats
beats = extract_beats("path/to/audio.mp3", detector_name="madmom", min_bpm=80, max_bpm=160)

# Check if beat extraction succeeded with validation
if beats is None:
    print("Beat validation failed, but raw beats were saved to .beats file")
else:
    print(f"Beat detection succeeded with {len(beats.timestamps)} beats at {beats.regular_tempo_bpm} BPM")
```

### Error Handling in Beat Processing

The beat processing system implements a fail-fast approach with graceful error handling:

1. **Raw Beat Extraction**: Always saves detected beats to a `.beats` file, even if validation fails.
2. **Beat Analysis & Validation**: When successful, creates a `._beat_stats` file containing validated stats.
3. **Video Generation**: Requires both `.beats` and `._beat_stats` files. Will fail with a clear error message if the stats file is missing.

This approach ensures:
- Raw beat data is always preserved
- Only validated beat data is used for video generation
- Clear error messages indicate validation failures

### Creating Custom Beat Detectors

You can create and register custom beat detector implementations:

```python
from beat_counter.core.registry import register
from beat_counter.core.detectors.base import BaseBeatDetector, DetectorConfig
from beat_counter.core.beats import RawBeats

@register("my_custom_detector")
class MyCustomDetector(BaseBeatDetector):
    def __init__(self, cfg: DetectorConfig):
        super().__init__(cfg)
        # Your initialization code here
        
    def detect_beats(self, audio_path: str) -> RawBeats:
        # Your implementation here
        # Access config via self.cfg.min_bpm, self.cfg.max_bpm, etc.
        pass

# Then use it like any other detector
from beat_counter.core.registry import build
detector = build("my_custom_detector", min_bpm=90, max_bpm=180)
```

## Testing

The project uses pytest for testing. Tests are located in the `tests/` and `beat_counter/tests/` directories.

### Running Tests

```bash
# Run all tests (excluding slow tests by default)
pytest

# Run all tests including slow tests
pytest --override-ini="addopts="

# Run only slow tests
pytest -m "slow"

# Run specific test modules or directories
pytest tests/test_beat_this.py  # This will include slow tests from the specified file
pytest tests/  # This will include slow tests from the tests directory
```

> **Note**: Tests in the top-level `tests/` directory are automatically marked as `slow` and are skipped by default in local development. In CI environments (GitHub Actions), all tests are run including slow tests.

### GPU vs CPU Testing

By default, the tests will use GPU if available for beat detection algorithms that support hardware acceleration (such as beat_this).

#### Force CPU Usage

For CI/CD environments without GPUs (such as GitHub Actions) or to replicate CI failures locally, you can force CPU usage by setting the `BEAT_COUNTER_FORCE_CPU` environment variable:

```bash
BEAT_COUNTER_FORCE_CPU=1 python -m pytest

# Or for specific tests:
BEAT_COUNTER_FORCE_CPU=1 python -m pytest tests/test_beat_this.py
```

This ensures consistent test behavior across different environments. The application automatically detects this environment variable and forces CPU usage for all detector initializations.

## Development Workflow

This project follows GitHub Flow for collaboration and development. The workflow is automated using GitHub Actions to ensure code quality and stability.

### Branching Strategy

- The `main` branch represents the stable, deployable state of the application
- All new work is done in short-lived feature branches created from `main`

### Local Development

1. Create a feature branch from `main`:
   ```bash
   git checkout main
   git pull
   git checkout -b feature/your-feature-name
   ```

2. (Optional) Install pre-commit hooks for local quality checks:
   ```bash
   # Install pre-commit
   uv pip install pre-commit
   
   # Install the hooks
   pre-commit install
   ```

3. Work on your feature, committing frequently to save progress
   - These commits do not need to pass all tests immediately
   - The pre-commit hooks (if installed) will run fast tests before each commit

4. When ready to integrate, push your branch and create a Pull Request:
   ```bash
   git push -u origin feature/your-feature-name
   ```

### Continuous Integration

When a Pull Request is opened, GitHub Actions will automatically:

1. Set up the Python environment
2. Install all dependencies using `uv`
3. Run the full test suite using pytest
4. Report results as status checks on the PR

The workflow configuration is located in `.github/workflows/test.yml`.

### Merging to Main

1. A Pull Request must have the following to be merged:
   - Approval from at least one reviewer
   - All required status checks must pass (including the test suite)

2. Once the PR is merged to `main`:
   - The GitHub Actions workflow will run again on the `main` branch
   - If successful, it will automatically create a Git tag with the format `tests-passed-YYYYMMDDHHMMSS`
   - These tags mark commits where all tests are confirmed to pass

### Branch Protection

The `main` branch is protected with the following rules:

- Pull requests are required for all changes
- At least one approval is required
- The GitHub Actions test workflow must pass before merging
- Force pushes are disabled

This ensures that the `main` branch always represents a stable state where all tests pass.

## License

MIT