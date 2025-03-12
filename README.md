# Beat Detection

A Python package for detecting beats in music files with optional video visualization.

## Features

- Detect beats in various audio formats (MP3, WAV, FLAC, M4A, OGG)
- Detect and mark downbeats (first beat of each measure)
- Analyze beat intervals for irregularities
- Skip intro sections with inconsistent beat patterns
- Generate statistics about detected beats (tempo, regularity)
- Process individual files or entire directories in batch mode (including subdirectories)
- Preserve directory structure in output files
- Generate summary statistics across multiple files
- Create visualization videos:
  - Counter videos: displays beat count (1-4 or custom pattern)
  - Optional flash videos: background flashes on each beat

## Installation

Requires Python 3.12 or newer:

```bash
# Install the package
pip install -e .

# Install development dependencies
pip install -e '.[dev]'
```

## Usage

### Command Line

#### Beat Detection (using detect-beats)

Detect beats in audio files:

```bash
# Process a single file
detect-beats path/to/file.mp3

# Process a directory (including all subdirectories)
detect-beats path/to/directory

# Process the default directory (data/original)
detect-beats

# The directory structure is preserved in the output
# For example, if you have audio files in data/original/salsa/,
# the output will be created in data/beats/salsa/
```

Beat detection options:

```bash
# Beat detection options
detect-beats --min-bpm 110 --max-bpm 150 file.mp3
detect-beats --no-skip-intro file.mp3
detect-beats --tolerance 15.0 file.mp3

# Output options
detect-beats --output-dir data/custom_beats file.mp3
detect-beats --quiet file.mp3
```

#### Video Generation (using generate-videos)

Generate videos from beat timestamp files:

```bash
# Generate counter videos from a single beat file (default)
generate-videos path/to/file_beats.txt

# Generate videos for all beat files in a directory
generate-videos path/to/directory

# You can also use the original audio file - it will find the corresponding beats file
generate-videos path/to/original_audio.mp3

# Specify custom options
generate-videos --meter 3 --resolution 1920x1080 path/to/file_beats.txt
```

Video generation options:

```bash
# Video type options
generate-videos --flash beats.txt        # Also generate flash video (off by default)
generate-videos --no-counter beats.txt   # Skip counter video

# Video format options
generate-videos --resolution 1920x1080 beats.txt  # Set resolution
generate-videos --fps 60 beats.txt               # Higher framerate
generate-videos --flash-duration 0.2 beats.txt   # Longer flash effect

# Time signature options
generate-videos --meter 3 beats.txt              # Use 3/4 time signature

# Output control
generate-videos --output-dir path beats.txt  # Custom output directory
```

### As a Library

#### Beat Detection

```python
from beat_detection.core.detector import BeatDetector
import numpy as np

# Create detector with custom parameters
detector = BeatDetector(min_bpm=110, max_bpm=140, tolerance_percent=10.0)

# Process a single file
input_file = "input.mp3"
beats_file = "output/input_beats.txt"

# Detect beats
beat_timestamps, stats, irregular_beats = detector.detect_beats(input_file)

# Save beat timestamps for later use
np.savetxt(beats_file, beat_timestamps, fmt='%.3f')
```

#### Video Generation

```python
from beat_detection.core.video import BeatVideoGenerator
import numpy as np

# Create video generator with custom settings
video_gen = BeatVideoGenerator(
    resolution=(1920, 1080),
    fps=30,
    bg_color=(20, 20, 20),
    flash_color=(255, 255, 255),
    count_colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],  # For custom time signatures
    font_size=200
)

# Load beat timestamps from file
beats_file = "output/input_beats.txt"
beat_timestamps = np.loadtxt(beats_file)

# Generate counter video (default output)
video_gen.create_counter_video(
    audio_file="input.mp3",
    beat_timestamps=beat_timestamps,
    output_file="output/input_counter.mp4",
    meter=3  # For 3/4 time
)

# Optionally generate flash video
video_gen.create_flash_video(
    audio_file="input.mp3",
    beat_timestamps=beat_timestamps,
    output_file="output/input_flash.mp4",
    flash_duration=0.1
)
```

## Output Files

### Beat Detection (`detect-beats`)

For each processed file, the tool generates:
- Text file with beat timestamps and downbeat flags (`*_beats.txt`)
  - Format: `<timestamp> <downbeat_flag>` (1=downbeat, 0=regular beat)
- Text file with beat statistics (`*_beat_stats.txt`)
- Text file with beat intervals for debugging (`*_intervals.txt`)

When processing multiple files, it also generates a summary file with statistics across all files (`batch_summary.txt`).

### Video Generation (`generate-videos`)

For each beat timestamp file, the tool can generate:
- Flash video: screen flashes on each beat (`*_flash.mp4`)
- Counter video: displays beat numbers (1-N) in sequence (`*_counter.mp4`)