#!/usr/bin/env python3
"""
Script to cut 20 seconds from the Bavarian polka audio file.
This creates a shorter version of the audio file for testing purposes.
"""

from pathlib import Path

from pydub import AudioSegment

# Define paths
TEST_FIXTURES_DIR = Path(__file__).parent / "fixtures"
POLKA_AUDIO_FILE = TEST_FIXTURES_DIR / "bavarian-beer-fest-316616.mp3"
OUTPUT_FILE = TEST_FIXTURES_DIR / "bavarian-beer-fest-20sec.mp3"


def cut_audio(input_path, output_path, start_ms=0, duration_ms=20000):
    """
    Cut a portion of an audio file and save it.

    Args:
        input_path: Path to the input audio file
        output_path: Path to save the output audio file
        start_ms: Start position in milliseconds (default: 0)
        duration_ms: Duration to extract in milliseconds (default: 20000 = 20 seconds)
    """
    print(f"Loading audio file: {input_path}")
    audio = AudioSegment.from_file(str(input_path))

    # Get the total duration of the audio
    total_duration_ms = len(audio)
    print(f"Total duration: {total_duration_ms/1000:.2f} seconds")

    # Ensure the start position is valid
    if start_ms >= total_duration_ms:
        raise ValueError(
            f"Start position ({start_ms}ms) exceeds audio duration ({total_duration_ms}ms)"
        )

    # Calculate end position, ensuring it doesn't exceed the audio length
    end_ms = min(start_ms + duration_ms, total_duration_ms)
    actual_duration = end_ms - start_ms

    print(
        f"Extracting segment from {start_ms/1000:.2f}s to {end_ms/1000:.2f}s (duration: {actual_duration/1000:.2f}s)"
    )
    segment = audio[start_ms:end_ms]

    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export the segment
    segment.export(str(output_path), format="mp3")
    print(f"Saved 20-second clip to: {output_path}")
    print(f"File size: {output_path.stat().st_size} bytes")


if __name__ == "__main__":
    print(f"Original file: {POLKA_AUDIO_FILE}")
    if not POLKA_AUDIO_FILE.exists():
        print(f"Error: Input file not found: {POLKA_AUDIO_FILE}")
        exit(1)

    # Cut the first 20 seconds (can adjust start_ms for a different section if desired)
    cut_audio(POLKA_AUDIO_FILE, OUTPUT_FILE)

    if OUTPUT_FILE.exists():
        print("Successfully created 20-second excerpt!")
    else:
        print("Failed to create output file.")
