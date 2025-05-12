"""
Video generation for beat visualization.
"""

import os
import numpy as np
import pathlib
import time
from typing import List, Tuple, Optional, Union, Callable
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    VideoClip,
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    ColorClip,
)
from beat_detection.core.beats import Beats, RawBeats
import logging
from pathlib import Path
from tqdm import tqdm
from beat_detection.utils.file_utils import find_audio_files

# Video dimensions constants
DEFAULT_VIDEO_WIDTH = 720
DEFAULT_VIDEO_HEIGHT = 540
DEFAULT_VIDEO_RESOLUTION = (DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT)

DEFAULT_FPS = 100  # Increased from 60 for smoother video generation

# Color constants
DEFAULT_BG_COLOR = (255, 255, 255)  # White background
DEFAULT_DOWNBEAT_COLOR = (80, 80, 80)  # Darker gray for downbeats (more contrast)
DEFAULT_REGULAR_BEAT_COLOR = (128, 128, 128)  # Medium gray for regular beats
DEFAULT_DOWNBEAT_BG_COLOR = (200, 200, 220)  # Light blue-gray for downbeat backgrounds
DEFAULT_REGULAR_BEAT_BG_COLOR = (
    220,
    220,
    220,
)  # Light gray for regular beat backgrounds

CODEC = "libx264"
# CODEC = 'mpeg4'
AUDIO_CODEC = "aac"


class BeatVideoGenerator:
    """Generates videos with visual beat indicators."""

    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        count_colors: List[Tuple[int, int, int]] = None,
        downbeat_color: Tuple[int, int, int] = (255, 255, 0),
        font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        font_size: int = 72,
        fps: int = DEFAULT_FPS,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        """
        Initialize the video generator.

        Parameters:
        -----------
        resolution : Tuple[int, int]
            Video resolution (width, height)
        bg_color : Tuple[int, int, int]
            Background color (R, G, B)
        count_colors : List[Tuple[int, int, int]]
            Colors for beat counts (R, G, B)
        downbeat_color : Tuple[int, int, int]
            Color for downbeats (R, G, B)
        font_path : str
            Path to font file
        font_size : int
            Font size for beat numbers
        fps : int
            Frames per second (default: 100 for smooth video)
        progress_callback : Optional[Callable[[str, float], None]]
            Callback function for progress updates
        """
        self.resolution = resolution
        self.bg_color = bg_color
        self.count_colors = count_colors or [
            (255, 255, 255),  # White
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
        ]
        self.downbeat_color = downbeat_color
        self.font_path = font_path
        self.font_size = font_size
        self.fps = fps

        # Initialize font
        self.font = ImageFont.truetype(font_path, font_size)

        # Store progress callback
        self.progress_callback = progress_callback

        # Cache for generated frames
        self._frame_cache = {}

    def _fill_frame_cache(self, beats: Beats) -> None:
        """
        Pre-generate all frames needed for the video.

        Parameters:
        -----------
        beats : Beats
            The beats object containing timing information
        """
        # Get the beats per bar value for coloring
        beats_per_bar_value = beats.beats_per_bar

        # Create and cache frames for each possible beat count (1 to beats_per_bar)
        for beat_count in range(beats_per_bar_value + 1):
            frame = self._create_beat_frame(beat_count, beats_per_bar_value)
            # Cache by beat count for direct access
            self._frame_cache[beat_count] = frame

    def _create_beat_frame(self, current_beat: int, beats_per_bar: int) -> np.ndarray:
        """
        Create a frame showing the current beat position.

        Parameters:
        -----------
        current_beat : int
            The current beat number (1-based, or 0 for no beat)
        beats_per_bar : int
            Number of beats per bar

        Returns:
        --------
        np.ndarray
            The frame as a numpy array
        """
        # Create a blank frame with background color
        frame = np.full(
            (self.resolution[1], self.resolution[0], 3), self.bg_color, dtype=np.uint8
        )

        # Convert frame to PIL Image for drawing
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        # If we're not in a regular sequence (current_beat = 0), just return white background
        if current_beat == 0:
            return np.array(img)

        # Calculate dimensions for the beat indicators
        indicator_width = self.resolution[0] // beats_per_bar
        indicator_height = self.resolution[1] // 2

        # Draw each beat indicator
        for beat in range(1, beats_per_bar + 1):
            # Calculate position for this beat indicator
            x_start = (beat - 1) * indicator_width
            x_end = x_start + indicator_width
            y_start = (self.resolution[1] - indicator_height) // 2
            y_end = y_start + indicator_height

            # Use gray for downbeat (beat 1), white for other beats
            color = (200, 200, 200) if beat == 1 else (255, 255, 255)

            # Draw the rectangle for this beat
            draw.rectangle([x_start, y_start, x_end, y_end], fill=color)

            # Only draw the number for the current beat
            if beat == current_beat:
                text = str(beat)
                # Increase text size to 90% of the smaller dimension
                text_size = int(min(indicator_width, indicator_height) * 0.9)
                font = self._find_large_font(text_size)

                # Calculate text position
                text_x = x_start + (indicator_width - text_size) // 2
                text_y = y_start + (indicator_height - text_size) // 2

                # Draw the number in black
                self._draw_number_with_font(
                    draw, text, (text_x, text_y), (0, 0, 0), font, text_size
                )

        # Convert back to numpy array
        return np.array(img)

    def _find_large_font(self, text_size: int) -> Optional[ImageFont.ImageFont]:
        """Find a font suitable for large text display."""
        # Try different approaches to get the largest possible font
        large_font = None

        # First approach: Try to get a system font
        try:
            # Find a default system font path that works well with numbers
            for font_name in [
                "Arial.ttf",
                "DejaVuSans.ttf",
                "FreeSans.ttf",
                "LiberationSans-Regular.ttf",
            ]:
                try:
                    font_path = f"/usr/share/fonts/truetype/{font_name}"
                    large_font = ImageFont.truetype(font_path, text_size)
                    break
                except:
                    pass
        except:
            pass

        # If that failed, try with default font or font path
        if large_font is None:
            try:
                if self.font_path:
                    large_font = ImageFont.truetype(self.font_path, text_size)
                else:
                    # Use default if nothing else works
                    large_font = ImageFont.load_default()
            except:
                try:
                    large_font = ImageFont.load_default()
                except:
                    pass

        return large_font

    def _calculate_text_position(
        self, draw, text: str, font: Optional[ImageFont.ImageFont], text_size: int
    ) -> Tuple[int, int]:
        """Calculate centered position for text."""
        # For very large text, we can approximate
        text_width = text_size * 0.6  # Approximate width
        text_height = text_size * 0.8  # Approximate height

        if font is not None:
            try:
                # Try to get actual dimensions if possible
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                # Use our approximation if this fails
                pass

        # Center position
        return (
            (self.resolution[0] - text_width) // 2,
            (self.resolution[1] - text_height) // 2,
        )

    def _draw_number_with_font(
        self,
        draw,
        text: str,
        position: Tuple[int, int],
        text_color: Tuple[int, int, int],
        font,
        text_size: int,
    ) -> None:
        """Draw a number using font with outline."""
        if font is None:
            return

        # Draw the text with a thick border for visibility
        border_size = max(4, text_size // 50)  # Thicker border for larger text
        border_color = (0, 0, 0)  # Black outline for visibility on white background

        for dx in range(-border_size, border_size + 1, border_size):
            for dy in range(-border_size, border_size + 1, border_size):
                if dx != 0 or dy != 0:  # Skip the center position
                    draw.text(
                        (position[0] + dx, position[1] + dy),
                        text,
                        fill=border_color,
                        font=font,
                    )

        # Draw the main text in its color
        draw.text(position, text, fill=text_color, font=font)

    def _draw_fallback_number(
        self,
        draw,
        text: str,
        center_x: int,
        center_y: int,
        rect_size: float,
        bg_rect,
        text_color: Tuple[int, int, int],
    ) -> None:
        """Draw a fallback number using simple lines when font rendering fails."""
        # Create a manual digit as a last resort
        draw.rectangle(bg_rect, fill=text_color)

        # Draw the number as a white digit in the center
        white = (255, 255, 255)
        simple_size = rect_size * 0.5
        line_width = int(simple_size // 10)

        # Just draw a huge white digit manually
        if text == "1":
            line_points = [
                (center_x, center_y - simple_size // 2),
                (center_x, center_y + simple_size // 2),
            ]
            draw.line(line_points, fill=white, width=line_width)
        elif text == "2":
            points = [
                (center_x - simple_size // 4, center_y - simple_size // 4),  # Top
                (center_x + simple_size // 4, center_y - simple_size // 4),  # Right top
                (center_x, center_y),  # Middle
                (
                    center_x - simple_size // 4,
                    center_y + simple_size // 4,
                ),  # Left bottom
                (
                    center_x + simple_size // 4,
                    center_y + simple_size // 4,
                ),  # Right bottom
            ]
            draw.line(points, fill=white, width=line_width)
        elif text == "3":
            # Draw a 3 using lines
            points = [
                (center_x - simple_size // 4, center_y - simple_size // 4),  # Left top
                (center_x + simple_size // 4, center_y - simple_size // 4),  # Right top
                (center_x, center_y),  # Middle
                (
                    center_x + simple_size // 4,
                    center_y + simple_size // 4,
                ),  # Right bottom
            ]
            draw.line(points, fill=white, width=line_width)
            draw.line(
                [
                    (center_x - simple_size // 4, center_y + simple_size // 4),
                    (center_x + simple_size // 4, center_y + simple_size // 4),
                ],
                fill=white,
                width=line_width,
            )
        else:  # text == "4" or other
            # Draw a simple 4
            draw.line(
                [
                    (center_x - simple_size // 4, center_y - simple_size // 4),
                    (center_x - simple_size // 4, center_y),
                ],
                fill=white,
                width=line_width,
            )
            draw.line(
                [
                    (center_x - simple_size // 4, center_y),
                    (center_x + simple_size // 4, center_y - simple_size // 4),
                ],
                fill=white,
                width=line_width,
            )
            draw.line(
                [
                    (center_x + simple_size // 4, center_y - simple_size // 4),
                    (center_x + simple_size // 4, center_y + simple_size // 4),
                ],
                fill=white,
                width=line_width,
            )

    def create_counter_frame(self, t: float, beats: Beats) -> np.ndarray:
        """
        Get the appropriate cached frame for a given time point.

        Parameters:
        -----------
        t : float
            Time in seconds
        beats : Beats
            Beats object containing beat information

        Returns:
        --------
        numpy.ndarray
            Frame with beat counter from cache
        """

        # Get current beat count, time since last beat, and beat index
        beat_count, time_since_beat, beat_idx = beats.get_info_at_time(t)

        # If beat_count is 0, return the "no beat" frame
        if beat_count == 0:
            return self._frame_cache[0]

        # Use beat_count directly as the cache key - Beats class already ensures it's valid
        cache_key = beat_count

        # Return the cached frame
        return self._frame_cache[cache_key]

    def generate_video(
        self,
        audio_path: Union[str, pathlib.Path],
        beats: Beats,
        output_path: Union[str, pathlib.Path],
        sample_beats: Optional[int] = None,
    ) -> str:
        """
        Generate a beat visualization video.

        Parameters:
        -----------
        audio_path : str or pathlib.Path
            Path to the audio file
        beats : Beats
            Beats object containing beat information
        output_path : str or pathlib.Path
            Path where to save the output video
        sample_beats : int, optional
            Number of beats to process (for testing/quick preview)

        Returns:
        --------
        str
            Path to the generated video file
        """
        # Ensure paths are strings
        audio_file_str = str(audio_path)
        output_file_str = str(output_path)

        # Get beat timestamps array from beats object
        beat_timestamps = beats.timestamps

        # If sample_beats is provided, limit the video to just those beats
        if sample_beats is not None and sample_beats > 0 and len(beat_timestamps) > 0:
            # Limit to the first N beats
            max_beats = min(sample_beats, len(beat_timestamps))
            max_time = (
                beat_timestamps[max_beats - 1] + 1.0
            )  # Add 1 second after the last beat
        else:
            max_time = None  # Use the full audio duration

        # Report progress - starting audio loading
        if self.progress_callback:
            print("DEBUG: Calling progress callback - Loading audio file")
            self.progress_callback("Loading audio file", 0.05)

        # Load audio file
        audio = AudioFileClip(audio_file_str)

        # If we're using a sample, trim the audio
        if max_time is not None:
            # In MoviePy 2.1.2, AudioFileClip doesn't have subclip method
            # Instead, we can create a new clip with the desired duration
            original_duration = audio.duration
            if max_time < original_duration:
                # Use the clip's duration property directly
                audio.duration = max_time

        # Report progress - preparing frames
        if self.progress_callback:
            print("DEBUG: Calling progress callback - Preparing video frames")
            self.progress_callback("Preparing video frames", 0.15)

        # Fill the frame cache with all possible frames
        self._fill_frame_cache(beats)

        # Store the last reported beat index and time to avoid duplicate updates
        last_reported = {"beat": -1, "time": 0}  # Using a dict for mutable reference

        # Create a function that returns the frame at time t
        def make_frame(t):
            # Calculate approximate progress based on current time position
            if self.progress_callback and audio.duration > 0:
                # Map time to progress between 30% and 80%
                progress = 0.3 + (t / audio.duration) * 0.5

                # Find which beat we're currently processing
                _, _, beat_idx = beats.get_info_at_time(t)

                # Report progress in two cases:
                # 1. When we move to a new beat
                # 2. When significant time has passed (at least 1 second) since last update
                current_time = time.time()
                time_since_last_update = current_time - last_reported["time"]

                # Only update progress if we have valid beat info
                if beat_idx >= 0 and beat_idx < len(beat_timestamps):
                    if (beat_idx != last_reported["beat"]) or (
                        time_since_last_update >= 1.0
                    ):
                        # Update last reported beat and time
                        last_reported["beat"] = beat_idx
                        last_reported["time"] = current_time

                        # Calculate progress percentage (30% to 80%)
                        beat_progress = (beat_idx + 1) / len(beat_timestamps)
                        progress = 0.3 + (beat_progress * 0.5)

                        print(
                            f"DEBUG: Calling progress callback - Processing beat {beat_idx+1}/{len(beat_timestamps)} at time {t:.2f}"
                        )
                        self.progress_callback(
                            f"Processing beat {beat_idx+1}/{len(beat_timestamps)}",
                            progress,
                        )

            return self.create_counter_frame(t, beats)

        # Report progress - creating video
        if self.progress_callback:
            print("DEBUG: Calling progress callback - Creating video clip")
            self.progress_callback("Creating video clip", 0.3)

        # Create video clip
        video = VideoClip(make_frame, duration=audio.duration)

        # Set audio
        video = video.with_audio(audio)

        # Report progress - writing video file
        if self.progress_callback:
            print("DEBUG: Calling progress callback - Starting video encoding")
            self.progress_callback("Starting video encoding", 0.1)

        # Write video file without callback
        video.write_videofile(
            output_file_str,
            fps=self.fps,
            codec=CODEC,
            audio_codec=AUDIO_CODEC,
            logger="bar",
        )

        # Report progress - completed
        if self.progress_callback:
            print("DEBUG: Calling progress callback - Video generation complete")
            self.progress_callback("Video generation complete", 1.0)

        return output_file_str


# ---------------------------------------------------------------------------
# High-level Generation Functions (Moved from CLI)
# ---------------------------------------------------------------------------

def generate_single_video_from_files(
    audio_file: Path,
    beats_file: Path,
    output_file: Optional[Path] = None,
    resolution: Tuple[int, int] = DEFAULT_VIDEO_RESOLUTION,
    fps: int = DEFAULT_FPS,
    sample_beats: Optional[int] = None,
    tolerance_percent: float = 10.0,
    min_measures: int = 5,
    verbose: bool = True,
) -> Path:
    """
    Generates a single beat counter video from audio and beat files.

    Loads raw beat data, reconstructs the Beats object, determines the output
    path, instantiates BeatVideoGenerator, and generates the video.

    Parameters
    ----------
    audio_file : Path
        Path to the input audio file.
    beats_file : Path
        Path to the corresponding .beats JSON file.
    output_file : Optional[Path], optional
        Path to save the generated video. If None, saves next to the audio
        file with a '_counter.mp4' suffix. Defaults to None.
    resolution : Tuple[int, int], optional
        Video resolution (width, height). Defaults to DEFAULT_VIDEO_RESOLUTION.
    fps : int, optional
        Frames per second for the output video. Defaults to DEFAULT_FPS.
    sample_beats : Optional[int], optional
        Generate a sample video with only the first N beats. Defaults to None.
    tolerance_percent : float, optional
        Tolerance percentage for reconstructing Beats stats/sections. Defaults to 10.0.
    min_measures : int, optional
        Minimum measures for reconstructing Beats stats/sections. Defaults to 5.
    verbose : bool, optional
        If True, print progress messages to stdout. Defaults to True.

    Returns
    -------
    Path
        The path to the generated video file.

    Raises
    ------
    FileNotFoundError
        If the audio or beats file does not exist.
    RuntimeError
        If loading raw beats or reconstructing the Beats object fails.
    Exception
        Re-raises exceptions from video generation.
    """
    if not audio_file.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    if not beats_file.is_file():
        raise FileNotFoundError(f"Beats file not found: {beats_file}")

    # Load Raw Beats Data
    try:
        raw_beats = RawBeats.load_from_file(beats_file)
        if verbose:
            print(f"Loaded raw beats data from {beats_file} with {len(raw_beats.timestamps)} beats")
    except Exception as e:
        # Use logging for errors, even if verbose is False
        logging.error(f"Failed to load raw beats from {beats_file}: {e}")
        raise RuntimeError(f"Failed to load raw beats from {beats_file}: {e}") from e

    # Reconstruct Beats object
    try:
        beats = Beats(
            raw_beats=raw_beats,
            beats_per_bar=None,  # Infer from pattern
            tolerance_percent=tolerance_percent,
            min_measures=min_measures
        )
        if verbose:
            print(f"Reconstructed Beats: bpb={beats.beats_per_bar}, tol={tolerance_percent}%, min_meas={min_measures}")
    except Exception as e:
        logging.error(f"Failed to reconstruct Beats object: {e}")
        raise RuntimeError(f"Failed to reconstruct Beats object: {e}") from e

    # Determine output video path
    if output_file is None:
        final_output_path = audio_file.with_name(f"{audio_file.stem}_counter.mp4")
    else:
        final_output_path = output_file
        # Ensure parent directory exists if an explicit path was provided
        final_output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Generating video for {audio_file} using reconstructed beats")

    # Create generator
    video_gen = BeatVideoGenerator(resolution=resolution, fps=fps)

    # Generate video
    try:
        # Pass progress callback if needed in the future
        generated_path_str = video_gen.generate_video(
            audio_path=audio_file,
            beats=beats,
            output_path=final_output_path,
            sample_beats=sample_beats,
        )
        generated_path = Path(generated_path_str)
        if verbose:
            print(f"Saved video to {generated_path}")
        return generated_path
    except Exception as e:
        logging.error(f"Video generation failed for {audio_file}: {e}")
        # Optional: Clean up partially generated file?
        # if final_output_path.exists():
        #     try:
        #         final_output_path.unlink()
        #     except OSError:
        #         logging.warning(f"Could not remove partially generated file: {final_output_path}")
        raise # Re-raise the exception after logging

# --- Placeholder for batch function ---

def generate_batch_videos(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    resolution: Tuple[int, int] = DEFAULT_VIDEO_RESOLUTION,
    fps: int = DEFAULT_FPS,
    sample_beats: Optional[int] = None,
    tolerance_percent: float = 10.0,
    min_measures: int = 5,
    verbose: bool = True,
    no_progress: bool = False,
) -> List[Tuple[str, bool, Optional[Path]]]:
    """
    Processes all audio files in a directory tree to generate beat videos.

    Recursively finds audio files, looks for corresponding .beats files, generates
    a video for each pair using generate_single_video_from_files, and returns results.

    Parameters
    ----------
    input_dir : Path
        The root directory to scan recursively for audio files.
    output_dir : Optional[Path], optional
        Directory to save output videos. If None, videos are saved alongside
        audio files. Defaults to None.
    resolution : Tuple[int, int], optional
        Video resolution. Defaults to DEFAULT_VIDEO_RESOLUTION.
    fps : int, optional
        Frames per second. Defaults to DEFAULT_FPS.
    sample_beats : Optional[int], optional
        Limit video generation to the first N beats. Defaults to None.
    tolerance_percent : float, optional
        Tolerance for Beats reconstruction. Defaults to 10.0.
    min_measures : int, optional
        Minimum measures for Beats reconstruction. Defaults to 5.
    verbose : bool, optional
        Print progress messages to stdout. Defaults to True.
    no_progress : bool, optional
        If True, disable the tqdm progress bar. Defaults to False.

    Returns
    -------
    List[Tuple[str, bool, Optional[Path]]]
        A list of tuples for each attempted file:
        (relative_audio_path, success_status, output_video_path_or_None)

    Raises
    ------
    FileNotFoundError
        If the input_dir does not exist.
    """
    if not input_dir.is_dir():
        logging.error(f"Input directory not found: {input_dir}")
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    audio_files = find_audio_files(input_dir)
    if not audio_files:
        logging.warning(f"No audio files found in {input_dir}")
        return []

    if verbose:
        print(f"Found {len(audio_files)} audio files in {input_dir} for video generation.")

    pbar: Optional[tqdm] = None
    if not no_progress and verbose:
        pbar = tqdm(audio_files, desc="Generating videos", unit="file", ncols=100)
        file_iterator = pbar
    else:
        file_iterator = audio_files

    results: List[Tuple[str, bool, Optional[Path]]] = []

    for audio_file in file_iterator:
        relative_path_str = str(audio_file.relative_to(input_dir))
        if pbar:
            pbar.set_description(f"Processing {audio_file.name}")
        elif verbose:
            print(f"\nProcessing: {relative_path_str}")

        beats_file = audio_file.with_suffix(".beats")
        single_output_file: Optional[Path] = None

        try:
            # Remove the check for beats file existence - let generate_single_video_from_files handle it
            
            # Determine the specific output file path for this audio file
            if output_dir:
                # Ensure output dir exists (should be done once, but safe here)
                output_dir.mkdir(parents=True, exist_ok=True)
                # Maintain relative structure within output_dir if input was recursive
                relative_audio_dir = audio_file.parent.relative_to(input_dir)
                specific_output_dir = output_dir / relative_audio_dir
                specific_output_dir.mkdir(parents=True, exist_ok=True)
                single_output_file = specific_output_dir / f"{audio_file.stem}_counter.mp4"
            else:
                # If output_dir is None, generate_single_video_from_files handles default
                single_output_file = None

            # Call the single-file processing function
            output_video_path = generate_single_video_from_files(
                audio_file=audio_file,
                beats_file=beats_file,
                output_file=single_output_file, # Pass the calculated path or None
                resolution=resolution,
                fps=fps,
                sample_beats=sample_beats,
                tolerance_percent=tolerance_percent,
                min_measures=min_measures,
                verbose=verbose, # Pass verbose flag
            )
            results.append((relative_path_str, True, output_video_path))
            if pbar is None and verbose:
                 print(f"Successfully generated video for {relative_path_str}")

        except FileNotFoundError as e:
            # Specifically catch missing beats file or audio file (less likely here)
            if verbose:
                logging.warning(f"Skipping {relative_path_str}: {e}")
            results.append((relative_path_str, False, None))
        except Exception as e:
            # Catch errors during loading, reconstruction, or video generation
            logging.error(f"Error processing {relative_path_str}: {e}")
            # Optionally log traceback for debugging
            # logging.exception(f"Traceback for error processing {relative_path_str}:")
            results.append((relative_path_str, False, None))

    if pbar:
        pbar.close()

    return results
