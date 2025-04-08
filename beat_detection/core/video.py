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
    VideoClip, AudioFileClip, CompositeVideoClip, 
    ImageClip, TextClip, ColorClip
)
from beat_detection.core.detector import Beats

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
DEFAULT_REGULAR_BEAT_BG_COLOR = (220, 220, 220)  # Light gray for regular beat backgrounds

CODEC = 'libx264'
#CODEC = 'mpeg4'
AUDIO_CODEC = 'aac'

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
        progress_callback: Optional[Callable[[str, float], None]] = None
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
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255)       # Blue
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
        
        # Create and cache the "no beat" frame (beat_count = 0)
        no_beat_frame = self._create_beat_frame(0, beats_per_bar_value)
        self._frame_cache[0] = no_beat_frame
        
        # Create and cache frames for each possible beat count (1 to beats_per_bar)
        for beat_count in range(1, beats_per_bar_value + 1):
            frame = self._create_beat_frame(beat_count, beats_per_bar_value)
            # Cache by beat count for direct access
            self._frame_cache[beat_count] = frame
        
        # Create and cache frames for each timestamp
        for beat_info in beats.beat_list:
            # Store the frame in our cache using timestamp as key
            self._frame_cache[beat_info.timestamp] = self._frame_cache[beat_info.beat_count]
    
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
        frame = np.full((self.resolution[1], self.resolution[0], 3), self.bg_color, dtype=np.uint8)
        
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
            
            # For no beat (current_beat = 0), use a dimmed version of the normal colors
            if current_beat == 0:
                # Use a dimmed version of the normal color for this position
                base_color = self.count_colors[beat - 1] if beat > 1 else self.downbeat_color
                color = tuple(max(c // 4, 0) for c in base_color)  # Dim the color to 25% brightness
            else:
                # Normal coloring logic
                if beat == current_beat:
                    color = self.downbeat_color
                else:
                    color = self.count_colors[beat - 1]
                
            # Draw the rectangle for this beat
            frame[y_start:y_end, x_start:x_end] = color
            
        return frame
    
    def _find_large_font(self, text_size: int) -> Optional[ImageFont.ImageFont]:
        """Find a font suitable for large text display."""
        # Try different approaches to get the largest possible font
        large_font = None
        
        # First approach: Try to get a system font
        try:
            # Find a default system font path that works well with numbers
            for font_name in ["Arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf", "LiberationSans-Regular.ttf"]:
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
        
    def _calculate_text_position(self, draw, text: str, font: Optional[ImageFont.ImageFont], 
                               text_size: int) -> Tuple[int, int]:
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
        return ((self.resolution[0] - text_width) // 2, (self.resolution[1] - text_height) // 2)
    
    def _draw_number_with_font(self, draw, text: str, position: Tuple[int, int], 
                             text_color: Tuple[int, int, int], font, text_size: int) -> None:
        """Draw a number using font with outline."""
        if font is None:
            return
            
        # Draw the text with a thick border for visibility
        border_size = max(4, text_size // 50)  # Thicker border for larger text
        border_color = (0, 0, 0)  # Black outline for visibility on white background
        
        for dx in range(-border_size, border_size + 1, border_size):
            for dy in range(-border_size, border_size + 1, border_size):
                if dx != 0 or dy != 0:  # Skip the center position
                    draw.text((position[0] + dx, position[1] + dy), text, fill=border_color, font=font)
        
        # Draw the main text in its color
        draw.text(position, text, fill=text_color, font=font)
    
    def _draw_fallback_number(self, draw, text: str, center_x: int, center_y: int, 
                            rect_size: float, bg_rect, text_color: Tuple[int, int, int]) -> None:
        """Draw a fallback number using simple lines when font rendering fails."""
        # Create a manual digit as a last resort
        draw.rectangle(bg_rect, fill=text_color)
        
        # Draw the number as a white digit in the center
        white = (255, 255, 255)
        simple_size = rect_size * 0.5
        line_width = int(simple_size // 10)
        
        # Just draw a huge white digit manually
        if text == "1":
            line_points = [(center_x, center_y - simple_size//2), (center_x, center_y + simple_size//2)]
            draw.line(line_points, fill=white, width=line_width)
        elif text == "2":
            points = [
                (center_x - simple_size//4, center_y - simple_size//4),  # Top
                (center_x + simple_size//4, center_y - simple_size//4),  # Right top
                (center_x, center_y),  # Middle
                (center_x - simple_size//4, center_y + simple_size//4),  # Left bottom
                (center_x + simple_size//4, center_y + simple_size//4)   # Right bottom
            ]
            draw.line(points, fill=white, width=line_width)
        elif text == "3":
            # Draw a 3 using lines
            points = [
                (center_x - simple_size//4, center_y - simple_size//4),  # Left top
                (center_x + simple_size//4, center_y - simple_size//4),  # Right top
                (center_x, center_y),  # Middle
                (center_x + simple_size//4, center_y + simple_size//4)   # Right bottom
            ]
            draw.line(points, fill=white, width=line_width)
            draw.line([(center_x - simple_size//4, center_y + simple_size//4), 
                     (center_x + simple_size//4, center_y + simple_size//4)], 
                     fill=white, width=line_width)
        else:  # text == "4" or other
            # Draw a simple 4
            draw.line([(center_x - simple_size//4, center_y - simple_size//4), 
                     (center_x - simple_size//4, center_y)], 
                     fill=white, width=line_width)
            draw.line([(center_x - simple_size//4, center_y), 
                     (center_x + simple_size//4, center_y - simple_size//4)], 
                     fill=white, width=line_width)
            draw.line([(center_x + simple_size//4, center_y - simple_size//4), 
                     (center_x + simple_size//4, center_y + simple_size//4)], 
                     fill=white, width=line_width)
    
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

    def generate_video(self, audio_path: Union[str, pathlib.Path], 
                      beats: Beats,
                      output_path: Union[str, pathlib.Path],
                      sample_beats: Optional[int] = None) -> str:
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
            max_time = beat_timestamps[max_beats - 1] + 1.0  # Add 1 second after the last beat
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
                    if (beat_idx != last_reported["beat"]) or (time_since_last_update >= 1.0):
                        # Update last reported beat and time
                        last_reported["beat"] = beat_idx
                        last_reported["time"] = current_time
                        
                        # Calculate progress percentage (30% to 80%)
                        beat_progress = (beat_idx + 1) / len(beat_timestamps)
                        progress = 0.3 + (beat_progress * 0.5)
                        
                        print(f"DEBUG: Calling progress callback - Processing beat {beat_idx+1}/{len(beat_timestamps)} at time {t:.2f}")
                        self.progress_callback(f"Processing beat {beat_idx+1}/{len(beat_timestamps)}", progress)
            
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
        video.write_videofile(output_file_str, fps=self.fps, 
                            codec=CODEC, audio_codec=AUDIO_CODEC,
                            logger='bar')
        
        # Report progress - completed
        if self.progress_callback:
            print("DEBUG: Calling progress callback - Video generation complete")
            self.progress_callback("Video generation complete", 1.0)
            
        return output_file_str