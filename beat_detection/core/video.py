"""
Video generation with beat visualizations.
"""

import os
import numpy as np
import pathlib
from typing import List, Tuple, Optional, Union, Callable
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    VideoClip, AudioFileClip, CompositeVideoClip, 
    ImageClip, TextClip, ColorClip
)


class BeatVideoGenerator:
    """Generate videos with visual beat indicators."""
    
    def __init__(self, 
                 resolution: Tuple[int, int] = (1280, 720),
                 bg_color: Tuple[int, int, int] = (0, 0, 0),
                 count_colors: List[Tuple[int, int, int]] = None,
                 downbeat_color: Tuple[int, int, int] = (255, 0, 0),
                 font_path: Optional[str] = None,
                 font_size: int = 500,  # Much larger font size
                 fps: int = 30,
                 meter: int = 4):
        """
        Initialize the beat video generator.
        
        Parameters:
        -----------
        resolution : tuple of (width, height)
            Video resolution in pixels
        bg_color : tuple of (r, g, b)
            Background color
        count_colors : list of (r, g, b) tuples
            Colors for each beat count (1, 2, 3, 4) or None for default
        downbeat_color : tuple of (r, g, b)
            Color for downbeat (first beat) counter display
        font_path : str or None
            Path to font file, or None for default
        font_size : int
            Font size for beat counter (default: 500 - very large)
        fps : int
            Frames per second for video
        meter : int
            Number of beats per measure (time signature numerator)
        """
        self.width, self.height = resolution
        self.bg_color = bg_color
        self.meter = meter
        self.downbeat_color = downbeat_color
        
        # Default count colors if not provided
        if count_colors is None:
            self.count_colors = [
                self.downbeat_color,  # 1: Red (downbeat)
                (0, 255, 0),          # 2: Green 
                (0, 0, 255),          # 3: Blue
                (255, 255, 0),        # 4: Yellow
            ]
        else:
            self.count_colors = count_colors
            
        self.font_path = font_path
        self.font_size = font_size
        self.fps = fps
    
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
        return ((self.width - text_width) // 2, (self.height - text_height) // 2)
    
    def _draw_number_with_font(self, draw, text: str, position: Tuple[int, int], 
                             text_color: Tuple[int, int, int], font, text_size: int) -> None:
        """Draw a number using font with outline."""
        if font is None:
            return
            
        # Draw the text with a thick border for visibility
        border_size = max(4, text_size // 50)  # Thicker border for larger text
        
        for dx in range(-border_size, border_size + 1, border_size):
            for dy in range(-border_size, border_size + 1, border_size):
                if dx != 0 or dy != 0:  # Skip the center position
                    draw.text((position[0] + dx, position[1] + dy), text, fill=(0, 0, 0), font=font)
        
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
    

    
    def get_current_beat_info(self, t: float, beat_timestamps: np.ndarray, 
                               downbeats: np.ndarray) -> Tuple[int, int, float, bool]:
        """
        Get beat information for a given time point.
        
        Parameters:
        -----------
        t : float
            Time in seconds
        beat_timestamps : numpy.ndarray
            Array of beat timestamps in seconds
        downbeats : numpy.ndarray
            Array of indices that correspond to downbeats (required)
            
        Returns:
        --------
        tuple
            (current_beat_idx, beat_count, time_since_beat, is_downbeat) where:
            - current_beat_idx: Index of the current/most recent beat (-1 if before first beat)
            - beat_count: Count number (1 to meter) for display
            - time_since_beat: Time elapsed since the current beat (seconds)
            - is_downbeat: Whether the current beat is a downbeat
        """
        # Check if beat_timestamps is empty
        if len(beat_timestamps) == 0:
            return -1, 0, 0.0, False
            
        # Find the current beat index
        current_beat_idx = np.searchsorted(beat_timestamps, t, side='right') - 1
        
        if current_beat_idx < 0:
            # Before first beat
            return -1, 0, 0.0, False
        
        # Determine if this is a downbeat
        is_downbeat = current_beat_idx in downbeats
        
        # Find the last downbeat (or use 0 if no downbeats before current beat)
        previous_downbeats = downbeats[downbeats <= current_beat_idx]
        last_downbeat_idx = previous_downbeats[-1] if len(previous_downbeats) > 0 else 0
        
        # Calculate beat number within measure (1-based)
        beats_since_downbeat = current_beat_idx - last_downbeat_idx
        beat_count = beats_since_downbeat + 1  # 1-based counting
        
        # If this is a downbeat, always make it beat 1
        if is_downbeat:
            beat_count = 1
        
        # Calculate time since the beat
        beat_time = beat_timestamps[current_beat_idx]
        time_since_beat = t - beat_time
        
        return current_beat_idx, beat_count, time_since_beat, is_downbeat
    
    def create_counter_frame(self, t: float, beat_timestamps: np.ndarray, 
                            downbeats: np.ndarray,
                            meter: Optional[int] = None) -> np.ndarray:
        """
        Create a frame with beat counter for a given time.
        
        Parameters:
        -----------
        t : float
            Time in seconds
        beat_timestamps : numpy.ndarray
            Array of beat timestamps in seconds
        downbeats : numpy.ndarray
            Array of indices that correspond to downbeats (required)
        meter : int, optional
            Number of beats per measure (overrides self.meter if provided)
            
        Returns:
        --------
        numpy.ndarray
            Frame with beat counter
        """
        # Use provided meter or fall back to the instance's meter
        meter_value = meter if meter is not None else self.meter
        
        # Get current beat information with downbeat awareness
        current_beat_idx, beat_count, time_since_beat, is_downbeat = self.get_current_beat_info(
            t, beat_timestamps, downbeats
        )
        
        # If we're before the first beat or have no beats
        if current_beat_idx < 0 or beat_count == 0:
            # Create a plain background
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:, :] = self.bg_color
            return frame
        
        # Prepare text and color
        color_idx = beat_count - 1
        text = str(beat_count)
        
        # Choose color based on whether it's a downbeat or regular beat
        if is_downbeat:
            text_color = self.downbeat_color
            beat_type = "DOWNBEAT"
        else:
            text_color = self.count_colors[color_idx % len(self.count_colors)]
            beat_type = "beat"
        
        # Debug print but only at beat transitions to reduce output
        if time_since_beat < 0.5 / self.fps:  # Just at the start of a beat
            print(f"{beat_type} at t={t:.3f}s: {beat_count}/{meter_value} (beat #{current_beat_idx+1})")
        
        # Create the image for drawing
        img = Image.new('RGB', (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Common calculations
        center_x = self.width // 2
        center_y = self.height // 2
        
        # Create a colored background for better visibility
        rect_size = min(self.width, self.height) * 0.9  # Almost full screen
        bg_rect = [
            (center_x - rect_size/2), (center_y - rect_size/2),
            (center_x + rect_size/2), (center_y + rect_size/2)
        ]
        
        # Draw a semi-transparent background square
        bg_color = tuple(int(c * 0.3) for c in text_color)  # Darker version of text color
        draw.rectangle(bg_rect, fill=bg_color)
        
        # For the number text
        try:
            # Make the text EXTRA large - at least 3/4 of the screen height
            text_size = int(self.height * 0.75)
            
            # Get an appropriate font
            large_font = self._find_large_font(text_size)
            
            # Calculate text position
            position = self._calculate_text_position(draw, text, large_font, text_size)
            
            # Draw the text
            self._draw_number_with_font(draw, text, position, text_color, large_font, text_size)
            
        except Exception as e:
            # If all attempts at drawing text fail, make a simple text indicator
            print(f"Font rendering error: {e}")
            self._draw_fallback_number(draw, text, center_x, center_y, rect_size, bg_rect, text_color)
        
        # Convert PIL Image to numpy array
        return np.array(img)
    

    
    def create_counter_video(self, audio_file: Union[str, pathlib.Path], 
                            output_file: Union[str, pathlib.Path],
                            beat_timestamps: np.ndarray,
                            downbeats: np.ndarray,
                            meter: Optional[int] = None) -> str:
        """
        Create a video with beat counter (1-N based on meter).
        
        Parameters:
        -----------
        audio_file : str or pathlib.Path
            Path to the input audio file
        output_file : str or pathlib.Path
            Path to save the output video file
        beat_timestamps : numpy.ndarray
            Array of beat timestamps in seconds
        downbeats : numpy.ndarray
            Array of indices that correspond to downbeats
        meter : int, optional
            Number of beats per measure, overrides object's meter if provided
            
        Returns:
        --------
        str
            Path to the created video file
        """
        # Ensure paths are strings
        audio_file_str = str(audio_file)
        output_file_str = str(output_file)
        
        # Use provided meter or fall back to the instance's meter
        meter_value = meter if meter is not None else self.meter
        
        # Load audio file
        audio = AudioFileClip(audio_file_str)
        
        # Create a function that returns the frame at time t
        def make_frame(t):
            return self.create_counter_frame(t, beat_timestamps, downbeats, meter_value)
        
        # Create video clip
        video = VideoClip(make_frame, duration=audio.duration)
        
        # Set audio
        video = video.with_audio(audio)
        
        # Write video file
        video.write_videofile(output_file_str, fps=self.fps, 
                             codec='libx264', audio_codec='aac')
        
        return output_file_str