"""
Utility functions for video processing and visualization
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def apply_clahe(frame: np.ndarray, clip_limit: float = 2.0, 
                tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization to improve visibility.
    
    Args:
        frame: Input frame (BGR)
        clip_limit: Contrast limit
        tile_size: Tile grid size
        
    Returns:
        Enhanced frame
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced = clahe.apply(gray)
    enhanced_frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced_frame


def draw_fps(frame: np.ndarray, fps: float, position: Tuple[int, int] = (10, 30),
             font_scale: float = 0.7, color: Tuple[int, int, int] = (0, 255, 0),
             thickness: int = 2) -> np.ndarray:
    """
    Draw FPS counter on frame.
    
    Args:
        frame: Input frame
        fps: FPS value to display
        position: Text position (x, y)
        font_scale: Font size
        color: Text color (BGR)
        thickness: Text thickness
        
    Returns:
        Frame with FPS text
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
               font_scale, color, thickness)
    
    return frame


def draw_text(frame: np.ndarray, text: str, position: Tuple[int, int],
             font_scale: float = 0.6, color: Tuple[int, int, int] = (255, 255, 255),
             thickness: int = 1, bg_color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Draw text on frame with optional background.
    
    Args:
        frame: Input frame
        text: Text to draw
        position: Text position (x, y)
        font_scale: Font size
        color: Text color (BGR)
        thickness: Text thickness
        bg_color: Optional background color
        
    Returns:
        Frame with text
    """
    if bg_color is not None:
        # Get text size for background
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                   font_scale, thickness)[0]
        x, y = position
        cv2.rectangle(frame, (x - 5, y - text_size[1] - 5),
                     (x + text_size[0] + 5, y + 5), bg_color, -1)
    
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
               font_scale, color, thickness)
    
    return frame


def draw_info_panel(frame: np.ndarray, info_dict: dict, 
                   position: Tuple[int, int] = (10, 30),
                   spacing: int = 25,
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 1) -> np.ndarray:
    """
    Draw information panel with multiple lines of text.
    
    Args:
        frame: Input frame
        info_dict: Dictionary with {label: value}
        position: Starting position (x, y)
        spacing: Line spacing
        color: Text color (BGR)
        thickness: Text thickness
        
    Returns:
        Frame with info panel
    """
    x, y = position
    
    for label, value in info_dict.items():
        text = f"{label}: {value}"
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, color, thickness)
        y += spacing
    
    return frame


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Normalize frame for better visualization.
    
    Args:
        frame: Input frame
        
    Returns:
        Normalized frame
    """
    return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)


def flip_frame(frame: np.ndarray, flip_code: int = 1) -> np.ndarray:
    """
    Flip frame horizontally or vertically.
    
    Args:
        frame: Input frame
        flip_code: 1 for horizontal, 0 for vertical, -1 for both
        
    Returns:
        Flipped frame
    """
    return cv2.flip(frame, flip_code)


class FPSCounter:
    """Helper class to calculate and track FPS"""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS Counter.
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.timestamps = []
    
    def update(self, timestamp: Optional[float] = None) -> float:
        """
        Update with new timestamp and return current FPS.
        
        Args:
            timestamp: Timestamp (if None, uses current time)
            
        Returns:
            Current FPS
        """
        import time
        
        if timestamp is None:
            timestamp = time.time()
        
        self.timestamps.append(timestamp)
        
        # Keep only window_size timestamps
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
        
        # Calculate FPS
        if len(self.timestamps) > 1:
            time_diff = self.timestamps[-1] - self.timestamps[0]
            if time_diff > 0:
                fps = (len(self.timestamps) - 1) / time_diff
                return fps
        
        return 0.0
    
    def reset(self):
        """Reset FPS counter"""
        self.timestamps = []
