"""
Hand Detection Module
Detects hands and landmarks in real-time using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional, List, Tuple


class HandDetector:
    """Hand detection using MediaPipe HandLandmarker"""
    
class HandDetector:
    """Hand detection using MediaPipe HandLandmarker"""
    
    def __init__(self, model_path: str, num_hands: int = 2, 
                 confidence_threshold: float = 0.5):
        """
        Initialize Hand Detector
        
        Args:
            model_path: Path to hand_landmarker.task model
            num_hands: Maximum number of hands to detect
            confidence_threshold: Confidence threshold for detections
        """
        self.num_hands = num_hands
        self.confidence_threshold = confidence_threshold
        
        # Initialize MediaPipe HandLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
    
    def detect(self, frame: np.ndarray) -> Tuple[List, List]:
        """
        Detect hands in frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (hand_landmarks, handedness)
        """
        # Convert BGR to RGB and create MediaPipe image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        detection_result = self.detector.detect(mp_image)
        
        return detection_result.hand_landmarks, detection_result.handedness
    
    def get_hand_centroid(self, landmarks, hand_idx: int = 0) -> Tuple[float, float]:
        """Get centroid of hand landmarks"""
        if not landmarks or hand_idx >= len(landmarks):
            return None
        
        x_coords = [lm.x for lm in landmarks[hand_idx]]
        y_coords = [lm.y for lm in landmarks[hand_idx]]
        
        return np.mean(x_coords), np.mean(y_coords)
