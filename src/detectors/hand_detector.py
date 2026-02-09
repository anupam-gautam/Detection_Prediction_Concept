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
    
    # Hand landmark indices
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    PALM_CENTER = 9
    
    # Hand connections for drawing
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)   # Pinky
    ]
    
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
    
    def draw_hands(self, frame: np.ndarray, hand_landmarks: List, 
                   handedness: List, color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
        """
        Draw hand landmarks and connections on frame
        
        Args:
            frame: Input frame
            hand_landmarks: List of hand landmark lists
            handedness: List of handedness classifications
            color: Color for drawing (BGR)
            thickness: Line thickness
            
        Returns:
            Frame with drawn landmarks
        """
        if not hand_landmarks:
            return frame
        
        h, w = frame.shape[:2]
        
        for hand_idx, landmarks in enumerate(hand_landmarks):
            # Get handedness - handedness is a list of lists, each containing Classification objects
            hand_label = "Unknown"
            if hand_idx < len(handedness) and handedness[hand_idx]:
                # handedness[hand_idx] is a list of Classification objects
                classification_list = handedness[hand_idx]
                if isinstance(classification_list, list) and len(classification_list) > 0:
                    # Get the first (and usually only) classification
                    classification = classification_list[0]
                    if hasattr(classification, 'category_name'):
                        hand_label = classification.category_name
                    else:
                        hand_label = str(classification)
                elif hasattr(classification_list, 'category_name'):
                    hand_label = classification_list.category_name
            
            # Draw connections
            for connection in self.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start = landmarks[start_idx]
                    end = landmarks[end_idx]
                    
                    start_pos = (int(start.x * w), int(start.y * h))
                    end_pos = (int(end.x * w), int(end.y * h))
                    
                    cv2.line(frame, start_pos, end_pos, color, thickness)
            
            # Draw landmarks as circles
            for landmark in landmarks:
                pos = (int(landmark.x * w), int(landmark.y * h))
                cv2.circle(frame, pos, 4, color, -1)
            
            # Draw hand label
            if hand_landmarks[hand_idx]:
                palm = hand_landmarks[hand_idx][self.PALM_CENTER]
                label_pos = (int(palm.x * w), int(palm.y * h) - 20)
                cv2.putText(frame, hand_label, label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def get_hand_centroid(self, landmarks, hand_idx: int = 0) -> Tuple[float, float]:
        """Get centroid of hand landmarks"""
        if not landmarks or hand_idx >= len(landmarks):
            return None
        
        x_coords = [lm.x for lm in landmarks[hand_idx]]
        y_coords = [lm.y for lm in landmarks[hand_idx]]
        
        return np.mean(x_coords), np.mean(y_coords)
