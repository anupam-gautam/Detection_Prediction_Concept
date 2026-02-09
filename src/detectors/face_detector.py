"""
Face Detection Module
Detects faces and facial landmarks in real-time using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional, List, Tuple


class FaceDetector:
    """Face detection using MediaPipe FaceLandmarker"""
    
    # Important face landmarks indices
    NOSE_TIP = 1
    LEFT_EYE = 33
    RIGHT_EYE = 263
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    CHIN = 152
    LEFT_CHEEK = 205
    RIGHT_CHEEK = 425
    
    # Face contour points for drawing
    FACE_OVAL = list(range(0, 17))  # Face boundary
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize Face Detector
        
        Args:
            model_path: Path to face_landmarker.task model
            confidence_threshold: Confidence threshold for detections
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize MediaPipe FaceLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=False
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
    
    def detect(self, frame: np.ndarray) -> List:
        """
        Detect faces in frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of face landmarks
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect faces
        detection_result = self.detector.detect(mp_image)
        
        return detection_result.face_landmarks
    
    def draw_faces(self, frame: np.ndarray, face_landmarks: List,
                   color: Tuple[int, int, int] = (255, 0, 0),
                   thickness: int = 2) -> np.ndarray:
        """
        Draw face landmarks on frame
        
        Args:
            frame: Input frame
            face_landmarks: List of face landmark lists
            color: Color for drawing (BGR)
            thickness: Line thickness
            
        Returns:
            Frame with drawn landmarks
        """
        if not face_landmarks:
            return frame
        
        h, w = frame.shape[:2]
        
        for face_idx, landmarks in enumerate(face_landmarks):
            # Draw face contour
            contour_points = []
            for idx in self.FACE_OVAL:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    contour_points.append([int(lm.x * w), int(lm.y * h)])
            
            if contour_points:
                contour_array = np.array(contour_points, dtype=np.int32)
                cv2.polylines(frame, [contour_array], False, color, thickness)
            
            # Draw key landmarks
            key_indices = [
                self.NOSE_TIP, self.LEFT_EYE, self.RIGHT_EYE,
                self.LEFT_EYE_INNER, self.RIGHT_EYE_INNER,
                self.MOUTH_LEFT, self.MOUTH_RIGHT, self.CHIN
            ]
            
            for idx in key_indices:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    pos = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(frame, pos, 3, color, -1)
            
            # Draw eyes
            if self.LEFT_EYE < len(landmarks) and self.RIGHT_EYE < len(landmarks):
                left_eye = landmarks[self.LEFT_EYE]
                right_eye = landmarks[self.RIGHT_EYE]
                
                left_pos = (int(left_eye.x * w), int(left_eye.y * h))
                right_pos = (int(right_eye.x * w), int(right_eye.y * h))
                
                cv2.circle(frame, left_pos, 5, (0, 255, 0), -1)
                cv2.circle(frame, right_pos, 5, (0, 255, 0), -1)
        
        return frame
    
    def get_key_landmarks(self, landmarks: List) -> Optional[dict]:
        """
        Extract key facial landmarks for processing
        
        Args:
            landmarks: Face landmarks from detection
            
        Returns:
            Dictionary with key landmark positions or None
        """
        if not landmarks:
            return None
        
        key_lms = {
            'nose': landmarks[self.NOSE_TIP] if self.NOSE_TIP < len(landmarks) else None,
            'left_eye': landmarks[self.LEFT_EYE] if self.LEFT_EYE < len(landmarks) else None,
            'right_eye': landmarks[self.RIGHT_EYE] if self.RIGHT_EYE < len(landmarks) else None,
            'left_mouth': landmarks[self.MOUTH_LEFT] if self.MOUTH_LEFT < len(landmarks) else None,
            'right_mouth': landmarks[self.MOUTH_RIGHT] if self.MOUTH_RIGHT < len(landmarks) else None,
            'chin': landmarks[self.CHIN] if self.CHIN < len(landmarks) else None,
        }
        
        return key_lms
    
    def get_face_bbox(self, landmarks: List, frame_shape: Tuple[int, int],
                     padding: float = 0.1) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box of face
        
        Args:
            landmarks: Face landmarks
            frame_shape: Shape of frame (height, width)
            padding: Padding ratio
            
        Returns:
            Bounding box as (x1, y1, x2, y2)
        """
        if not landmarks:
            return None
        
        h, w = frame_shape[:2]
        
        # Get min/max coordinates
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        # Apply padding
        x_padding = (x_max - x_min) * padding
        y_padding = (y_max - y_min) * padding
        
        x1 = int(max(0, (x_min - x_padding) * w))
        y1 = int(max(0, (y_min - y_padding) * h))
        x2 = int(min(w, (x_max + x_padding) * w))
        y2 = int(min(h, (y_max + y_padding) * h))
        
        return x1, y1, x2, y2
