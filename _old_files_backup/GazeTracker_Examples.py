"""
Advanced Gaze Tracking Examples
Demonstrates different use cases and customizations
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
import os
from time import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from GazeTracker import GazeTracker

# Configuration
FACE_LANDMARK_MODEL = "./face_landmarker.task"
CAMERA_INDEX = 0


class GazeAnalyzer:
    """Advanced gaze analysis with AOI (Area of Interest) tracking"""
    
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.gaze_history = []
        self.max_history = 30  # Store last 30 gaze points for smoothing
        
        # Define Areas of Interest (AOIs)
        self.aois = {
            'top_left': ((0, 0), (frame_width // 3, frame_height // 3)),
            'top_center': ((frame_width // 3, 0), (2 * frame_width // 3, frame_height // 3)),
            'top_right': ((2 * frame_width // 3, 0), (frame_width, frame_height // 3)),
            'center': ((frame_width // 3, frame_height // 3), (2 * frame_width // 3, 2 * frame_height // 3)),
            'bottom_left': ((0, 2 * frame_height // 3), (frame_width // 3, frame_height)),
            'bottom_center': ((frame_width // 3, 2 * frame_height // 3), (2 * frame_width // 3, frame_height)),
            'bottom_right': ((2 * frame_width // 3, 2 * frame_height // 3), (frame_width, frame_height)),
        }
        
        self.aoi_colors = {
            'top_left': (255, 0, 0),
            'top_center': (0, 255, 0),
            'top_right': (0, 0, 255),
            'center': (255, 255, 0),
            'bottom_left': (255, 0, 255),
            'bottom_center': (0, 255, 255),
            'bottom_right': (128, 128, 0),
        }
    
    def add_gaze_point(self, gaze_2d):
        """Add gaze point to history for smoothing and analysis"""
        if gaze_2d is not None:
            self.gaze_history.append(gaze_2d.copy())
            if len(self.gaze_history) > self.max_history:
                self.gaze_history.pop(0)
    
    def get_smoothed_gaze(self):
        """Get smoothed gaze point using moving average"""
        if not self.gaze_history:
            return None
        return np.mean(self.gaze_history, axis=0)
    
    def get_current_aoi(self, gaze_2d):
        """Determine which AOI the gaze is currently in"""
        if gaze_2d is None:
            return None
        
        x, y = gaze_2d
        for aoi_name, (top_left, bottom_right) in self.aois.items():
            if (top_left[0] <= x <= bottom_right[0] and 
                top_left[1] <= y <= bottom_right[1]):
                return aoi_name
        
        return None
    
    def draw_aois(self, frame):
        """Draw Areas of Interest on frame"""
        for aoi_name, (top_left, bottom_right) in self.aois.items():
            color = self.aoi_colors[aoi_name]
            cv2.rectangle(frame, top_left, bottom_right, color, 1)
            cv2.putText(frame, aoi_name, (top_left[0] + 5, top_left[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    def draw_gaze_trail(self, frame):
        """Draw trail of gaze points for visualization"""
        if len(self.gaze_history) < 2:
            return
        
        for i in range(1, len(self.gaze_history)):
            pt1 = tuple(self.gaze_history[i-1].astype(int))
            pt2 = tuple(self.gaze_history[i].astype(int))
            
            # Fade color based on age (older points are darker)
            intensity = int(255 * i / len(self.gaze_history))
            color = (intensity, intensity, 255)
            
            cv2.line(frame, pt1, pt2, color, 1)


def example_1_basic_gaze_tracking():
    """Example 1: Basic gaze tracking with simple visualization"""
    print("\n=== Example 1: Basic Gaze Tracking ===")
    print("Press 'q' to quit")
    
    face_base_options = python.BaseOptions(model_asset_path=FACE_LANDMARK_MODEL)
    face_landmark_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_landmark_options)
    gaze_tracker = GazeTracker()
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_result = face_landmarker.detect(mp_image)
        
        if face_result.face_landmarks:
            for face in face_result.face_landmarks:
                gaze_2d, debug_info = gaze_tracker.track_gaze(frame, face, w, h)
                gaze_tracker.draw_gaze_visualization(frame, gaze_2d, debug_info)
        
        cv2.imshow("Example 1: Basic Gaze Tracking", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def example_2_aoi_tracking():
    """Example 2: Track which area of screen user is looking at"""
    print("\n=== Example 2: Area of Interest (AOI) Tracking ===")
    print("Press 'q' to quit")
    print("The screen is divided into 7 regions (top-left, top-center, etc.)")
    
    face_base_options = python.BaseOptions(model_asset_path=FACE_LANDMARK_MODEL)
    face_landmark_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_landmark_options)
    gaze_tracker = GazeTracker()
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    analyzer = GazeAnalyzer(w, h)
    aoi_counter = {aoi: 0 for aoi in analyzer.aois.keys()}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Draw AOIs
        analyzer.draw_aois(frame)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_result = face_landmarker.detect(mp_image)
        
        if face_result.face_landmarks:
            for face in face_result.face_landmarks:
                gaze_2d, debug_info = gaze_tracker.track_gaze(frame, face, w, h)
                
                if gaze_2d is not None:
                    analyzer.add_gaze_point(gaze_2d)
                    
                    # Get current AOI
                    current_aoi = analyzer.get_current_aoi(gaze_2d)
                    if current_aoi:
                        aoi_counter[current_aoi] += 1
                    
                    # Draw gaze trail
                    analyzer.draw_gaze_trail(frame)
                
                gaze_tracker.draw_gaze_visualization(frame, gaze_2d, debug_info)
        
        # Display AOI statistics
        stats_text = "AOI Distribution: "
        cv2.putText(frame, stats_text, (10, h - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset = h - 40
        for aoi_name, count in sorted(aoi_counter.items()):
            if count > 0:
                text = f"{aoi_name}: {count}"
                color = analyzer.aoi_colors[aoi_name]
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset -= 20
        
        cv2.imshow("Example 2: AOI Tracking", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def example_3_comparison_both_eyes():
    """Example 3: Compare gaze from left eye vs right eye vs both"""
    print("\n=== Example 3: Left vs Right vs Both Eyes ===")
    print("Press 'q' to quit")
    print("Comparing gaze tracking accuracy between eyes")
    
    face_base_options = python.BaseOptions(model_asset_path=FACE_LANDMARK_MODEL)
    face_landmark_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_landmark_options)
    gaze_tracker = GazeTracker()
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Split screen into 3 sections for visualization
    section_width = w // 3
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Create 3 copies for different eye selections
        frame_left = frame.copy()
        frame_right = frame.copy()
        frame_both = frame.copy()
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_result = face_landmarker.detect(mp_image)
        
        if face_result.face_landmarks:
            for face in face_result.face_landmarks:
                # Left eye only
                gaze_2d_left, debug_left = gaze_tracker.track_gaze(
                    frame_left, face, w, h, eye_selection='left'
                )
                gaze_tracker.draw_gaze_visualization(frame_left, gaze_2d_left, debug_left)
                
                # Right eye only
                gaze_2d_right, debug_right = gaze_tracker.track_gaze(
                    frame_right, face, w, h, eye_selection='right'
                )
                gaze_tracker.draw_gaze_visualization(frame_right, gaze_2d_right, debug_right)
                
                # Both eyes (averaged)
                gaze_2d_both, debug_both = gaze_tracker.track_gaze(
                    frame_both, face, w, h, eye_selection='both'
                )
                gaze_tracker.draw_gaze_visualization(frame_both, gaze_2d_both, debug_both)
        
        # Create combined display
        combined = np.hstack([frame_left, frame_right, frame_both])
        
        # Add labels
        cv2.putText(combined, "LEFT EYE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(combined, "RIGHT EYE", (section_width + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, "BOTH EYES", (2 * section_width + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Example 3: Eye Comparison", combined)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    print("\n" + "="*50)
    print("GAZE TRACKING - ADVANCED EXAMPLES")
    print("="*50)
    print("\nChoose an example to run:")
    print("1. Basic Gaze Tracking")
    print("2. Area of Interest (AOI) Tracking")
    print("3. Left vs Right vs Both Eyes Comparison")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-3): ").strip()
    
    if choice == '1':
        example_1_basic_gaze_tracking()
    elif choice == '2':
        example_2_aoi_tracking()
    elif choice == '3':
        example_3_comparison_both_eyes()
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
