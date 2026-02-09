"""
Simple test script to verify gaze tracking functionality
Runs gaze tracking without hand detection for faster testing
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from GazeTracker import GazeTracker

# Configuration
FACE_LANDMARK_MODEL = "./face_landmarker.task"
CAMERA_INDEX = 0

def main():
    print("=== Gaze Tracking Test ===")
    print("Press 'q' to quit")
    print("Press 's' to switch eye (left/right/both)")
    print()
    
    # Initialize MediaPipe
    print("Initializing MediaPipe...")
    face_base_options = python.BaseOptions(model_asset_path=FACE_LANDMARK_MODEL)
    face_landmark_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_landmark_options)
    
    # Initialize Gaze Tracker
    print("Initializing Gaze Tracker...")
    gaze_tracker = GazeTracker()
    
    # Initialize webcam
    print("Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Get actual resolution
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {w}x{h}")
    print()
    
    eye_selection = 'left'
    frame_count = 0
    success_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame")
            break
        
        frame_count += 1
        
        # Flip frame for selfie view
        frame = cv2.flip(frame, 1)
        
        # Detect faces
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_result = face_landmarker.detect(mp_image)
        
        # Process each detected face
        if face_result.face_landmarks:
            for face_idx, face in enumerate(face_result.face_landmarks):
                # Track gaze
                gaze_2d, debug_info = gaze_tracker.track_gaze(
                    frame, face, w, h, eye_selection=eye_selection
                )
                
                # Get face landmarks for visualization
                face_2d = gaze_tracker.get_face_2d_points(face)
                face_2d[:, 0] *= w
                face_2d[:, 1] *= h
                
                # Get head pose for visualization
                rotation_vec = debug_info.get('head_rotation')
                translation_vec = debug_info.get('head_translation')
                
                # Draw gaze visualization
                gaze_tracker.draw_gaze_visualization(
                    frame, gaze_2d, debug_info, face_2d,
                    rotation_vec, translation_vec
                )
                
                if debug_info.get('success'):
                    success_count += 1
        
        # Display statistics
        fps = frame_count / max(1, success_count) if success_count > 0 else 0
        stats_text = f"Faces: {len(face_result.face_landmarks)} | Eye: {eye_selection} | Success Rate: {100*success_count//max(1,frame_count)}%"
        cv2.putText(frame, stats_text, (10, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Gaze Tracking Test", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Cycle through eye selection options
            if eye_selection == 'left':
                eye_selection = 'right'
            elif eye_selection == 'right':
                eye_selection = 'both'
            else:
                eye_selection = 'left'
            print(f"Eye selection changed to: {eye_selection}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print()
    print("=== Test Results ===")
    print(f"Total frames: {frame_count}")
    print(f"Successful gaze detections: {success_count}")
    print(f"Success rate: {100*success_count//max(1,frame_count)}%")
    print("Test completed!")

if __name__ == "__main__":
    main()
