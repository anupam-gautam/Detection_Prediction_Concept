"""
Multi-Detection System
Real-time detection of hands, faces, and gaze using MediaPipe
"""

import cv2
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import *
from detectors import HandDetector, FaceDetector, GazeTracker
from utils.visualization import draw_fps, draw_text, draw_info_panel, FPSCounter, draw_hand_landmarks


class MultiDetectionSystem:
    """Unified system for hand, face, and gaze detection"""
    
    def __init__(self, enable_hand: bool = True, enable_face: bool = True,
                 enable_gaze: bool = True):
        """
        Initialize detection system.
        
        Args:
            enable_hand: Enable hand detection
            enable_face: Enable face detection
            enable_gaze: Enable gaze tracking
        """
        self.enable_hand = enable_hand
        self.enable_face = enable_face
        self.enable_gaze = enable_gaze and enable_face  # Gaze needs face
        
        # Initialize detectors
        self.hand_detector = None
        self.face_detector = None
        self.gaze_tracker = None
        
        self.fps_counter = FPSCounter()
        self.frame_count = 0
        
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize all enabled detectors"""
        print("[INFO] Initializing detectors...")
        
        try:
            if self.enable_hand:
                print(f"  - Loading hand detector from {HAND_MODEL_PATH}")
                self.hand_detector = HandDetector(
                    HAND_MODEL_PATH,
                    num_hands=MAX_HANDS,
                    confidence_threshold=HAND_CONFIDENCE_THRESHOLD
                )
                print("  ✓ Hand detector ready")
        except Exception as e:
            print(f"  ✗ Hand detector failed: {e}")
            self.enable_hand = False
        
        try:
            if self.enable_face:
                print(f"  - Loading face detector from {FACE_MODEL_PATH}")
                self.face_detector = FaceDetector(
                    FACE_MODEL_PATH,
                    confidence_threshold=FACE_CONFIDENCE_THRESHOLD
                )
                print("  ✓ Face detector ready")
        except Exception as e:
            print(f"  ✗ Face detector failed: {e}")
            self.enable_face = False
            self.enable_gaze = False
        
        try:
            if self.enable_gaze:
                print("  - Initializing gaze tracker")
                
                # Create camera matrix from config
                camera_matrix = np.array([
                    [FOCAL_LENGTH_X, 0, PRINCIPAL_POINT_X],
                    [0, FOCAL_LENGTH_Y, PRINCIPAL_POINT_Y],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                self.gaze_tracker = GazeTracker(
                    camera_matrix=camera_matrix,
                    face_3d_model=FACE_3D_MODEL,
                    eye_3d_model=EYE_3D_MODEL
                )
                print("  ✓ Gaze tracker ready")
        except Exception as e:
            print(f"  ✗ Gaze tracker failed: {e}")
            self.enable_gaze = False
        
        print("\n[INFO] Detector initialization complete")
        print(f"  Hand detection: {'✓' if self.enable_hand else '✗'}")
        print(f"  Face detection: {'✓' if self.enable_face else '✗'}")
        print(f"  Gaze tracking: {'✓' if self.enable_gaze else '✗'}\n")
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process single frame with all enabled detections.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Detection results dictionary
        """
        h, w = frame.shape[:2]
        results = {
            'hand_landmarks': None,
            'handedness': None,
            'face_landmarks': None,
            'gaze_point': None,
            'gaze_info': None,
        }
        
        # Hand detection
        if self.enable_hand and self.hand_detector:
            try:
                hand_landmarks, handedness = self.hand_detector.detect(frame)
                results['hand_landmarks'] = hand_landmarks
                results['handedness'] = handedness
            except Exception as e:
                print(f"[ERROR] Hand detection: {e}")
        
        # Face detection
        if self.enable_face and self.face_detector:
            try:
                face_landmarks = self.face_detector.detect(frame)
                results['face_landmarks'] = face_landmarks
                
                # Gaze tracking (requires face)
                if self.enable_gaze and self.gaze_tracker and face_landmarks:
                    try:
                        gaze_point, gaze_info = self.gaze_tracker.track_gaze(
                            face_landmarks[0],  # Use first face
                            w, h, eye='both'
                        )
                        results['gaze_point'] = gaze_point
                        results['gaze_info'] = gaze_info
                    except Exception as e:
                        print(f"[ERROR] Gaze tracking: {e}")
            except Exception as e:
                print(f"[ERROR] Face detection: {e}")
        
        return results
    
    def draw_results(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """
        Draw detection results on frame.
        
        Args:
            frame: Input frame
            results: Detection results from process_frame
            
        Returns:
            Frame with visualizations
        """
        # Draw hand landmarks
        if results['hand_landmarks']:
            frame = draw_hand_landmarks(
                frame,
                results['hand_landmarks'],
                results['handedness'],
                color=COLOR_HAND
            )
        
        # Draw face landmarks
        if results['face_landmarks']:
            frame = self.face_detector.draw_faces(
                frame,
                results['face_landmarks'],
                color=COLOR_FACE
            )
        
        # Draw gaze point (with debug info for camera-relative arrow)
        if results['gaze_point'] is not None:
            frame = self.gaze_tracker.draw_gaze(
                frame,
                debug_info=results.get('gaze_info'),
                color=COLOR_GAZE
            )
        
        return frame
    
    def run(self, camera_index: int = CAMERA_INDEX, output_path: Optional[str] = None):
        """
        Run detection system on webcam.
        
        Args:
            camera_index: Camera index
            output_path: Optional path to save video
        """
        print(f"[INFO] Opening camera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {camera_index}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        # Get actual frame size
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFO] Camera opened: {actual_width}x{actual_height}")
        print("[INFO] Press 'q' or ESC to exit")
        print()
        
        # Setup video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, CAMERA_FPS,
                                    (actual_width, actual_height))
            print(f"[INFO] Recording to {output_path}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame")
                    break
                
                # Process detections
                results = self.process_frame(frame)
                
                # Draw results
                frame = self.draw_results(frame, results)
                
                # Draw FPS and info
                if SHOW_FPS:
                    fps = self.fps_counter.update()
                    frame = draw_fps(frame, fps)
                
                # Draw detection status
                status_text = "Detections: "
                status_parts = []
                if self.enable_hand:
                    status_parts.append(f"Hand({len(results['hand_landmarks']) if results['hand_landmarks'] else 0})")
                if self.enable_face:
                    status_parts.append(f"Face({len(results['face_landmarks']) if results['face_landmarks'] else 0})")
                if self.enable_gaze:
                    status_parts.append(f"Gaze({'✓' if results['gaze_point'] is not None else '✗'})")
                
                status_text += " | ".join(status_parts)
                frame = draw_text(frame, status_text, (10, 55),
                                 color=COLOR_TEXT, bg_color=COLOR_BACKGROUND)
                
                # Display frame
                cv2.imshow(WINDOW_NAME, frame)
                
                # Save frame if recording
                if writer:
                    writer.write(frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key in EXIT_KEYS:
                    print("\n[INFO] Exiting...")
                    break
                
                self.frame_count += 1
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            print(f"[INFO] Processed {self.frame_count} frames")
            if self.frame_count > 0:
                avg_fps = self.fps_counter.update()
                print(f"[INFO] Average FPS: {avg_fps:.2f}")


def main():
    """Main entry point"""
    # Create detection system
    system = MultiDetectionSystem(
        enable_hand=True,
        enable_face=True,
        enable_gaze=True
    )
    
    # Run on webcam
    system.run(camera_index=CAMERA_INDEX)


if __name__ == '__main__':
    # Import numpy for config
    import numpy as np
    main()
