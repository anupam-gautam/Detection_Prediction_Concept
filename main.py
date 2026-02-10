import cv2
import time
import sys
import os

# Add src to python path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from features.input_monitor import InputMonitor
from features.vision_detector import VisionDetector
from fusion.engine import FusionEngine

def main():
    # 1. Initialize Components
    print("Initializing components...")
    input_monitor = InputMonitor()
    vision_detector = VisionDetector(model_path='models/face_landmarker.task')
    fusion_engine = FusionEngine(smoothing_window=5)

    # 2. Start Detectors
    input_monitor.start()
    vision_detector.start() # Opens camera

    print("System Started. Press 'Esc' to exit.")

    try:
        while True:
            # 3. Get Data
            frame, vision_data = vision_detector.get_frame_data()
            
            if frame is None:
                print("Failed to get frame")
                break

            # Inputs
            is_face_present = vision_data.get("is_face_present", False)
            is_looking_at_screen = vision_data.get("is_looking_at_screen", False)
            gaze_ratio = vision_data.get("gaze_ratio", 0.0)
            
            # Input Activity
            input_active = input_monitor.is_active(threshold_seconds=5.0)
            idle_time = input_monitor.get_idle_time()

            # 4. Fusion
            on_screen_status = fusion_engine.update(
                is_looking_at_screen, 
                is_face_present, 
                input_active
            )

            # 5. Visualization
            # Text info
            status_text = "ON SCREEN" if on_screen_status else "OFF SCREEN"
            status_color = (0, 255, 0) if on_screen_status else (0, 0, 255)
            
            # Draw Status
            cv2.putText(frame, f"Status: {status_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
            
            # Draw Debug Info
            cv2.putText(frame, f"Gaze: {'Screen' if is_looking_at_screen else 'Away'} ({gaze_ratio:.2f})", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Face: {'Yes' if is_face_present else 'No'}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Input: {'Active' if input_active else 'Idle'} ({idle_time:.1f}s)", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show Frame
            cv2.imshow('On-Screen Detection System', frame)

            if cv2.waitKey(5) & 0xFF == 27: # Esc to exit
                break
            
            # Optional: Sleep to control loop rate if needed, but waitKey handles it partially
            
    except KeyboardInterrupt:
        pass
    finally:
        # 6. Cleanup
        print("Stopping system...")
        input_monitor.stop()
        vision_detector.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
