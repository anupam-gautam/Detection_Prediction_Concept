import cv2
import time
import sys
import os

# Add src to python path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from features.input_monitor import InputMonitor
from features.vision_detector import VisionDetector
from fusion.engine import InferenceEngine

# Runtime Flag for Camera Preview
CAMERA_PREVIEW_ENABLED = True # Set to False to disable camera preview and run in background mode

def main():
    # 1. Initialize Components
    print("Initializing components...")
    input_monitor = InputMonitor()
    vision_detector = VisionDetector(model_path='models/face_landmarker.task')
    inference_engine = InferenceEngine(smoothing_window=10) # Increased smoothing

    # 2. Start Detectors
    input_monitor.start()
    vision_detector.start() # Opens camera

    print("System Started. Press 'Esc' to exit.")
    if not CAMERA_PREVIEW_ENABLED:
        print("Camera preview disabled. Running in background mode.")

    try:
        while True:
            # 3. Get Data
            frame, vision_data = vision_detector.get_frame_data()
            
            if frame is None:
                print("Failed to get frame or camera disconnected.")
                # Continue loop or break? If camera fails, we might still have input
                # But vision_detector returns None if cap read fails.
                # Let's assume we want to keep running even if camera fails, 
                # but for now break as per original logic, or handle gracefully.
                # Requirement: "Confidence must decrease gracefully when signals are missing"
                # If frame is None, we treat vision signals as missing.
                vision_data = {} # Clear data
                # We shouldn't break if we want to support Input-only.
                # But current vision_detector implementation might need fixing to separate capture from processing if we want robust recovery.
                # For now, let's break on camera failure as per original simple loop, 
                # OR just continue with empty vision data.
                # Let's try to continue.
                if not vision_detector.cap.isOpened():
                     break 
            
            # Inputs
            is_face_present = vision_data.get("is_face_present", False)
            is_looking_at_screen = vision_data.get("is_looking_at_screen", False)
            # gaze_ratio = vision_data.get("gaze_ratio", 0.5) # Not used directly in new engine yet
            
            # Input Activity
            idle_time = input_monitor.get_idle_time()

            # 4. Inference
            result = inference_engine.update(
                is_looking_at_screen, 
                is_face_present, 
                idle_time
            )

            # 5. Visualization / Output
            # Print status to console periodically or every frame?
            # Let's print every 1 second to avoid spam, or just overlay on frame.
            
            if CAMERA_PREVIEW_ENABLED and frame is not None:
                # Text info
                state = result["usage_state"]
                mode = result["attention_mode"]
                prob = result["active_usage_probability"]
                conf = result["confidence_score"]
                reason = result["reasoning_summary"]

                # Colors
                color_map = {
                    "Active": (0, 255, 0),    # Green
                    "Passive": (0, 255, 255), # Yellow
                    "Inactive": (0, 0, 255)   # Red
                }
                status_color = color_map.get(state, (200, 200, 200))

                # Draw Status
                y_start = 50
                line_height = 30
                
                cv2.putText(frame, f"State: {state} ({prob:.2f})", (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                cv2.putText(frame, f"Mode: {mode}", (20, y_start + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Conf: {conf:.2f} ({result['confidence_level']})", (20, y_start + 2*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Reason: {reason}", (20, y_start + 3*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # Show Frame
                cv2.imshow('On-Screen Detection System', frame)
                if cv2.waitKey(5) & 0xFF == 27: # Esc to exit
                    break
            else:
                 # Console mode
                 # Just sleep a bit to not burn CPU if no waitKey
                 time.sleep(0.03) # ~30FPS
                 # Check for break condition - hard without window focus.
                 # Usually console apps use KeyboardInterrupt.
                 pass

    except KeyboardInterrupt:
        pass
    finally:
        # 6. Cleanup
        print("Stopping system...")
        input_monitor.stop()
        vision_detector.stop()
        
        # Save Analytics
        print("Saving session analytics...")
        inference_engine.export_session_data("session_raw_log.txt", "session_report.txt")
        
        if CAMERA_PREVIEW_ENABLED:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
