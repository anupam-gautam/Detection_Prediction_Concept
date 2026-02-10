"""
Quick test of hand detection and drawing
"""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import HAND_MODEL_PATH, CAMERA_INDEX
from detectors import HandDetector

print("[*] Testing hand detection drawing...\n")

# Initialize detector
hand_detector = HandDetector(HAND_MODEL_PATH, num_hands=2)

# Open camera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("✗ Cannot open camera")
    sys.exit(1)

print("[*] Detecting hands... (show your hand, press 'q' to exit)")
frame_count = 0
hands_detected = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect
        hand_landmarks, handedness = hand_detector.detect(frame)
        
        if hand_landmarks:
            hands_detected = True
            print(f"\n✓ Hand detected on frame {frame_count}!")
            print(f"  Hands: {len(hand_landmarks)}")
            print(f"  Handedness entries: {len(handedness)}")
            
            # Try to draw
            try:
                frame = hand_detector.draw_hands(frame, hand_landmarks, handedness)
                print("  ✓ Drawing successful")
            except Exception as e:
                print(f"  ✗ Drawing error: {e}")
                raise
        
        # Show with FPS
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if hands_detected:
            cv2.putText(frame, "HANDS DETECTED!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Hand Detection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    cap.release()
    cv2.destroyAllWindows()

print(f"\n{'✓' if hands_detected else '⚠'} Test complete - Frames: {frame_count}, Hands detected: {hands_detected}")
