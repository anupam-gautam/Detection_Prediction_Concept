"""
Diagnostic script to understand hand detection data structure
"""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import HAND_MODEL_PATH, CAMERA_INDEX
from detectors import HandDetector

print("[*] Starting hand detection diagnostic...\n")

# Initialize detector
print("[*] Loading hand detector...")
hand_detector = HandDetector(HAND_MODEL_PATH, num_hands=2)
print("✓ Hand detector loaded\n")

# Open camera
print(f"[*] Opening camera {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"✗ Cannot open camera {CAMERA_INDEX}")
    sys.exit(1)

print("✓ Camera opened")
print("\n[*] Detecting hands (show your hand to camera, press 'q' to exit)...\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect hands
        hand_landmarks, handedness = hand_detector.detect(frame)
        
        if hand_landmarks:
            print(f"\n✓ Hand detected!")
            print(f"  Type of hand_landmarks: {type(hand_landmarks)}")
            print(f"  Number of hands: {len(hand_landmarks)}")
            print(f"  Type of first hand: {type(hand_landmarks[0])}")
            
            print(f"\n  Type of handedness: {type(handedness)}")
            print(f"  Number of handedness entries: {len(handedness)}")
            if handedness:
                print(f"  Type of first handedness: {type(handedness[0])}")
                print(f"  First handedness value: {handedness[0]}")
                
                # Check if it's a list
                if isinstance(handedness[0], list):
                    print(f"  Length of first handedness list: {len(handedness[0])}")
                    if len(handedness[0]) > 0:
                        print(f"  Type of first element: {type(handedness[0][0])}")
                        print(f"  First element: {handedness[0][0]}")
                        print(f"  Dir of first element: {[x for x in dir(handedness[0][0]) if not x.startswith('_')]}")
                
                # Check attributes
                print(f"  Dir of handedness[0]: {[x for x in dir(handedness[0]) if not x.startswith('_')]}")
            
            # Try to get the label
            try:
                if hasattr(handedness[0], 'category_name'):
                    label = handedness[0].category_name
                elif isinstance(handedness[0], list) and len(handedness[0]) > 0 and hasattr(handedness[0][0], 'category_name'):
                    label = handedness[0][0].category_name
                else:
                    label = "Unknown"
                print(f"\n  Hand label: {label}")
            except Exception as e:
                print(f"\n  Error getting label: {e}")
            
            break
        
        cv2.imshow("Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nNo hands detected (quit by user)")
            break

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    cap.release()
    cv2.destroyAllWindows()
