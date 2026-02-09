"""
Quick diagnostic script to test hand and face detection
"""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import HAND_MODEL_PATH, FACE_MODEL_PATH, CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT
from detectors import HandDetector, FaceDetector

print("[*] Starting detection diagnostic...\n")

# Initialize detectors
print("[*] Loading detectors...")
try:
    hand_detector = HandDetector(HAND_MODEL_PATH, num_hands=2)
    print("✓ Hand detector loaded successfully")
except Exception as e:
    print(f"✗ Hand detector failed: {e}")
    sys.exit(1)

try:
    face_detector = FaceDetector(FACE_MODEL_PATH)
    print("✓ Face detector loaded successfully")
except Exception as e:
    print(f"✗ Face detector failed: {e}")
    sys.exit(1)

# Open camera
print(f"\n[*] Opening camera {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"✗ Cannot open camera {CAMERA_INDEX}")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
print("✓ Camera opened")

# Test detection
print("\n[*] Testing detection (press 'q' to exit)...\n")

frame_count = 0
hand_detections = 0
face_detections = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✗ Failed to read frame")
            break
        
        frame_count += 1
        
        # Detect hands
        hand_landmarks, handedness = hand_detector.detect(frame)
        if hand_landmarks:
            hand_detections += 1
            frame = hand_detector.draw_hands(frame, hand_landmarks, handedness)
            status_hand = f"HANDS: {len(hand_landmarks)} detected"
        else:
            status_hand = "HANDS: None"
        
        # Detect faces
        face_landmarks = face_detector.detect(frame)
        if face_landmarks:
            face_detections += 1
            frame = face_detector.draw_faces(frame, face_landmarks)
            status_face = f"FACES: {len(face_landmarks)} detected"
        else:
            status_face = "FACES: None"
        
        # Draw status
        cv2.putText(frame, status_hand, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, status_face, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Detection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"\n✗ Error during detection: {e}")
    import traceback
    traceback.print_exc()

finally:
    cap.release()
    cv2.destroyAllWindows()

# Print statistics
print("\n" + "="*60)
print("[*] Detection Test Complete")
print("="*60)
print(f"Frames processed:       {frame_count}")
print(f"Frames with hands:      {hand_detections} ({100*hand_detections//max(frame_count,1)}%)")
print(f"Frames with faces:      {face_detections} ({100*face_detections//max(frame_count,1)}%)")
print("="*60)

if hand_detections > 0:
    print("✓ Hand detection is working!")
else:
    print("⚠ No hands detected - check camera and lighting")

if face_detections > 0:
    print("✓ Face detection is working!")
else:
    print("⚠ No faces detected - check camera and lighting")
