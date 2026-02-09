"""
Test MediaPipe hand detection directly
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

MODEL_PATH = "models/hand_landmarker.task"

# Initialize
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

# Test with camera
cap = cv2.VideoCapture(0)

print("Capturing frame...")
ret, frame = cap.read()
cap.release()

if ret:
    import mediapipe as mp_image_lib
    mp_image = mp_image_lib.Image(image_format=mp_image_lib.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    result = detector.detect(mp_image)
    
    print(f"\nDetection result type: {type(result)}")
    print(f"Result attributes: {[x for x in dir(result) if not x.startswith('_')]}")
    
    print(f"\nhand_landmarks type: {type(result.hand_landmarks)}")
    print(f"hand_landmarks length: {len(result.hand_landmarks)}")
    
    print(f"\nhandedness type: {type(result.handedness)}")
    print(f"handedness length: {len(result.handedness)}")
    
    if result.handedness:
        print(f"\nFirst handedness item type: {type(result.handedness[0])}")
        print(f"First handedness item: {result.handedness[0]}")
        print(f"First handedness attributes: {[x for x in dir(result.handedness[0]) if not x.startswith('_')]}")
        
        # Try to get category
        if hasattr(result.handedness[0], '__iter__'):
            print(f"First handedness is iterable, length: {len(result.handedness[0])}")
            if len(result.handedness[0]) > 0:
                print(f"First item in handedness[0]: {result.handedness[0][0]}")
                if hasattr(result.handedness[0][0], 'category_name'):
                    print(f"Category name: {result.handedness[0][0].category_name}")
