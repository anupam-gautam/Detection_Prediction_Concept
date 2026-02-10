import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Setup the Detector
base_options = python.BaseOptions(
    model_asset_path='models/blaze_face_short_range.tflite'
)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# 2. Open Camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 3. Convert OpenCV BGR to MediaPipe Image object
    # MediaPipe Tasks requires its own Image wrapper
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 4. Perform Detection
    detection_result = detector.detect(mp_image)

    # 5. Process Results (Drawing Bounding Boxes)
    if detection_result.detections:
        for detection in detection_result.detections:
            # Get Bounding Box (Coordinates are in pixels here)
            bbox = detection.bounding_box
            start_point = int(bbox.origin_x), int(bbox.origin_y)
            end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
            
            # Draw Box
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

    cv2.imshow('MediaPipe Tasks Face Detection', frame)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()