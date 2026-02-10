import mediapipe as mp
import cv2
import time


def eye_gaze_ratio(face, iris, inner, outer):
    iris_x  = face[iris].x
    inner_x = face[inner].x
    outer_x = face[outer].x

    dist_to_inner = abs(iris_x - inner_x)
    total_width = abs(outer_x - inner_x)

    return dist_to_inner / total_width



# 1. Setup MediaPipe (Standard Configuration)
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Load the model file you downloaded
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO) # VIDEO mode is simpler than LIVE_STREAM

# 2. Open Webcam
cap = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret: break

        # MediaPipe requires the time in milliseconds
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(time.time() * 1000)

        # 3. Detect Landmarks
        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.face_landmarks:
            # We just take the first face found
            face = result.face_landmarks[0]
            
            left_ratio  = eye_gaze_ratio(face, 473, 362, 263)
            right_ratio = eye_gaze_ratio(face, 468, 133, 33)
            avg_ratio = (left_ratio + right_ratio) / 2

            
            # If ratio is between 0.40 and 0.60, the iris is in the center
            if 0.46 < avg_ratio  < 0.54:
                text = "LOOKING AT SCREEN"
                color = (0, 255, 0) # Green
            else:
                text = "LOOKING AWAY"
                color = (0, 0, 255) # Red

            # Show text on screen
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.imshow("Simple Gaze", frame)
        if cv2.waitKey(1) == 27: break # Press ESC to exit

cap.release()
cv2.destroyAllWindows()


