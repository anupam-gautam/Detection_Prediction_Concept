import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
import os

# Add parent directory to path to import GazeTracker
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GazeTracker import GazeTracker

# ----------------- Paths -----------------
HAND_MODEL_PATH = "./hand_landmarker.task"
FACE_LANDMARK_MODEL = "./face_landmarker.task"  # MediaPipe Face Landmarker model
CAMERA_INDEX = 0

# ----------------- Hand Detector (IMAGE mode) -----------------
hand_base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options,
    num_hands=2
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# ----------------- Face Landmark Detector (IMAGE mode) -----------------
face_base_options = python.BaseOptions(model_asset_path=FACE_LANDMARK_MODEL)
face_landmark_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    running_mode=vision.RunningMode.IMAGE  # IMAGE mode for per-frame detection
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_landmark_options)

# ----------------- Gaze Tracker -----------------
gaze_tracker = GazeTracker()


# ----------------- Drawing Function -----------------
def draw_hands_and_faces(frame, hand_detector, face_landmarker):
    h, w = frame.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # ---- Hands ----
    hand_result = hand_detector.detect(mp_image)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9,10), (10,11), (11,12),
        (0,13), (13,14), (14,15), (15,16),
        (0,17), (17,18), (18,19), (19,20)
    ]
    if hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            points = [(int(lm.x*w), int(lm.y*h)) for lm in hand]
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, points[start], points[end], (0,255,0), 2)
            for idx, (x,y) in enumerate(points):
                cv2.circle(frame, (x,y), 5, (0,0,255), -1)
                cv2.putText(frame, str(idx), (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

    # ---- Face Landmarks ----
    face_result = face_landmarker.detect(mp_image)
    if face_result.face_landmarks:
        for face in face_result.face_landmarks:
            for idx, lm in enumerate(face):
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 2, (0,255,255), -1)
            
            # ---- Gaze Tracking ----
            gaze_2d, debug_info = gaze_tracker.track_gaze(
                frame, face, w, h, eye_selection='left'
            )
            
            # Draw gaze visualization
            face_2d = gaze_tracker.get_face_2d_points(face)
            face_2d[:, 0] *= w
            face_2d[:, 1] *= h
            
            rotation_vec = debug_info.get('head_rotation')
            translation_vec = debug_info.get('head_translation')
            
            gaze_tracker.draw_gaze_visualization(
                frame, gaze_2d, debug_info, face_2d,
                rotation_vec, translation_vec
            )

    cv2.imshow("Hand + Face", frame)

# ----------------- Main Loop -----------------
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw_hands_and_faces(frame, hand_detector, face_landmarker)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
