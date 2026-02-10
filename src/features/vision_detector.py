import mediapipe as mp
import cv2
import time
import os

class VisionDetector:
    def __init__(self, model_path='models/face_landmarker.task'):
        self.model_path = model_path
        self.cap = None
        self.landmarker = None
        self.mp_image = None
        
        # Calibration thresholds (from original gaze_detector.py)
        self.GAZE_Threshold_MIN = 0.46
        self.GAZE_Threshold_MAX = 0.54

        self._init_mediapipe()

    def _init_mediapipe(self):
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # check if model exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def _eye_gaze_ratio(self, face, iris, inner, outer):
        iris_x  = face[iris].x
        inner_x = face[inner].x
        outer_x = face[outer].x

        dist_to_inner = abs(iris_x - inner_x)
        total_width = abs(outer_x - inner_x)
        if total_width == 0:
            return 0.5 
        return dist_to_inner / total_width

    def start(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
             print("Error: Could not open camera.")

    def stop(self):
        if self.cap:
            self.cap.release()
        if self.landmarker:
            self.landmarker.close()

    def get_frame_data(self):
        """
        Reads a frame and analyzes it.
        Returns:
            frame: encoding (BGR)
            results: dict with keys:
                - is_face_present: bool
                - is_looking_at_screen: bool
                - gaze_ratio: float
                - face_landmarks: list (optional, for drawing)
        """
        if not self.cap or not self.cap.isOpened():
            return None, {}

        success, frame = self.cap.read()
        if not success:
            return None, {}

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp = int(time.time() * 1000)

        detection_result = self.landmarker.detect_for_video(mp_image, timestamp)

        data = {
            "is_face_present": False,
            "is_looking_at_screen": False,
            "gaze_ratio": 0.5,
            "face_landmarks": None
        }

        if detection_result.face_landmarks:
            data["is_face_present"] = True
            face = detection_result.face_landmarks[0] # Take first face
            data["face_landmarks"] = face

            # Gaze Calculation
            left_ratio  = self._eye_gaze_ratio(face, 473, 362, 263)
            right_ratio = self._eye_gaze_ratio(face, 468, 133, 33)
            avg_ratio = (left_ratio + right_ratio) / 2
            data["gaze_ratio"] = avg_ratio

            if self.GAZE_Threshold_MIN < avg_ratio < self.GAZE_Threshold_MAX:
                data["is_looking_at_screen"] = True
            else:
                data["is_looking_at_screen"] = False
        
        return frame, data
