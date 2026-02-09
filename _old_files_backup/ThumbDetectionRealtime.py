"""
ThumbDetectionRealtime.py

Opens the webcam, runs MediaPipe HandLandmarker on each frame, and overlays
visual feedback when a thumb (thumb tip landmark) is detected.

Requirements satisfied:
- Uses camera preview from CameraInput.py style
- Integrates thumb detection logic from HandDetection_MediaPipe.py
- Runs detection frame-by-frame in real-time
- Displays detection overlay (thumb tip marker + bounding box)

Usage:
    python ThumbDetectionRealtime.py

Press 'q' or ESC to quit.
""" 

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "hand_landmarker.task"  # model asset from project root
CAMERA_INDEX = 0  # default webcam
# Radius of the thumb marker circle
THUMB_MARKER_RADIUS = 8
# Colors in BGR
COLOR_THUMB = (0, 255, 0)  # green
COLOR_BOX = (0, 128, 255)  # orange
COLOR_TEXT = (255, 255, 255)

# -----------------------------
# MediaPipe HandLandmarker setup (create once)
# -----------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

# create detector (this may raise if model file missing)
detector = vision.HandLandmarker.create_from_options(options)


def enhance_frame_clahe(frame, clipLimit=2.0, tileGridSize=(8, 8)):
    """Apply CLAHE to the grayscale version of the frame.

    Returns:
      enhanced_bgr: 3-channel BGR image reconstructed from enhanced grayscale
      enhanced_gray: single-channel enhanced grayscale image (for debug)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    enhanced_gray = clahe.apply(gray)

    # Convert back to BGR so downstream code that expects 3 channels can use it
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    return enhanced_bgr, enhanced_gray


def detect_thumb_in_frame(frame, detector, apply_clahe=True):
    """Run hand detection on the given BGR frame (optionally enhanced) and draw overlays.

    The enhancement step applies CLAHE to the grayscale image and converts it back
    to a 3-channel BGR image which is then converted to RGB for MediaPipe.

    Returns the annotated frame and a boolean indicating if a thumb was found.
    """
    h, w = frame.shape[:2]

    # Optionally enhance the frame prior to detection to improve low-light performance
    if apply_clahe:
        enhanced_bgr, enhanced_gray = enhance_frame_clahe(frame)
    else:
        enhanced_bgr = frame.copy()
        enhanced_gray = None

    # Convert enhanced BGR -> RGB for MediaPipe
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image from the numpy array.
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=enhanced_rgb)
        # mp_image = mp.Image.create_from_array(enhanced_rgb)
    except Exception as e:
        raise RuntimeError("Could not create MediaPipe Image from array: " + str(e))

    # Run detection
    detection_result = detector.detect(mp_image)

    thumb_found = False

    # If no hands detected, overlay status and return frame
    if not detection_result.hand_landmarks:
        status_text = "Thumb detected" if thumb_found else "No thumb"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2)
        return frame, thumb_found

    # There may be multiple hands; process each
    for hand_landmarks in detection_result.hand_landmarks:
        # Thumb tip is landmark index 4 (per MediaPipe hand landmark specification)
        thumb_tip = hand_landmarks[4]
        tx = int(thumb_tip.x * w)
        ty = int(thumb_tip.y * h)

        # Draw a filled circle at the thumb tip on the original color frame
        cv2.circle(frame, (tx, ty), THUMB_MARKER_RADIUS, COLOR_THUMB, -1)

        # Compute bounding box from landmarks for visual feedback
        xs = [int(lm.x * w) for lm in hand_landmarks]
        ys = [int(lm.y * h) for lm in hand_landmarks]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Expand box a bit for visibility
        pad = 10
        x1 = max(x_min - pad, 0)
        y1 = max(y_min - pad, 0)
        x2 = min(x_max + pad, w - 1)
        y2 = min(y_max + pad, h - 1)

        # Draw bounding box and label on the original frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
        cv2.putText(frame, "Thumb", (tx + 10, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)

        thumb_found = True

    # Optionally: overlay status text
    status_text = "Thumb detected" if thumb_found else "No thumb"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2)

    return frame, thumb_found


def main():
    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Mirror the frame for a natural selfie view
        frame = cv2.flip(frame, 1)

        # Detect thumbs and get annotated frame
        try:
            annotated_frame, found = detect_thumb_in_frame(frame, detector)
        except RuntimeError as e:
            print("Runtime error during detection:", e)
            break

        # Show the result
        cv2.imshow("Thumb Detection Realtime", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
