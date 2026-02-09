import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "data"
MODEL_PATH = "hand_landmarker.task"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

# -----------------------------
# MediaPipe setup (do this ONCE)
# -----------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

# -----------------------------
# Loop over images
# -----------------------------
for filename in os.listdir(DATA_DIR):

    if not filename.lower().endswith(VALID_EXTENSIONS):
        continue

    image_path = os.path.join(DATA_DIR, filename)
    print(f"\nProcessing: {image_path}")

    # Load image with OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not load image, skipping")
        continue

    h, w = img.shape[:2]

    # Load image for MediaPipe
    mp_image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(mp_image)

    print("Number of hands:", len(detection_result.hand_landmarks))

    # Print handedness
    for i, handedness in enumerate(detection_result.handedness):
        print(f"Hand {i}: {handedness[0].category_name}")

    # Draw thumb tip for each detected hand
    for hand_landmarks in detection_result.hand_landmarks:
        x = int(hand_landmarks[4].x * w)
        y = int(hand_landmarks[4].y * h)
        cv2.circle(img, (x, y), 10, (0, 255, 0), -1)

    # Show result
    cv2.imshow("Hand Detection", img)

    key = cv2.waitKey(0)
    if key == 27:  # ESC to exit early
        break

cv2.destroyAllWindows()
