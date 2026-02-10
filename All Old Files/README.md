# Multi-Detection System

A unified real-time detection system for **hands**, **faces**, and **gaze** using MediaPipe and OpenCV.

## 🎯 Features

- **Hand Detection**: Real-time detection of hands with 21 landmarks per hand
- **Face Detection**: Facial landmark detection with 468 points per face
- **Gaze Tracking**: Eye gaze direction estimation with head pose compensation

All components run efficiently on CPU and can process video streams at ~30 FPS.

## 📋 Requirements

- Python 3.8+
- Webcam or video input
- GPU recommended for better performance (optional)

## 🚀 Quick Start

This section provides the essential steps to get the project up and running quickly. For more detailed installation instructions and troubleshooting, please refer to the [SETUP.md](SETUP.md) guide.

### 1. Setup

```bash
# Navigate to the project directory
cd "Sign Language Detection"

# (Optional) Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

MediaPipe models are required for detection. Download them from the official MediaPipe solutions pages:
- **Hand Landmarker**: [hand_landmarker.task](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- **Face Landmarker**: [face_landmarker.task](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)

Place the downloaded `.task` files into the `models/` directory in your project root.

### 3. Run the Application

```bash
# Execute the main application
python main.py
```
A webcam feed should open, displaying real-time hand, face, and gaze detections. Press `q` or `ESC` to exit the application.

---

## 📁 Project Structure

```
Sign Language Detection/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── src/
│   ├── config.py          # Configuration and settings
│   ├── detectors/         # Detection modules
│   │   ├── __init__.py
│   │   ├── hand_detector.py      # Hand detection
│   │   ├── face_detector.py      # Face detection
│   │   └── gaze_tracker.py       # Gaze tracking
│   └── utils/             # Utility functions
│       ├── visualization.py      # Drawing and visualization
│       └── file_utils.py         # File management
│
├── models/                # Model files (download separately)
│   ├── hand_landmarker.task
│   ├── face_landmarker.task
│   └── blaze_face_short_range.tflite (optional)
│
└── sample_data/           # Sample images/videos for testing
```

## ⚙️ Configuration

Edit `src/config.py` to customize:

- **Camera settings**: Resolution, FPS, camera index
- **Detection parameters**: Confidence thresholds, max detections
- **Visualization options**: Colors, display modes
- **Gaze tracking**: Eye selection, calibration mode

## 🎮 Usage Examples

### Basic Detection
```python
from src.detectors import HandDetector, FaceDetector, GazeTracker
from src.config import *

# Initialize detectors
hand_detector = HandDetector(HAND_MODEL_PATH)
face_detector = FaceDetector(FACE_MODEL_PATH)
gaze_tracker = GazeTracker()

# Process frame
hand_landmarks, handedness = hand_detector.detect(frame)
face_landmarks = face_detector.detect(frame)
gaze_point, gaze_info = gaze_tracker.track_gaze(face_landmarks[0], width, height)
```

### Visualization
```python
# Draw detections
frame = hand_detector.draw_hands(frame, hand_landmarks, handedness)
frame = face_detector.draw_faces(frame, face_landmarks)
frame = gaze_tracker.draw_gaze(frame, gaze_point)
```

## 📊 Key Landmarks

### Hand (21 points per hand)
- Landmarks for: wrist, palm, fingers (thumb, index, middle, ring, pinky)

### Face (468 points per face)
- Key points: eyes, nose, mouth, face contours, cheeks

### Gaze
- Direction vector from eye center through pupil
- Head pose compensated for natural looking gaze

## 🔧 API Reference

### HandDetector
```python
detector = HandDetector(model_path, num_hands=2, confidence_threshold=0.5)

# Detect hands in frame
landmarks, handedness = detector.detect(frame)

# Draw results
frame = detector.draw_hands(frame, landmarks, handedness)

# Get hand properties
centroid = detector.get_hand_centroid(landmarks, hand_idx=0)
```

### FaceDetector
```python
detector = FaceDetector(model_path, confidence_threshold=0.5)

# Detect faces
landmarks = detector.detect(frame)

# Draw results
frame = detector.draw_faces(frame, landmarks)

# Get face properties
key_lms = detector.get_key_landmarks(landmarks[0])
bbox = detector.get_face_bbox(landmarks[0], frame.shape)
```

### GazeTracker
```python
tracker = GazeTracker(camera_matrix, face_3d_model, eye_3d_model)

# Track gaze
gaze_point, gaze_info = tracker.track_gaze(face_landmarks, width, height, eye='both')

# Draw gaze
frame = tracker.draw_gaze(frame, gaze_point)
```

## 🎥 Recording Video Output

Modify `main.py` to save output:
```python
system.run(camera_index=0, output_path='output.mp4')
```

## 🐛 Troubleshooting

### Models Not Found
- Ensure model files are in `models/` directory
- Check file names match exactly
- Download latest models from MediaPipe

### Low FPS
- Reduce `CAMERA_WIDTH` and `CAMERA_HEIGHT` in config
- Close other applications
- Consider using GPU acceleration

### Inaccurate Gaze
- Ensure good lighting conditions
- Camera should be at eye level
- Face should be clearly visible

## 📚 References

- [MediaPipe Solutions](https://developers.google.com/mediapipe)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Eye Gaze Tracking using Camera and OpenCV](https://medium.com/analytics-vidhya/eye-gaze-tracking-using-opencv-b6e6483f7cbf)

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to modify and extend the system for your needs:
- Add new detectors
- Improve accuracy
- Optimize performance
- Create custom visualizations

---

**Last Updated**: February 2026

## 📚 Documentation

This project provides comprehensive documentation to help you understand, set up, and extend the system.

*   **[SETUP.md](SETUP.md)**: Detailed step-by-step installation instructions and in-depth troubleshooting.
*   **[QUICK_START.md](QUICK_START.md)**: A quick reference guide for common tasks, API examples, and configuration tips.
*   **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)**: An overview of the codebase refactoring, new architecture, and removed components.
*   **[COMPLETION_CHECKLIST.md](COMPLETION_CHECKLIST.md)**: A checklist of completed tasks, project status, and code quality metrics.
*   **[INDEX.md](INDEX.md)**: A navigation guide and map of all documentation files in the project.
