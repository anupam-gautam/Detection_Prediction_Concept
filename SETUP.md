# Setup & Installation Guide

## Prerequisites

- Python 3.8 or later
- pip package manager
- Webcam (for real-time detection)
- ~500MB free disk space (for models)

## Step-by-Step Installation

### 1. Clone/Navigate to Project

```bash
cd "d:\Projects\Sign Language Detection"
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `opencv-python` - Computer vision library
- `mediapipe` - ML framework for perception tasks
- `numpy` - Numerical computing

### 4. Download MediaPipe Models

Models are required for detection. Download from official sources:

#### Option A: Download from MediaPipe (Recommended)

1. **Hand Landmarker** (~20 MB)
   - URL: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
   - File: `hand_landmarker.task`

2. **Face Landmarker** (~20 MB)
   - URL: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
   - File: `face_landmarker.task`

3. **Face Detection** (Optional, ~380 KB)
   - File: `blaze_face_short_range.tflite`

#### Option B: Using Python Script

Create a `download_models.py` script:

```python
import urllib.request
import os

models = {
    'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task',
    'face_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker.task',
}

os.makedirs('models', exist_ok=True)

for filename, url in models.items():
    filepath = os.path.join('models', filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"✓ Downloaded {filename}")
    else:
        print(f"✓ {filename} already exists")
```

Run it:
```bash
python download_models.py
```

### 5. Verify Installation

```bash
# Check if models are in place
python -c "from src.config import HAND_MODEL_PATH, FACE_MODEL_PATH; print('Models configured at:', HAND_MODEL_PATH, FACE_MODEL_PATH)"

# Test imports
python -c "from src.detectors import HandDetector, FaceDetector, GazeTracker; print('✓ All modules imported successfully')"
```

### 6. Run Application

```bash
python main.py
```

You should see:
- Camera feed with detected hands, faces, and gaze direction
- FPS counter in top-left corner
- Detection status

Press `q` or `ESC` to exit.

## Configuration

### Adjust Settings

Edit `src/config.py`:

```python
# Camera settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Detection thresholds
HAND_CONFIDENCE_THRESHOLD = 0.5
FACE_CONFIDENCE_THRESHOLD = 0.5

# Display options
SHOW_FPS = True
SHOW_HAND_LANDMARKS = True
SHOW_FACE_LANDMARKS = True
SHOW_GAZE_VECTOR = True
```

### Disable Specific Detections

In `main.py`:

```python
# Run with only hand and face (no gaze)
system = MultiDetectionSystem(
    enable_hand=True,
    enable_face=True,
    enable_gaze=False
)
```

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'mediapipe'

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Model files not found

**Solution**: Ensure models are downloaded to `models/` directory
```bash
# Check models exist
ls models/
# Should show: hand_landmarker.task  face_landmarker.task
```

### Issue: Camera not opening

**Solution**: Check camera availability
```python
import cv2
cap = cv2.VideoCapture(0)
print(cap.isOpened())  # Should be True
```

### Issue: Very low FPS

**Solution**: Reduce resolution or disable some detections
- Edit `CAMERA_WIDTH` and `CAMERA_HEIGHT` in config
- Set `enable_gaze=False` if not needed
- Ensure good lighting

### Issue: Inaccurate gaze direction

**Solution**: 
- Ensure good lighting
- Face should be clearly visible
- Head should be in upright position
- Camera calibration might help (advanced)

## Next Steps

1. **Explore the code** in `src/detectors/` to understand implementation
2. **Customize visualization** in `src/utils/visualization.py`
3. **Add features** like recording, image processing, etc.
4. **Integrate with your app** using the provided API

## Advanced Setup

### GPU Acceleration

For faster processing with NVIDIA GPU:

```bash
# Install GPU-optimized MediaPipe
pip install mediapipe-gpu

# Or use OpenCV GPU
pip install opencv-contrib-python
```

### For Development

Install development tools:
```bash
pip install -r requirements.txt
pip install black pylint pytest  # Code quality tools
```

### Camera Calibration

For improved accuracy, calibrate your camera:

```python
# Create calibration script to find camera matrix
# This involves detecting checkerboard patterns in images
# See OpenCV documentation for calibration procedure
```

## Useful Commands

```bash
# List installed packages
pip list

# Check specific package version
pip show mediapipe

# Upgrade packages
pip install --upgrade mediapipe opencv-python

# Deactivate virtual environment
deactivate
```

## Getting Help

1. Check `README.md` for API documentation
2. Review code comments in `src/detectors/`
3. Check MediaPipe [official documentation](https://developers.google.com/mediapipe)
4. Search for issues/solutions online

---

**Setup Complete!** You're ready to use the Multi-Detection System.
