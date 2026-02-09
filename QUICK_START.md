# Quick Reference Guide

## 🎯 Common Tasks

### Run the Application
```bash
python main.py
```
Press `q` or `ESC` to exit.

### Enable/Disable Detections
Edit `main.py`:
```python
system = MultiDetectionSystem(
    enable_hand=True,      # Hand detection
    enable_face=True,      # Face detection
    enable_gaze=True       # Gaze tracking
)
```

### Adjust Camera Settings
Edit `src/config.py`:
```python
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_INDEX = 0  # 0 = default, 1 = external, etc.
```

### Change Detection Confidence
Edit `src/config.py`:
```python
HAND_CONFIDENCE_THRESHOLD = 0.5     # 0-1, higher = stricter
FACE_CONFIDENCE_THRESHOLD = 0.5     # 0-1, higher = stricter
```

### Customize Colors
Edit `src/config.py`:
```python
COLOR_HAND = (0, 255, 0)        # Green (BGR)
COLOR_FACE = (255, 0, 0)        # Blue (BGR)
COLOR_GAZE = (0, 255, 255)      # Cyan (BGR)
```

### Save Video Output
Edit `main.py`:
```python
system.run(camera_index=0, output_path='output.mp4')
```

---

## 📚 File Reference

### Core Files
| File | Purpose |
|------|---------|
| `main.py` | Application entry point |
| `src/config.py` | Configuration settings |
| `src/detectors/hand_detector.py` | Hand detection |
| `src/detectors/face_detector.py` | Face detection |
| `src/detectors/gaze_tracker.py` | Gaze tracking |
| `src/utils/visualization.py` | Drawing utilities |

### Documentation
| File | Content |
|------|---------|
| `README.md` | Complete guide |
| `SETUP.md` | Installation steps |
| `REFACTORING_SUMMARY.md` | What changed |

---

## 🔌 API Quick Examples

### Using HandDetector
```python
from src.detectors import HandDetector
from src.config import HAND_MODEL_PATH

detector = HandDetector(HAND_MODEL_PATH)
landmarks, handedness = detector.detect(frame)
frame = detector.draw_hands(frame, landmarks, handedness)
```

### Using FaceDetector
```python
from src.detectors import FaceDetector
from src.config import FACE_MODEL_PATH

detector = FaceDetector(FACE_MODEL_PATH)
landmarks = detector.detect(frame)
frame = detector.draw_faces(frame, landmarks)
```

### Using GazeTracker
```python
from src.detectors import GazeTracker
from src.config import *

tracker = GazeTracker()
gaze_point, gaze_info = tracker.track_gaze(face_landmarks[0], width, height)
frame = tracker.draw_gaze(frame, gaze_point)
```

---

## 🐛 Troubleshooting

### Problem: ModuleNotFoundError
```bash
pip install -r requirements.txt
```

### Problem: Models not found
```bash
# Check models directory
ls models/
# Should contain: hand_landmarker.task, face_landmarker.task
```

### Problem: Camera not opening
- Check camera index (try 0, 1, 2...)
- Ensure no other application is using the camera
- Try different USB port

### Problem: Low FPS
- Reduce resolution: `CAMERA_WIDTH = 640`, `CAMERA_HEIGHT = 480`
- Disable gaze tracking: `enable_gaze=False`
- Close other applications

### Problem: Inaccurate detections
- Check lighting conditions
- Ensure face/hands are clearly visible
- Adjust confidence thresholds in config

---

## 📊 Performance Tips

| Setting | Impact |
|---------|--------|
| Lower resolution | ⚡ Faster |
| Fewer detections | ⚡ Faster |
| Disable gaze | ⚡ Faster |
| Lower FPS target | ⚡ Faster |
| GPU acceleration | ⚡⚡ Much faster |

---

## 🎨 Visualization Options

### Show/Hide Elements
Edit `src/config.py`:
```python
SHOW_FPS = True                    # FPS counter
SHOW_HAND_LANDMARKS = True         # Hand details
SHOW_FACE_LANDMARKS = True         # Face details
SHOW_GAZE_VECTOR = True            # Gaze direction
```

### Adjust Display
```python
# In src/utils/visualization.py
# Modify draw functions for custom layouts
```

---

## 📁 Directory Guide

```
models/              ← Download MediaPipe models here
sample_data/         ← Place test images here
src/                 ← Main source code
  detectors/         ← Detection modules
  utils/             ← Helper functions
  config.py          ← Settings
_old_files_backup/   ← Old files for reference
```

---

## 🔗 Resources

- **MediaPipe**: https://developers.google.com/mediapipe
- **OpenCV**: https://docs.opencv.org/
- **Python Docs**: https://docs.python.org/3/

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `ESC` | Quit |

---

## 📞 Getting Help

1. Check **README.md** for detailed documentation
2. Review code comments in `src/`
3. See **SETUP.md** for installation issues
4. Check **REFACTORING_SUMMARY.md** for architecture

---

**Last Updated**: February 9, 2026  
**Version**: 1.0.0
