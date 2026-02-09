# Gaze Tracking Implementation - Quick Start Guide

## What Was Implemented

A complete eye gaze tracking system based on the article methodology that:

1. ✅ Uses MediaPipe for real-time face landmark detection
2. ✅ Implements 3D face model with nose tip as origin
3. ✅ Uses `solvePnP()` for head pose estimation  
4. ✅ Projects 2D pupils to 3D space using `estimateAffine3D()`
5. ✅ Computes gaze direction vectors
6. ✅ Compensates for head movement
7. ✅ Visualizes gaze points in real-time

---

## Quick Start - Running the Implementation

### Option 1: Test Script (Recommended First)
```bash
cd "d:\Projects\Sign Language Detection"
python GazeTracker_Test.py
```
- Simple standalone gaze tracking demo
- Press 'q' to quit
- Press 's' to switch between left/right/both eyes

### Option 2: Full Integration (With Hand & Face Detection)
```bash
cd "d:\Projects\Sign Language Detection\Detection Project"
python Initial.py
```
- Full pipeline with hand and face detection
- Press 'q' to quit

### Option 3: Advanced Examples
```bash
cd "d:\Projects\Sign Language Detection"
python GazeTracker_Examples.py
```
- Menu-driven examples:
  - Basic gaze tracking
  - Area of Interest (AOI) tracking
  - Left vs Right vs Both eyes comparison

---

## File Structure

```
d:\Projects\Sign Language Detection\
├── GazeTracker.py                  # Main implementation (450+ lines)
├── GazeTracker_Test.py             # Simple test script
├── GazeTracker_Examples.py         # Advanced examples
├── README_GAZE.md                  # Detailed documentation
├── QUICKSTART.md                   # This file
├── Detection Project/
│   └── Initial.py                  # Integrated version (modified)
├── face_landmarker.task            # MediaPipe model
├── hand_landmarker.task            # MediaPipe model
└── data/                           # Data folder
```

---

## How It Works - Core Algorithm

### The Pipeline (5 Main Steps)

1. **Face Detection** (MediaPipe)
   ```
   Input: Video frame
   Output: 468 facial landmarks
   ```

2. **Head Pose Estimation** (solvePnP)
   ```
   Input: 2D landmarks + 3D face model
   Output: Rotation & Translation vectors
   ```

3. **Pupil Localization** (Face landmarks + 2D extraction)
   ```
   Input: Face landmarks
   Output: 2D pupil coordinates in image
   ```

4. **2D→3D Projection** (estimateAffine3D)
   ```
   Input: 2D pupil + head pose
   Output: 3D pupil location in model space
   ```

5. **Gaze Direction + Compensation**
   ```
   Gaze Vector = Pupil 3D - Eye Center 3D
   Compensated = Gaze Vector - Head Rotation Vector
   Output: Final 2D gaze point on screen
   ```

---

## Key Classes and Methods

### GazeTracker
Main class with all gaze tracking functionality

**Key Methods:**
- `track_gaze()` - Main pipeline (returns gaze point)
- `estimate_head_pose()` - Head pose estimation
- `compute_gaze_direction()` - Calculate gaze vector
- `draw_gaze_visualization()` - Visualize results

**Usage:**
```python
from GazeTracker import GazeTracker

tracker = GazeTracker()
gaze_2d, debug_info = tracker.track_gaze(frame, face_landmarks, width, height)
tracker.draw_gaze_visualization(frame, gaze_2d, debug_info)
```

---

## Parameters You Can Tune

### 1. Distance Magic Number (Default: 10)
Controls how far gaze points appear from the eye
```python
# In compute_gaze_direction()
gaze_vector = gaze_vector * distance_magic_number
```
- **Increase** (15-20): Gaze points appear farther away
- **Decrease** (5-8): Gaze points appear closer

### 2. Head Pose Magic Number (Default: 40)
Controls head movement compensation strength
```python
# In compensate_head_movement()
head_pose_normalized = (head_pose / norm) * head_pose_magic_number
```
- **Increase**: More head movement compensation
- **Decrease**: Less head movement compensation

### 3. Camera Matrix
Define your camera's intrinsic parameters
```python
# In __init__()
self.camera_matrix = np.array([
    [900, 0, 320],   # fx, 0, cx
    [0, 900, 240],   # 0, fy, cy
    [0, 0, 1]        # 0, 0, 1
])
```

### 4. Eye Selection
Choose which eye(s) to track:
```python
# 'left' - left eye only
# 'right' - right eye only  
# 'both' - average both eyes (smoother)
gaze_2d, info = tracker.track_gaze(frame, face, w, h, eye_selection='both')
```

---

## Visualization Guide

What you see on screen:

- **Green dots** - Face landmarks (nose, eyes, chin, mouth)
- **Blue dots** - Pupil positions
- **Red circle** - Gaze point (where person is looking) ⭐
- **Yellow line** - Line from center to gaze point
- **Colored axes** - Head pose (X=red, Y=green, Z=blue)

---

## Troubleshooting

### Problem: "No faces detected"
- **Solution**: Ensure face is clearly visible to camera
- Check lighting conditions (avoid backlighting)
- Position face 0.5-1.5 meters from camera

### Problem: Gaze points are jittery
- **Solution 1**: Use `eye_selection='both'` for averaging
- **Solution 2**: Reduce `distance_magic_number` 
- **Solution 3**: Tune `head_pose_magic_number`
- **Solution 4**: Check camera stability (no movement)

### Problem: Gaze tracking is inaccurate
- **Solution 1**: Calibrate camera (see Calibration section)
- **Solution 2**: Increase `distance_magic_number`
- **Solution 3**: Verify MediaPipe model paths are correct
- **Solution 4**: Check if face model is properly loaded

### Problem: Model file not found
- **Solution**: Ensure `face_landmarker.task` is in the correct directory
- Run from: `d:\Projects\Sign Language Detection\`
- Or update `FACE_LANDMARK_MODEL` path in code

---

## Camera Calibration (Optional - For Better Accuracy)

To get accurate gaze tracking, you should calibrate your camera:

```python
import cv2

# Use OpenCV camera calibration
# See: https://docs.opencv.org/master/d9/d0c/group__calib3d.html

# After calibration, you'll have:
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.array([k1, k2, p1, p2, k3])

# Pass to GazeTracker:
tracker = GazeTracker(camera_matrix=camera_matrix)
```

---

## Performance Tips

1. **Frame Resolution**
   - 1280x720: Good balance (default)
   - 640x480: Faster but less accurate
   - 1920x1080: More accurate but slower

2. **Processing Optimization**
   - Use `eye_selection='left'` for faster processing
   - Use `both` for production quality

3. **Head Movement**
   - Keep head relatively still for best results
   - System compensates for small movements

4. **Lighting**
   - Use consistent, uniform lighting
   - Avoid shadows on face
   - Face should be well-lit

---

## Advanced Usage - Integration

### Using in Your Own Code

```python
from GazeTracker import GazeTracker
import cv2
import mediapipe as mp

# Setup
tracker = GazeTracker()
face_detector = setup_mediapipe()  # Your setup
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Detect faces
    faces = face_detector.detect(frame)
    
    # Track gaze for each face
    for face in faces:
        gaze_point, info = tracker.track_gaze(
            frame, face, frame.shape[1], frame.shape[0]
        )
        
        if gaze_point is not None:
            # Do something with gaze point
            x, y = int(gaze_point[0]), int(gaze_point[1])
            cv2.circle(frame, (x, y), 10, (0, 0, 255), 2)
    
    cv2.imshow("Gaze Tracking", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Next Steps - Improvements

The article mentions several improvements for better accuracy:

1. **Camera Calibration** - Calibrate your specific camera
2. **Both Eyes** - Use both eyes and average (already supported!)
3. **Better Pupil Localization** - Use eye structure analysis
4. **Distance Estimation** - Estimate actual distance from camera
5. **Head-Mounted Solution** - Use infrared for stability

These are excellent areas for further development!

---

## References

- **Original Article**: "Eye Gaze Tracking using Camera and OpenCV" by Amit Aflalo
- **GitHub**: https://github.com/amitt1236/Gaze_estimation
- **MediaPipe**: https://ai.google.dev/edge/mediapipe/solutions/guide
- **OpenCV**: https://docs.opencv.org/
- **Pinhole Camera Model**: https://en.wikipedia.org/wiki/Pinhole_camera_model

---

## Support & Debugging

### Enable Debug Info
```python
gaze_2d, debug_info = tracker.track_gaze(frame, face, w, h)

# Check what's available:
print(f"Success: {debug_info['success']}")
print(f"Head Rotation: {debug_info['head_rotation']}")
print(f"Gaze Direction: {debug_info['gaze_direction']}")
print(f"Left Pupil: {debug_info['left_pupil_2d']}")
print(f"Right Pupil: {debug_info['right_pupil_2d']}")
```

### Check MediaPipe Landmarks
```python
# Number of landmarks detected
print(f"Landmarks: {len(face_landmarks)}")

# Access specific landmarks
nose_tip = face_landmarks[1]
print(f"Nose: ({nose_tip.x}, {nose_tip.y})")
```

---

## License & Attribution

This implementation is based on the methodology from:
**"Eye Gaze Tracking using Camera and OpenCV"** by Amit Aflalo

Implementation provided as-is for educational purposes.

---

**Happy Gaze Tracking! 👀**
