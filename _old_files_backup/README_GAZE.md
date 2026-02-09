# Gaze Tracking Implementation

## Overview

This project implements real-time eye gaze detection using a standard webcam, based on the methodology described in the article "Eye Gaze Tracking using Camera and OpenCV". The implementation uses **MediaPipe** for face landmark detection and **OpenCV** for 3D-to-2D projection and geometric calculations.

## Key Concepts

### 1. **Face Recognition & Pupil Localization**
- Uses MediaPipe's pre-trained face landmarker to detect 468 2D face landmarks in real-time
- Identifies the nose tip, eye centers, chin, and mouth corners
- Locates pupil positions from the detected eye landmarks

### 2. **3D Face Model**
The system uses a generic 3D face model with 6 reference points (in millimeters):
```
- Nose tip: (0, 0, 0)              - Origin of coordinate system
- Chin: (0, -330, -65)             - Lower face reference
- Left eye center: (-225, 170, -135)
- Right eye center: (225, 170, -135)
- Left mouth corner: (-150, -150, -125)
- Right mouth corner: (150, -150, -125)
```

### 3. **Pinhole Camera Model & Head Pose Estimation**
Uses OpenCV's `solvePnP()` function to determine the head's 3D position and orientation:
- Maps 6 known 3D points (face model) to their 2D projections (detected landmarks)
- Returns rotation vector and translation vector
- These vectors represent how the 3D world is transformed to appear on the 2D image plane

### 4. **2D-to-3D Pupil Projection**
Uses `estimateAffine3D()` to project 2D pupil coordinates back into 3D space:
- Takes 2D image points (x, y, 0) and corresponding 3D model points
- Returns affine transformation matrix
- Estimates where the pupil is located in the 3D model coordinate system

### 5. **Gaze Direction Computation**
Calculates the direction the eye is looking:
```
Gaze Vector = (Pupil 3D - Eye Center 3D) * distance_factor
```
- The `distance_factor` (default 10) is a scaling value since we don't know actual distance from camera
- This vector is then projected back to 2D image space for visualization

### 6. **Head Movement Compensation**
Accounts for head rotation to get a more stable gaze direction:
```
Compensated Gaze = Gaze Vector - Head Pose Vector
```
- Subtracts the head rotation from the raw gaze to remove head movement effects
- Makes the gaze tracker resilient to head movements

## File Structure

```
Sign Language Detection/
├── GazeTracker.py          # Main gaze tracking implementation
├── Detection Project/
│   └── Initial.py          # Integrated gaze tracking with hand/face detection
├── hand_landmarker.task    # MediaPipe hand model
├── face_landmarker.task    # MediaPipe face model
└── README_GAZE.md          # This file
```

## Class: GazeTracker

### Key Methods

#### `__init__(camera_matrix=None)`
Initializes the tracker with:
- 3D face model points
- 3D eye model points
- Camera intrinsic matrix (estimated if not provided)
- Landmark indices for MediaPipe face landmarks

#### `get_face_2d_points(face_landmarks)`
Extracts 2D coordinates of key face landmarks from MediaPipe output

#### `estimate_head_pose(face_2d)`
**Implements: Head Pose Estimation**
- Uses `solvePnP()` to find rotation and translation vectors
- Maps 3D face model to 2D image landmarks
- Returns: (rotation_vec, translation_vec, success_flag)

#### `project_3d_to_2d(points_3d, rotation_vec, translation_vec)`
**Implements: Pinhole Camera Model**
- Projects 3D points onto 2D image plane using head pose
- Uses `projectPoints()` from OpenCV

#### `get_pupil_3d_from_2d(pupil_2d, left_eye_3d, right_eye_3d, is_left_eye)`
**Implements: 2D-to-3D Pupil Projection**
- Estimates 3D position of pupil from 2D image coordinates
- Uses `estimateAffine3D()` for transformation
- Returns: 3D pupil location or None if estimation failed

#### `compute_gaze_direction(pupil_3d, eye_center_3d, distance_magic_number)`
**Implements: Gaze Direction Computation**
- Calculates vector from eye center to pupil
- Scales by magic number (distance estimation)
- Returns: Normalized gaze direction vector

#### `compensate_head_movement(gaze_vector_2d, head_pose_2d, head_pose_magic_number)`
**Implements: Head Movement Compensation**
- Subtracts head pose effect from gaze vector
- Makes tracking more stable across head movements
- Returns: Compensated gaze vector

#### `track_gaze(frame, face_landmarks, image_width, image_height, eye_selection)`
**Main Pipeline:**
1. Extracts 2D face landmarks
2. Estimates head pose using `solvePnP()`
3. Gets 3D eye positions
4. Locates pupils from face landmarks
5. Projects pupil 2D→3D using affine transformation
6. Computes gaze direction
7. Projects gaze direction back to 2D
8. Compensates for head movement
9. Returns gaze point and debug information

#### `draw_gaze_visualization(frame, gaze_2d, debug_info, face_2d, rotation_vec, translation_vec)`
Visualizes:
- 2D face landmarks (green dots)
- Pupil positions (blue dots)
- **Gaze point (red circle)** - where the person is looking
- Head pose axes (X=red, Y=green, Z=blue)
- Center-to-gaze line (yellow)

## Usage

### Basic Usage

```python
from GazeTracker import GazeTracker
import cv2
import mediapipe as mp

# Initialize
gaze_tracker = GazeTracker()
face_landmarker = setup_mediapipe()  # Your MediaPipe setup

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    
    # Detect faces
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    face_result = face_landmarker.detect(mp_image)
    
    if face_result.face_landmarks:
        for face in face_result.face_landmarks:
            # Track gaze
            gaze_2d, debug_info = gaze_tracker.track_gaze(
                frame, face, w, h, eye_selection='left'
            )
            
            # Draw visualization
            gaze_tracker.draw_gaze_visualization(
                frame, gaze_2d, debug_info
            )
    
    cv2.imshow("Gaze Tracking", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Eye Selection Options
- `eye_selection='left'` - Track left eye only
- `eye_selection='right'` - Track right eye only
- `eye_selection='both'` - Average both eyes for smoother tracking

## Parameters & Magic Numbers

The implementation uses a few "magic numbers" for estimation:

### `distance_magic_number` (default: 10)
- Used in `compute_gaze_direction()`
- Represents estimated distance scale (since we don't know real camera distance)
- Can be tuned based on your setup (larger = farther gaze points)

### `head_pose_magic_number` (default: 40)
- Used in `compensate_head_movement()`
- Scaling factor for head pose compensation
- Can be tuned to reduce head movement sensitivity

### Camera Matrix (Intrinsic Parameters)
```python
camera_matrix = [
    [900, 0,   320],  # fx, 0, cx
    [0,   900, 240],  # 0, fy, cy
    [0,   0,   1  ]   # 0, 0,  1
]
```
- These are estimated values for a typical webcam
- For better accuracy, calibrate your specific camera using OpenCV camera calibration

## Accuracy Improvements

The article suggests several ways to improve accuracy:

1. **Camera Calibration**
   - Calibrate your camera properly instead of using estimated matrix
   - Reduces projection errors

2. **Both Eyes**
   - Use both eyes and average the results
   - Current implementation supports this with `eye_selection='both'`

3. **Better Pupil Localization**
   - Use eye structure and pupil position within eye socket
   - More accurate than simple affine transformation

4. **Distance Estimation**
   - Estimate actual distance of subject from camera
   - Would give absolute gaze points instead of just direction

5. **Head-Mounted Solution**
   - Use infrared LEDs and cameras for better stability
   - Eliminate environmental lighting variations

## Troubleshooting

### Gaze Tracking Not Working
- Ensure face is clearly visible to camera
- Check MediaPipe model file paths are correct
- Verify camera has good lighting

### Jittery Gaze Points
- Increase averaging (use `eye_selection='both'`)
- Tune magic numbers based on your setup
- Ensure consistent lighting

### Head Rotation Issues
- Make sure head pose estimation is working (check debug info)
- Verify 3D face model landmarks align with actual face

## Mathematical Foundation

### Pinhole Camera Model
```
s * [u]   [fx  0 cx] [r11 r12 r13 | tx] [X]
    [v] = [ 0 fy cy] [r21 r22 r23 | ty] [Y]
    [1]   [ 0  0  1] [r31 r32 r33 | tz] [Z]
                                         [1]

Where:
- (u,v) = 2D image point
- (X,Y,Z) = 3D world point
- R|t = Rotation and translation (head pose)
- [fx,fy] = Focal lengths
- [cx,cy] = Principal point
- s = Scaling factor
```

### Line-Plane Intersection (Gaze Point Calculation)
The gaze direction and head pose are combined to find where the eye is looking on a 2D plane.

## References

- Original Article: "Eye Gaze Tracking using Camera and OpenCV" by Amit Aflalo
- MediaPipe Documentation: https://ai.google.dev/edge/mediapipe/solutions/guide
- OpenCV Documentation: https://docs.opencv.org/
- Camera Calibration: https://docs.opencv.org/master/d9/d0c/group__calib3d.html

## License

This implementation is provided as-is for educational purposes.
