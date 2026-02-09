"""
Configuration file for Detection System
Centralized settings for hand, face, and gaze detection
"""

import numpy as np
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

# Model files
HAND_MODEL_PATH = str(MODELS_DIR / "hand_landmarker.task")
FACE_MODEL_PATH = str(MODELS_DIR / "face_landmarker.task")
FACE_DETECTOR_MODEL = str(MODELS_DIR / "blaze_face_short_range.tflite")

# ============================================================================
# CAMERA CONFIGURATION
# ============================================================================
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Camera intrinsic parameters (estimated values)
# For better accuracy, calibrate your camera and replace these values
FOCAL_LENGTH_X = 900.0      # fx in pixels
FOCAL_LENGTH_Y = 900.0      # fy in pixels
PRINCIPAL_POINT_X = 640.0   # cx in pixels
PRINCIPAL_POINT_Y = 360.0   # cy in pixels

# Lens distortion coefficients
DIST_K1 = 0.0
DIST_K2 = 0.0
DIST_P1 = 0.0
DIST_P2 = 0.0

# ============================================================================
# HAND DETECTION CONFIGURATION
# ============================================================================
MAX_HANDS = 2
HAND_CONFIDENCE_THRESHOLD = 0.5

# ============================================================================
# FACE DETECTION CONFIGURATION
# ============================================================================
FACE_CONFIDENCE_THRESHOLD = 0.5

# ============================================================================
# GAZE TRACKING CONFIGURATION
# ============================================================================

# Generic 3D face model points (in mm, relative to nose tip as origin)
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],           # Nose tip (origin)
    [0.0, -330.0, -65.0],      # Chin
    [-225.0, 170.0, -135.0],   # Left eye center
    [225.0, 170.0, -135.0],    # Right eye center
    [-150.0, -150.0, -125.0],  # Left mouth corner
    [150.0, -150.0, -125.0],   # Right mouth corner
], dtype=np.float32)

# 3D eye model points (relative to eye center)
EYE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],           # Eye center
    [0.0, -20.0, -30.0],       # Eyebrow
    [-20.0, -15.0, -30.0],     # Left eye corner
    [20.0, -15.0, -30.0],      # Right eye corner
    [-20.0, -50.0, -30.0],     # Lower left
    [20.0, -50.0, -30.0],      # Lower right
], dtype=np.float32)

# Gaze tracking mode: 'direction' or 'calibrated'
GAZE_MODE = 'direction'

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
# Colors in BGR format
COLOR_HAND = (0, 255, 0)        # Green
COLOR_FACE = (255, 0, 0)        # Blue
COLOR_GAZE = (0, 255, 255)      # Cyan
COLOR_TEXT = (255, 255, 255)    # White
COLOR_BACKGROUND = (30, 30, 30) # Dark gray

# Display options
SHOW_FPS = True
SHOW_HAND_LANDMARKS = True
SHOW_FACE_LANDMARKS = True
SHOW_GAZE_VECTOR = True

# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================
WINDOW_NAME = "Multi-Detection System (Hand, Face, Gaze)"
EXIT_KEYS = [ord('q'), 27]  # 'q' or ESC
