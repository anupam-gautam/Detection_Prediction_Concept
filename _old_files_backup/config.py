"""
Gaze Tracker Configuration File
Centralized parameters for easy tuning
"""

import numpy as np

# ============================================================================
# CAMERA CONFIGURATION
# ============================================================================

# Camera index (0 = default webcam, 1 = external camera, etc.)
CAMERA_INDEX = 0

# Target resolution (higher = more accurate but slower)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Camera intrinsic parameters (estimated values)
# For better accuracy, calibrate your camera and replace these values
FOCAL_LENGTH_X = 900.0      # fx in pixels
FOCAL_LENGTH_Y = 900.0      # fy in pixels
PRINCIPAL_POINT_X = 320.0   # cx in pixels
PRINCIPAL_POINT_Y = 240.0   # cy in pixels

# Lens distortion coefficients (assumed minimal)
DIST_K1 = 0.0
DIST_K2 = 0.0
DIST_P1 = 0.0
DIST_P2 = 0.0
DIST_K3 = 0.0

# ============================================================================
# GAZE TRACKING PARAMETERS
# ============================================================================

# Distance estimation magic number
# Controls how far gaze points appear from the eye
# Range: 5-20 (higher = gaze points farther from face)
# Default: 10
DISTANCE_MAGIC_NUMBER = 10

# Head pose compensation magic number
# Controls head movement compensation strength
# Range: 20-60 (higher = more compensation)
# Default: 40
HEAD_POSE_MAGIC_NUMBER = 40

# Eye selection for gaze tracking
# Options: 'left', 'right', 'both'
# 'left' - faster, less accurate
# 'right' - faster, less accurate
# 'both' - slower, more accurate (averaged)
EYE_SELECTION = 'both'

# ============================================================================
# 3D FACE MODEL
# ============================================================================

# Generic 3D face model points (in millimeters)
# Relative to nose tip as origin
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],           # 0: Nose tip (origin)
    [0.0, -330.0, -65.0],      # 1: Chin
    [-225.0, 170.0, -135.0],   # 2: Left eye center
    [225.0, 170.0, -135.0],    # 3: Right eye center
    [-150.0, -150.0, -125.0],  # 4: Left mouth corner
    [150.0, -150.0, -125.0],   # 5: Right mouth corner
], dtype=np.float32)

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Draw face landmarks
DRAW_FACE_LANDMARKS = True
FACE_LANDMARK_COLOR = (0, 255, 255)  # Cyan
FACE_LANDMARK_RADIUS = 2

# Draw pupil positions
DRAW_PUPILS = True
PUPIL_COLOR = (255, 0, 0)  # Blue
PUPIL_RADIUS = 2

# Draw gaze point (main result)
DRAW_GAZE_POINT = True
GAZE_POINT_COLOR = (0, 0, 255)  # Red
GAZE_POINT_RADIUS = 8
GAZE_POINT_INNER_RADIUS = 3

# Draw gaze line from center to gaze point
DRAW_GAZE_LINE = True
GAZE_LINE_COLOR = (0, 255, 255)  # Yellow
GAZE_LINE_THICKNESS = 2

# Draw head pose axes (3D visualization)
DRAW_HEAD_POSE_AXES = True
HEAD_POSE_AXIS_LENGTH = 50

# Draw status text
DRAW_STATUS_TEXT = True
STATUS_TEXT_COLOR = (0, 255, 0)  # Green
STATUS_TEXT_SCALE = 0.7
STATUS_TEXT_THICKNESS = 2

# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================

# Maximum history for gaze smoothing
GAZE_HISTORY_SIZE = 30

# Frame processing mode
# 'fast' - process every frame
# 'balanced' - process every 2 frames
# 'quality' - process every frame with optimization
PROCESSING_MODE = 'balanced'

# Enable performance monitoring
MONITOR_PERFORMANCE = True

# ============================================================================
# AREA OF INTEREST (AOI) CONFIGURATION
# ============================================================================

# Enable AOI tracking
ENABLE_AOI_TRACKING = False

# AOI definitions (x_min, y_min, x_max, y_max)
# Normalized to (0-1) range
AOIS = {
    'top_left': ((0.0, 0.0), (0.33, 0.33)),
    'top_center': ((0.33, 0.0), (0.67, 0.33)),
    'top_right': ((0.67, 0.0), (1.0, 0.33)),
    'center': ((0.33, 0.33), (0.67, 0.67)),
    'bottom_left': ((0.0, 0.67), (0.33, 1.0)),
    'bottom_center': ((0.33, 0.67), (0.67, 1.0)),
    'bottom_right': ((0.67, 0.67), (1.0, 1.0)),
}

# ============================================================================
# MediaPipe CONFIGURATION
# ============================================================================

# Path to face landmarker model
FACE_LANDMARK_MODEL_PATH = "./face_landmarker.task"

# Path to hand landmarker model (for Initial.py)
HAND_LANDMARK_MODEL_PATH = "./hand_landmarker.task"

# MediaPipe detection mode
# Options: 'IMAGE', 'VIDEO', 'LIVE_STREAM'
MEDIAPIPE_MODE = 'IMAGE'

# ============================================================================
# DEBUG CONFIGURATION
# ============================================================================

# Enable debug output to console
DEBUG_MODE = True

# Print debug info every N frames
DEBUG_FRAME_INTERVAL = 30

# Save debug information to file
SAVE_DEBUG_LOG = False
DEBUG_LOG_PATH = "./gaze_debug.log"

# ============================================================================
# CALIBRATION CONFIGURATION
# ============================================================================

# Enable calibration mode
CALIBRATION_MODE = False

# Calibration points for multi-point calibration
# Used to adjust distance and head pose magic numbers
CALIBRATION_POINTS = [
    (0.5, 0.5),    # Center
    (0.1, 0.1),    # Top-left
    (0.9, 0.1),    # Top-right
    (0.1, 0.9),    # Bottom-left
    (0.9, 0.9),    # Bottom-right
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_camera_matrix():
    """Get camera intrinsic matrix from configuration"""
    return np.array([
        [FOCAL_LENGTH_X, 0, PRINCIPAL_POINT_X],
        [0, FOCAL_LENGTH_Y, PRINCIPAL_POINT_Y],
        [0, 0, 1]
    ], dtype=np.float32)


def get_distortion_coeffs():
    """Get distortion coefficients from configuration"""
    return np.array([
        [DIST_K1, DIST_K2, DIST_P1, DIST_P2, DIST_K3]
    ], dtype=np.float32)


def print_configuration():
    """Print current configuration to console"""
    print("\n" + "="*60)
    print("GAZE TRACKING CONFIGURATION")
    print("="*60)
    
    print("\nCamera:")
    print(f"  Camera Index: {CAMERA_INDEX}")
    print(f"  Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"  Focal Length: ({FOCAL_LENGTH_X}, {FOCAL_LENGTH_Y})")
    print(f"  Principal Point: ({PRINCIPAL_POINT_X}, {PRINCIPAL_POINT_Y})")
    
    print("\nGaze Tracking:")
    print(f"  Distance Magic Number: {DISTANCE_MAGIC_NUMBER}")
    print(f"  Head Pose Magic Number: {HEAD_POSE_MAGIC_NUMBER}")
    print(f"  Eye Selection: {EYE_SELECTION}")
    
    print("\nVisualization:")
    print(f"  Draw Face Landmarks: {DRAW_FACE_LANDMARKS}")
    print(f"  Draw Pupils: {DRAW_PUPILS}")
    print(f"  Draw Gaze Point: {DRAW_GAZE_POINT}")
    print(f"  Draw Head Pose Axes: {DRAW_HEAD_POSE_AXES}")
    
    print("\nPerformance:")
    print(f"  Gaze History Size: {GAZE_HISTORY_SIZE}")
    print(f"  Processing Mode: {PROCESSING_MODE}")
    print(f"  Monitor Performance: {MONITOR_PERFORMANCE}")
    
    print("\nDebug:")
    print(f"  Debug Mode: {DEBUG_MODE}")
    print(f"  Debug Frame Interval: {DEBUG_FRAME_INTERVAL}")
    print(f"  Save Debug Log: {SAVE_DEBUG_LOG}")
    
    print("="*60 + "\n")


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

class Presets:
    """Predefined configurations for different scenarios"""
    
    @staticmethod
    def fast_tracking():
        """Optimized for speed (16 eye, lower quality)"""
        return {
            'EYE_SELECTION': 'left',
            'DISTANCE_MAGIC_NUMBER': 8,
            'HEAD_POSE_MAGIC_NUMBER': 30,
            'DRAW_HEAD_POSE_AXES': False,
            'GAZE_HISTORY_SIZE': 10,
        }
    
    @staticmethod
    def balanced():
        """Balanced speed and quality (both eyes, medium quality)"""
        return {
            'EYE_SELECTION': 'both',
            'DISTANCE_MAGIC_NUMBER': 10,
            'HEAD_POSE_MAGIC_NUMBER': 40,
            'DRAW_HEAD_POSE_AXES': True,
            'GAZE_HISTORY_SIZE': 20,
        }
    
    @staticmethod
    def high_quality():
        """Optimized for accuracy (both eyes, high quality)"""
        return {
            'EYE_SELECTION': 'both',
            'DISTANCE_MAGIC_NUMBER': 12,
            'HEAD_POSE_MAGIC_NUMBER': 50,
            'DRAW_HEAD_POSE_AXES': True,
            'GAZE_HISTORY_SIZE': 30,
        }
    
    @staticmethod
    def calibration():
        """Configuration for camera calibration"""
        return {
            'CALIBRATION_MODE': True,
            'DEBUG_MODE': True,
            'DEBUG_FRAME_INTERVAL': 5,
            'DRAW_FACE_LANDMARKS': True,
            'DRAW_PUPILS': True,
            'DRAW_HEAD_POSE_AXES': True,
        }


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_configuration():
    """Validate configuration parameters"""
    
    errors = []
    
    # Camera parameters
    if CAMERA_WIDTH <= 0 or CAMERA_HEIGHT <= 0:
        errors.append("Camera resolution must be positive")
    
    if FOCAL_LENGTH_X <= 0 or FOCAL_LENGTH_Y <= 0:
        errors.append("Focal length must be positive")
    
    # Gaze tracking parameters
    if DISTANCE_MAGIC_NUMBER <= 0:
        errors.append("Distance magic number must be positive")
    
    if HEAD_POSE_MAGIC_NUMBER <= 0:
        errors.append("Head pose magic number must be positive")
    
    if EYE_SELECTION not in ['left', 'right', 'both']:
        errors.append("Eye selection must be 'left', 'right', or 'both'")
    
    if PROCESSING_MODE not in ['fast', 'balanced', 'quality']:
        errors.append("Processing mode must be 'fast', 'balanced', or 'quality'")
    
    if GAZE_HISTORY_SIZE <= 0:
        errors.append("Gaze history size must be positive")
    
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  ❌ {error}")
        return False
    
    print("✅ Configuration is valid")
    return True


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print_configuration()
    validate_configuration()
    print("\nTo use this configuration in your code:")
    print("  from config import *")
    print("  or")
    print("  import config")
    print("  camera_matrix = config.get_camera_matrix()")
