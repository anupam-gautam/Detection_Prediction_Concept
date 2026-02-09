"""
Gaze Tracking Implementation based on MediaPipe Face Landmarks
Article: Eye Gaze Tracking using Camera and OpenCV

This module implements gaze tracking by:
1. Using MediaPipe to detect face landmarks and pupil locations
2. Using a 3D face model to estimate head pose
3. Using affine transformation to project 2D pupil to 3D space
4. Computing gaze direction relative to head pose
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class GazeTracker:
    """
    Tracks eye gaze direction using a generic 3D face model and pupil localization.
    
    Key components:
    - Head pose estimation using solvePnP
    - Pupil 3D localization using affine transformation
    - Gaze direction computation accounting for head movement
    """
    
    def __init__(self, camera_matrix: Optional[np.ndarray] = None):
        """
        Initialize the GazeTracker.
        
        Args:
            camera_matrix: Optional camera intrinsic matrix. If None, uses estimated values.
        """
        # Generic 3D face model points (in mm)
        # Relative to nose tip as origin
        self.face_3d_model = np.array([
            [0.0, 0.0, 0.0],           # Nose tip (origin)
            [0.0, -330.0, -65.0],      # Chin
            [-225.0, 170.0, -135.0],   # Left eye center
            [225.0, 170.0, -135.0],    # Right eye center
            [-150.0, -150.0, -125.0],  # Left mouth corner
            [150.0, -150.0, -125.0],   # Right mouth corner
        ], dtype=np.float32)
        
        # 3D eye model points (relative to eye center)
        # For computing gaze direction
        self.eye_3d_model = np.array([
            [0.0, 0.0, 0.0],           # Eye center
            [0.0, -20.0, -30.0],       # Eyebrow
            [-20.0, -15.0, -30.0],     # Left eye corner
            [20.0, -15.0, -30.0],      # Right eye corner
            [-25.0, 5.0, -30.0],       # Left lower lid
            [25.0, 5.0, -30.0],        # Right lower lid
        ], dtype=np.float32)
        
        # Camera matrix (estimated for typical webcam)
        if camera_matrix is None:
            self.camera_matrix = np.array([
                [900, 0, 320],
                [0, 900, 240],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            self.camera_matrix = camera_matrix
        
        # Distortion coefficients (assumed minimal for webcams)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # Key face landmark indices (from MediaPipe 468 landmarks)
        self.landmark_indices = {
            'nose_tip': 1,              # Nose tip
            'chin': 152,                # Chin
            'left_eye_center': 33,      # Left eye center approximation
            'right_eye_center': 263,    # Right eye center approximation
            'left_mouth_corner': 61,    # Left mouth corner
            'right_mouth_corner': 291,  # Right mouth corner
            'left_pupil': 468,          # Left pupil (if available in newer versions)
            'right_pupil': 469,         # Right pupil (if available)
            'left_eye_inner': 133,      # For fallback pupil estimation
            'right_eye_inner': 362,     # For fallback pupil estimation
        }
    
    def get_face_2d_points(self, face_landmarks) -> np.ndarray:
        """
        Extract 2D face landmarks corresponding to the 3D model points.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            
        Returns:
            Array of 2D points (x, y) for face model
        """
        face_2d = np.array([
            [face_landmarks[self.landmark_indices['nose_tip']].x,
             face_landmarks[self.landmark_indices['nose_tip']].y],
            [face_landmarks[self.landmark_indices['chin']].x,
             face_landmarks[self.landmark_indices['chin']].y],
            [face_landmarks[self.landmark_indices['left_eye_center']].x,
             face_landmarks[self.landmark_indices['left_eye_center']].y],
            [face_landmarks[self.landmark_indices['right_eye_center']].x,
             face_landmarks[self.landmark_indices['right_eye_center']].y],
            [face_landmarks[self.landmark_indices['left_mouth_corner']].x,
             face_landmarks[self.landmark_indices['left_mouth_corner']].y],
            [face_landmarks[self.landmark_indices['right_mouth_corner']].x,
             face_landmarks[self.landmark_indices['right_mouth_corner']].y],
        ], dtype=np.float32)
        
        return face_2d
    
    def estimate_head_pose(self, face_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate head pose (rotation and translation vectors) using solvePnP.
        
        Uses the pinhole camera model to solve for camera pose relative to face model.
        
        Args:
            face_2d: 2D face landmarks in image space
            
        Returns:
            Tuple of (rotation_vector, translation_vector, success)
        """
        # Solve PnP problem
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.face_3d_model,
            face_2d,
            self.camera_matrix,
            self.dist_coeffs,
            useExtrinsicGuess=False,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        return rotation_vec, translation_vec, success
    
    def project_3d_to_2d(self, 
                        points_3d: np.ndarray, 
                        rotation_vec: np.ndarray, 
                        translation_vec: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image plane using head pose.
        
        Args:
            points_3d: 3D points to project
            rotation_vec: Rotation vector from head pose estimation
            translation_vec: Translation vector from head pose estimation
            
        Returns:
            Projected 2D points
        """
        projected_points, _ = cv2.projectPoints(
            points_3d,
            rotation_vec,
            translation_vec,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        return projected_points.reshape(-1, 2)
    
    def get_pupil_3d_from_2d(self, 
                            pupil_2d: np.ndarray,
                            left_eye_3d: np.ndarray,
                            right_eye_3d: np.ndarray,
                            is_left_eye: bool = True) -> Optional[np.ndarray]:
        """
        Estimate 3D pupil location from 2D image coordinates using affine transformation.
        
        Uses estimateAffine3D to find transformation between image and model coordinates.
        
        Args:
            pupil_2d: 2D pupil location in image
            left_eye_3d: 3D left eye center (from head pose)
            right_eye_3d: 3D right eye center (from head pose)
            is_left_eye: Whether processing left eye
            
        Returns:
            3D pupil location in model coordinates, or None if estimation failed
        """
        # Select appropriate eye
        eye_center_3d = left_eye_3d if is_left_eye else right_eye_3d
        
        # Create 2D points around pupil location (with z=0)
        pupil_2d_points = np.array([
            [pupil_2d[0] - 5, pupil_2d[1] - 5, 0],
            [pupil_2d[0] + 5, pupil_2d[1] - 5, 0],
            [pupil_2d[0], pupil_2d[1] + 5, 0],
        ], dtype=np.float32)
        
        # Corresponding 3D eye model points
        eye_model_2d_points = np.array([
            [eye_center_3d[0] - 5, eye_center_3d[1] - 5, 0],
            [eye_center_3d[0] + 5, eye_center_3d[1] - 5, 0],
            [eye_center_3d[0], eye_center_3d[1] + 5, 0],
        ], dtype=np.float32)
        
        # Estimate affine transformation
        try:
            # Reshape for estimateAffine3D
            mat, inliers = cv2.estimateAffine3D(pupil_2d_points, eye_model_2d_points)
            
            if mat is None or inliers is None:
                return None
            
            # Project pupil 2D point to 3D using transformation
            pupil_2d_homog = np.array([pupil_2d[0], pupil_2d[1], 0, 1], dtype=np.float32)
            pupil_3d = mat @ pupil_2d_homog[:3]
            
            return pupil_3d
        except:
            return None
    
    def compute_gaze_direction(self,
                              pupil_3d: np.ndarray,
                              eye_center_3d: np.ndarray,
                              distance_magic_number: int = 10) -> np.ndarray:
        """
        Compute gaze direction as a vector from pupil to a point in space.
        
        The distance (distance_magic_number) is arbitrary since we don't know
        the actual distance of the subject from the camera.
        
        Args:
            pupil_3d: 3D pupil location
            eye_center_3d: 3D eye center location
            distance_magic_number: Scaling factor for gaze vector (default 10)
            
        Returns:
            Gaze direction vector (unit vector)
        """
        # Vector from eye center to pupil
        pupil_vector = pupil_3d - eye_center_3d
        
        if np.linalg.norm(pupil_vector) < 1e-6:
            return np.array([0, 0, 1], dtype=np.float32)
        
        # Normalize and scale
        gaze_vector = pupil_vector / np.linalg.norm(pupil_vector)
        gaze_vector = gaze_vector * distance_magic_number
        
        return gaze_vector
    
    def compensate_head_movement(self,
                                gaze_vector_2d: np.ndarray,
                                head_pose_2d: np.ndarray,
                                head_pose_magic_number: int = 40) -> np.ndarray:
        """
        Compensate for head movement to get head-invariant gaze direction.
        
        Subtracts head pose vector from gaze vector to remove head rotation effects.
        
        Args:
            gaze_vector_2d: Gaze vector projected to 2D image
            head_pose_2d: Head pose vector projected to 2D image
            head_pose_magic_number: Scaling factor for head pose (default 40)
            
        Returns:
            Compensated gaze direction in image space
        """
        # Normalize and scale head pose
        head_norm = np.linalg.norm(head_pose_2d)
        if head_norm > 1e-6:
            head_pose_2d = (head_pose_2d / head_norm) * head_pose_magic_number
        
        # Compensate
        compensated_gaze = gaze_vector_2d - head_pose_2d
        
        return compensated_gaze
    
    def track_gaze(self, 
                   frame: np.ndarray,
                   face_landmarks,
                   image_width: int,
                   image_height: int,
                   eye_selection: str = 'left') -> Tuple[Optional[np.ndarray], dict]:
        """
        Main gaze tracking pipeline.
        
        Args:
            frame: Input video frame
            face_landmarks: MediaPipe face landmarks
            image_width: Frame width
            image_height: Frame height
            eye_selection: 'left', 'right', or 'both'
            
        Returns:
            Tuple of (gaze_2d, debug_info) where gaze_2d is the 2D gaze point on screen
        """
        debug_info = {
            'success': False,
            'head_rotation': None,
            'head_translation': None,
            'gaze_direction': None,
            'left_pupil_2d': None,
            'right_pupil_2d': None,
        }
        
        if not face_landmarks:
            return None, debug_info
        
        # Step 1: Get 2D face landmarks
        face_2d = self.get_face_2d_points(face_landmarks)
        face_2d[:, 0] *= image_width
        face_2d[:, 1] *= image_height
        
        # Step 2: Estimate head pose
        rotation_vec, translation_vec, success = self.estimate_head_pose(face_2d)
        if not success:
            return None, debug_info
        
        debug_info['head_rotation'] = rotation_vec
        debug_info['head_translation'] = translation_vec
        
        # Step 3: Get 3D eye center positions
        left_eye_3d = self.face_3d_model[2]  # Left eye center from model
        right_eye_3d = self.face_3d_model[3]  # Right eye center from model
        
        # Project to 2D for head pose visualization
        head_pose_point_3d = np.array([[0, 0, 50]], dtype=np.float32)
        head_pose_2d = self.project_3d_to_2d(head_pose_point_3d, rotation_vec, translation_vec)[0]
        
        # Step 4: Get pupil locations from face landmarks
        # Try to get pupil if available, otherwise estimate from eye inner corner
        left_pupil_idx = self.landmark_indices.get('left_pupil')
        right_pupil_idx = self.landmark_indices.get('right_pupil')
        
        # Fallback indices
        if left_pupil_idx is None or left_pupil_idx >= len(face_landmarks):
            left_pupil_idx = self.landmark_indices['left_eye_inner']
        if right_pupil_idx is None or right_pupil_idx >= len(face_landmarks):
            right_pupil_idx = self.landmark_indices['right_eye_inner']
        
        left_pupil_2d = np.array([
            face_landmarks[left_pupil_idx].x * image_width,
            face_landmarks[left_pupil_idx].y * image_height
        ], dtype=np.float32)
        
        right_pupil_2d = np.array([
            face_landmarks[right_pupil_idx].x * image_width,
            face_landmarks[right_pupil_idx].y * image_height
        ], dtype=np.float32)
        
        debug_info['left_pupil_2d'] = left_pupil_2d
        debug_info['right_pupil_2d'] = right_pupil_2d
        
        # Step 5: Get 3D eye positions from head pose
        left_eye_3d_proj = self.project_3d_to_2d(
            np.array([left_eye_3d], dtype=np.float32),
            rotation_vec, translation_vec
        )[0]
        
        right_eye_3d_proj = self.project_3d_to_2d(
            np.array([right_eye_3d], dtype=np.float32),
            rotation_vec, translation_vec
        )[0]
        
        # Step 6: Compute gaze based on selected eye
        gaze_2d = None
        
        if eye_selection in ['left', 'both']:
            # Get 3D pupil location
            pupil_3d_left = self.get_pupil_3d_from_2d(
                left_pupil_2d, left_eye_3d, right_eye_3d, is_left_eye=True
            )
            
            if pupil_3d_left is not None:
                # Compute gaze direction
                gaze_vec = self.compute_gaze_direction(pupil_3d_left, left_eye_3d)
                
                # Project to 2D
                gaze_point_3d = left_eye_3d + gaze_vec
                gaze_2d_left = self.project_3d_to_2d(
                    np.array([gaze_point_3d], dtype=np.float32),
                    rotation_vec, translation_vec
                )[0]
                
                # Compensate for head movement
                head_vec = head_pose_2d - left_eye_3d_proj
                gaze_2d = self.compensate_head_movement(
                    gaze_2d_left - left_eye_3d_proj,
                    head_vec
                ) + left_eye_3d_proj
        
        if eye_selection in ['right', 'both']:
            # Get 3D pupil location
            pupil_3d_right = self.get_pupil_3d_from_2d(
                right_pupil_2d, left_eye_3d, right_eye_3d, is_left_eye=False
            )
            
            if pupil_3d_right is not None:
                # Compute gaze direction
                gaze_vec = self.compute_gaze_direction(pupil_3d_right, right_eye_3d)
                
                # Project to 2D
                gaze_point_3d = right_eye_3d + gaze_vec
                gaze_2d_right = self.project_3d_to_2d(
                    np.array([gaze_point_3d], dtype=np.float32),
                    rotation_vec, translation_vec
                )[0]
                
                # Compensate for head movement
                head_vec = head_pose_2d - right_eye_3d_proj
                gaze_2d_right_comp = self.compensate_head_movement(
                    gaze_2d_right - right_eye_3d_proj,
                    head_vec
                ) + right_eye_3d_proj
                
                if gaze_2d is None:
                    gaze_2d = gaze_2d_right_comp
                elif eye_selection == 'both':
                    # Average both eyes
                    gaze_2d = (gaze_2d + gaze_2d_right_comp) / 2
        
        if gaze_2d is not None:
            # Clamp to frame boundaries
            gaze_2d[0] = np.clip(gaze_2d[0], 0, image_width)
            gaze_2d[1] = np.clip(gaze_2d[1], 0, image_height)
            debug_info['success'] = True
            debug_info['gaze_direction'] = gaze_2d
        
        return gaze_2d, debug_info
    
    def draw_gaze_visualization(self,
                               frame: np.ndarray,
                               gaze_2d: Optional[np.ndarray],
                               debug_info: dict,
                               face_2d: Optional[np.ndarray] = None,
                               rotation_vec: Optional[np.ndarray] = None,
                               translation_vec: Optional[np.ndarray] = None) -> None:
        """
        Draw gaze tracking visualization on frame.
        
        Args:
            frame: Input frame
            gaze_2d: 2D gaze point
            debug_info: Debug information
            face_2d: 2D face landmarks
            rotation_vec: Head rotation vector
            translation_vec: Head translation vector
        """
        h, w = frame.shape[:2]
        
        # Draw 2D face landmarks
        if face_2d is not None:
            for point in face_2d:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        # Draw pupil positions
        left_pupil = debug_info.get('left_pupil_2d')
        right_pupil = debug_info.get('right_pupil_2d')
        
        if left_pupil is not None:
            cv2.circle(frame, tuple(left_pupil.astype(int)), 2, (255, 0, 0), -1)
        if right_pupil is not None:
            cv2.circle(frame, tuple(right_pupil.astype(int)), 2, (255, 0, 0), -1)
        
        # Draw gaze point
        if gaze_2d is not None:
            x, y = int(gaze_2d[0]), int(gaze_2d[1])
            cv2.circle(frame, (x, y), 8, (0, 0, 255), 2)  # Red circle for gaze
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Red dot at center
            
            # Draw line from center to gaze point
            center = (w // 2, h // 2)
            cv2.line(frame, center, (x, y), (0, 255, 255), 2)  # Yellow line
        
        # Draw status text
        status_text = "Gaze: OK" if debug_info.get('success') else "Gaze: TRACKING"
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw head pose visualization if available
        if rotation_vec is not None and translation_vec is not None:
            # Draw 3D axes at nose tip
            axis_points = np.float32([
                [0, 0, 0], [50, 0, 0], [0, 50, 0], [0, 0, 50]
            ])
            img_points, _ = cv2.projectPoints(
                axis_points, rotation_vec, translation_vec,
                self.camera_matrix, self.dist_coeffs
            )
            img_points = img_points.astype(int)
            
            # Draw axes
            origin = tuple(img_points[0].ravel())
            cv2.line(frame, origin, tuple(img_points[1].ravel()), (0, 0, 255), 2)  # X-red
            cv2.line(frame, origin, tuple(img_points[2].ravel()), (0, 255, 0), 2)  # Y-green
            cv2.line(frame, origin, tuple(img_points[3].ravel()), (255, 0, 0), 2)  # Z-blue
