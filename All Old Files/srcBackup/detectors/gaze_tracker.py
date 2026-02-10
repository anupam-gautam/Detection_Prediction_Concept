"""
Gaze Tracking Module
Tracks eye gaze direction using 3D face model and pupil localization
Based on: Eye Gaze Tracking using Camera and OpenCV
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class GazeTracker:
    """
    Tracks eye gaze direction using a generic 3D face model and pupil localization.
    
    Key components:
    - Head pose estimation using solvePnP
    - Pupil 3D localization using affine transformation
    - Gaze direction computation accounting for head movement
    """
    
    # Face landmark indices from MediaPipe (468 landmarks)
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_CENTER = 33
    RIGHT_EYE_CENTER = 263
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    
    def __init__(self, camera_matrix: Optional[np.ndarray] = None,
                 face_3d_model: Optional[np.ndarray] = None,
                 eye_3d_model: Optional[np.ndarray] = None):
        """
        Initialize the GazeTracker.
        
        Args:
            camera_matrix: Optional camera intrinsic matrix
            face_3d_model: Optional 3D face model points (mm)
            eye_3d_model: Optional 3D eye model points (mm)
        """
        # Generic 3D face model points (in mm, relative to nose tip as origin)
        self.face_3d_model = face_3d_model if face_3d_model is not None else np.array([
            [0.0, 0.0, 0.0],           # Nose tip (origin)
            [0.0, -330.0, -65.0],      # Chin
            [-225.0, 170.0, -135.0],   # Left eye center
            [225.0, 170.0, -135.0],    # Right eye center
            [-150.0, -150.0, -125.0],  # Left mouth corner
            [150.0, -150.0, -125.0],   # Right mouth corner
        ], dtype=np.float32)
        
        # 3D eye model points (relative to eye center)
        self.eye_3d_model = eye_3d_model if eye_3d_model is not None else np.array([
            [0.0, 0.0, 0.0],           # Eye center
            [0.0, -20.0, -30.0],       # Eyebrow
            [-20.0, -15.0, -30.0],     # Left eye corner
            [20.0, -15.0, -30.0],      # Right eye corner
            [-25.0, 5.0, -30.0],       # Left lower lid
            [25.0, 5.0, -30.0],        # Right lower lid
        ], dtype=np.float32)
        
        # Camera matrix (estimated for typical webcam)
        self.camera_matrix = camera_matrix if camera_matrix is not None else np.array([
            [900, 0, 640],
            [0, 900, 360],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Distortion coefficients
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    def get_face_2d_points(self, face_landmarks) -> np.ndarray:
        """
        Extract 2D face landmarks corresponding to the 3D model points.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            
        Returns:
            Array of 2D points (x, y) for face model
        """
        face_2d = np.array([
            [face_landmarks[self.NOSE_TIP].x,
             face_landmarks[self.NOSE_TIP].y],
            [face_landmarks[self.CHIN].x,
             face_landmarks[self.CHIN].y],
            [face_landmarks[self.LEFT_EYE_CENTER].x,
             face_landmarks[self.LEFT_EYE_CENTER].y],
            [face_landmarks[self.RIGHT_EYE_CENTER].x,
             face_landmarks[self.RIGHT_EYE_CENTER].y],
            [face_landmarks[self.LEFT_MOUTH_CORNER].x,
             face_landmarks[self.LEFT_MOUTH_CORNER].y],
            [face_landmarks[self.RIGHT_MOUTH_CORNER].x,
             face_landmarks[self.RIGHT_MOUTH_CORNER].y],
        ], dtype=np.float32)
        
        return face_2d
    
    def estimate_head_pose(self, face_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Estimate head pose using solvePnP.
        
        Args:
            face_2d: 2D face landmarks in image space
            
        Returns:
            Tuple of (rotation_vector, translation_vector, success)
        """
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.face_3d_model,
            face_2d,
            self.camera_matrix,
            self.dist_coeffs,
            useExtrinsicGuess=False,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        return rotation_vec, translation_vec, success
    
    def project_3d_to_2d(self, points_3d: np.ndarray, 
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
                            eye_center_3d: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate 3D pupil location from 2D image coordinates using affine transformation.
        
        Args:
            pupil_2d: 2D pupil location in image
            eye_center_3d: 3D eye center from head pose
            
        Returns:
            3D pupil location or None if estimation failed
        """
        # Create 2D points around pupil location
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
        
        try:
            mat, inliers = cv2.estimateAffine3D(pupil_2d_points, eye_model_2d_points)
            
            if mat is None:
                return None
            
            # Project pupil 2D point to 3D
            pupil_2d_homog = np.array([pupil_2d[0], pupil_2d[1], 0], dtype=np.float32)
            pupil_3d = mat[:3, :3] @ pupil_2d_homog + mat[:3, 3]
            
            return pupil_3d
        except:
            return None
    
    def compute_gaze_direction(self,
                              pupil_3d: np.ndarray,
                              eye_center_3d: np.ndarray,
                              distance_scale: int = 10) -> np.ndarray:
        """
        Compute gaze direction as a vector from eye center to pupil.
        
        Args:
            pupil_3d: 3D pupil location
            eye_center_3d: 3D eye center location
            distance_scale: Scaling factor for gaze vector
            
        Returns:
            Gaze direction vector
        """
        pupil_vector = pupil_3d - eye_center_3d
        
        if np.linalg.norm(pupil_vector) < 1e-6:
            return np.array([0, 0, 1], dtype=np.float32)
        
        gaze_vector = pupil_vector / np.linalg.norm(pupil_vector)
        gaze_vector = gaze_vector * distance_scale
        
        return gaze_vector
    
    def compensate_head_movement(self,
                                gaze_vector_2d: np.ndarray,
                                head_pose_2d: np.ndarray,
                                head_pose_scale: int = 40) -> np.ndarray:
        """
        Compensate for head movement to get head-invariant gaze direction.
        
        Args:
            gaze_vector_2d: Gaze vector projected to 2D
            head_pose_2d: Head pose vector projected to 2D
            head_pose_scale: Scaling factor for head pose
            
        Returns:
            Compensated gaze direction
        """
        head_norm = np.linalg.norm(head_pose_2d)
        if head_norm > 1e-6:
            head_pose_2d = (head_pose_2d / head_norm) * head_pose_scale
        
        compensated_gaze = gaze_vector_2d - head_pose_2d
        
        return compensated_gaze
    
    def track_gaze(self, 
                   face_landmarks,
                   image_width: int,
                   image_height: int,
                   eye: str = 'left') -> Tuple[Optional[np.ndarray], dict]:
        """
        Main gaze tracking pipeline.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            image_width: Frame width
            image_height: Frame height
            eye: 'left', 'right', or 'both'
            
        Returns:
            Tuple of (gaze_2d, debug_info)
        """
        debug_info = {
            'success': False,
            'head_rotation': None,
            'head_translation': None,
            'gaze_point': None,
        }
        
        if not face_landmarks:
            return None, debug_info
        
        # Get 2D face landmarks
        face_2d = self.get_face_2d_points(face_landmarks)
        face_2d[:, 0] *= image_width
        face_2d[:, 1] *= image_height
        
        # Store nose tip 2D
        nose_tip_2d = face_2d[0].astype(int)
        debug_info['nose_tip_2d'] = nose_tip_2d

        # Estimate head pose
        rotation_vec, translation_vec, success = self.estimate_head_pose(face_2d)
        if not success:
            return None, debug_info
        
        debug_info['head_rotation'] = rotation_vec
        debug_info['head_translation'] = translation_vec
        
        # Get 3D eye center positions (model coordinates)
        left_eye_3d = self.face_3d_model[2]
        right_eye_3d = self.face_3d_model[3]

        # Project eye centers to image for visualization
        left_eye_2d = self.project_3d_to_2d(np.array([left_eye_3d], dtype=np.float32), rotation_vec, translation_vec)[0]
        right_eye_2d = self.project_3d_to_2d(np.array([right_eye_3d], dtype=np.float32), rotation_vec, translation_vec)[0]
        debug_info['left_eye_2d'] = left_eye_2d
        debug_info['right_eye_2d'] = right_eye_2d
        
        # Get pupil locations
        left_pupil_2d = np.array([
            face_landmarks[self.LEFT_EYE_INNER].x * image_width,
            face_landmarks[self.LEFT_EYE_INNER].y * image_height
        ], dtype=np.float32)
        
        right_pupil_2d = np.array([
            face_landmarks[self.RIGHT_EYE_INNER].x * image_width,
            face_landmarks[self.RIGHT_EYE_INNER].y * image_height
        ], dtype=np.float32)
        
        # Compute gaze based on selected eye
        gaze_2d = None
        
        gaze_vectors_cam = []
        if eye in ['left', 'both']:
            pupil_3d = self.get_pupil_3d_from_2d(left_pupil_2d, left_eye_3d)
            
            if pupil_3d is not None:
                gaze_vec = self.compute_gaze_direction(pupil_3d, left_eye_3d)
                gaze_point_3d = left_eye_3d + gaze_vec
                gaze_2d = self.project_3d_to_2d(
                    np.array([gaze_point_3d], dtype=np.float32),
                    rotation_vec, translation_vec
                )[0]

                # Convert gaze vector to camera coordinates (rotate only)
                R, _ = cv2.Rodrigues(rotation_vec)
                gaze_vector_cam = R @ gaze_vec.reshape(3, 1)
                gaze_vectors_cam.append(gaze_vector_cam.ravel())
                debug_info['gaze_vector_3d_cam_left'] = gaze_vector_cam.ravel()
                debug_info['gaze_point_3d_left'] = gaze_point_3d
        
        if eye in ['right', 'both']:
            pupil_3d = self.get_pupil_3d_from_2d(right_pupil_2d, right_eye_3d)
            
            if pupil_3d is not None:
                gaze_vec = self.compute_gaze_direction(pupil_3d, right_eye_3d)
                gaze_point_3d = right_eye_3d + gaze_vec
                gaze_2d_right = self.project_3d_to_2d(
                    np.array([gaze_point_3d], dtype=np.float32),
                    rotation_vec, translation_vec
                )[0]
                
                if gaze_2d is None:
                    gaze_2d = gaze_2d_right
                elif eye == 'both':
                    gaze_2d = (gaze_2d + gaze_2d_right) / 2

                # Convert right gaze vector to camera coordinates
                R, _ = cv2.Rodrigues(rotation_vec)
                gaze_vector_cam_r = R @ gaze_vec.reshape(3, 1)
                gaze_vectors_cam.append(gaze_vector_cam_r.ravel())
                debug_info['gaze_vector_3d_cam_right'] = gaze_vector_cam_r.ravel()
                debug_info['gaze_point_3d_right'] = gaze_point_3d
        
        if gaze_2d is not None:
            gaze_2d[0] = np.clip(gaze_2d[0], 0, image_width)
            gaze_2d[1] = np.clip(gaze_2d[1], 0, image_height)
            debug_info['success'] = True
            debug_info['gaze_point'] = gaze_2d

            # For convenience include the final projected gaze point
            debug_info['gaze_point_2d'] = gaze_2d
        
        # Calculate single gaze vector from nose relative to camera
        if gaze_vectors_cam:
            avg_gaze_vector_cam = np.mean(gaze_vectors_cam, axis=0)
            # Nose tip 3D position in camera coordinates is translation_vec
            nose_tip_3d_cam = translation_vec.ravel()
            
            # Scale the gaze vector for visualization
            gaze_vector_length = 100.0  # Adjustable length
            gaze_endpoint_3d_cam = nose_tip_3d_cam + avg_gaze_vector_cam * gaze_vector_length
            
            # Project the gaze endpoint to 2D
            gaze_direction_2d_from_nose = self.project_3d_to_2d(
                np.array([gaze_endpoint_3d_cam], dtype=np.float32),
                rotation_vec, translation_vec
            )[0]
            
            debug_info['gaze_direction_2d_from_nose'] = gaze_direction_2d_from_nose.astype(int)
            
        return gaze_2d, debug_info
    
    def draw_gaze(self, frame: np.ndarray, debug_info: dict = None,
                  color: Tuple[int, int, int] = (0, 255, 255),
                  thickness: int = 2) -> np.ndarray:
        """
        Draw gaze point and direction on frame.
        
        Args:
            frame: Input frame
            debug_info: Dictionary containing debug information including gaze vectors
            color: Drawing color (BGR)
            thickness: Line thickness
            
        Returns:
            Frame with gaze visualization
        """
        if debug_info is None or not debug_info.get('success'):
            return frame

        nose_tip_2d = debug_info.get('nose_tip_2d')
        gaze_direction_2d_from_nose = debug_info.get('gaze_direction_2d_from_nose')

        if nose_tip_2d is not None and gaze_direction_2d_from_nose is not None:
            # Draw arrow from nose tip to gaze direction endpoint
            p1 = tuple(nose_tip_2d)
            p2 = tuple(gaze_direction_2d_from_nose)
            cv2.arrowedLine(frame, p1, p2, color, thickness, tipLength=0.2)
            # Draw a small circle at the nose tip for clarity
            cv2.circle(frame, p1, 3, (0, 255, 0), -1)

        return frame
