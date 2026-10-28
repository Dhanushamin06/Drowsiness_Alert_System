"""Feature extraction: EAR, head pose, and smoothing."""
import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple, List
from config import CONFIG


# Landmark indices for 3D model points used in solvePnP
MODEL_POINTS_3D = np.array(CONFIG.head_pose_model_points, dtype=np.float32)

# 2D landmark indices corresponding to the 3D model points
# (nose_tip, chin, left_eye_corner, right_eye_corner, left_mouth_corner, right_mouth_corner)
LANDMARK_INDICES_2D = (1, 152, 33, 263, 61, 291)


class MovingAverage:
    """Simple moving average filter using O(1) deque."""
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.values: deque = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        self.values.append(value)
        return float(np.mean(self.values))

    def reset(self):
        self.values.clear()


class ExponentialSmoothing:
    """Exponential moving average filter."""
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, value: float) -> float:
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


def calculate_ear(landmarks: np.ndarray, eye_indices: Tuple[int, ...]) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) for one eye.
    EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|)
    """
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return float(ear)


def compute_ear(landmarks: np.ndarray) -> float:
    """Compute average EAR for both eyes."""
    right_ear = calculate_ear(landmarks, CONFIG.right_eye_indices)
    left_ear = calculate_ear(landmarks, CONFIG.left_eye_indices)
    return (right_ear + left_ear) / 2.0


def estimate_head_pose(landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[float, float, float]:
    """
    Estimate head pose (pitch, yaw, roll) using solvePnP.
    Returns angles in degrees, normalized to -90..90 range.
    """
    h, w = frame_shape[:2]

    # 2D image points from landmarks
    image_points = np.array([landmarks[i] for i in LANDMARK_INDICES_2D], dtype=np.float32)

    # Camera matrix with focal length = frame width
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    # Convert rotation vector to rotation matrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)

    # Extract Euler angles (pitch, yaw, roll)
    # Using ZYX convention (yaw, pitch, roll)
    sy = np.sqrt(rotation_mat[0, 0]**2 + rotation_mat[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(-rotation_mat[2, 0], sy)
        yaw = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
        roll = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
    else:
        pitch = np.arctan2(-rotation_mat[2, 0], sy)
        yaw = np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1])
        roll = 0.0

    # Convert to degrees
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    roll = np.degrees(roll)

    # Normalize to -90..90 range to avoid wraparound
    def normalize_angle(angle: float) -> float:
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180
        return angle

    return normalize_angle(pitch), normalize_angle(yaw), normalize_angle(roll)


class FeatureExtractor:
    """Extracts and smooths EAR and head pose features."""
    def __init__(self):
        self.ear_smoother = MovingAverage(CONFIG.ear_smoothing_frames)
        self.pitch_smoother = MovingAverage(CONFIG.head_pose_smoothing_frames)
        self.yaw_smoother = MovingAverage(CONFIG.head_pose_smoothing_frames)
        self.roll_smoother = MovingAverage(CONFIG.head_pose_smoothing_frames)
        self.neutral_yaw: float = 0.0
        self.neutral_pitch: float = 0.0

    def set_neutral_pose(self, yaw: float, pitch: float):
        self.neutral_yaw = yaw
        self.neutral_pitch = pitch

    def reset_smoothers(self):
        self.ear_smoother.reset()
        self.pitch_smoother.reset()
        self.yaw_smoother.reset()
        self.roll_smoother.reset()

    def process(self, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """
        Process landmarks and return smoothed EAR, pitch, yaw, roll.
        All angles are relative to calibrated neutral pose.
        """
        ear = compute_ear(landmarks)
        smoothed_ear = self.ear_smoother.update(ear)

        pitch, yaw, roll = estimate_head_pose(landmarks, frame_shape)
        smoothed_pitch = self.pitch_smoother.update(pitch) - self.neutral_pitch
        smoothed_yaw = self.yaw_smoother.update(yaw) - self.neutral_yaw
        smoothed_roll = self.roll_smoother.update(roll)

        # Re-normalize after baseline subtraction
        def normalize(angle: float) -> float:
            while angle > 90:
                angle -= 180
            while angle < -90:
                angle += 180
            return angle

        return (
            smoothed_ear,
            normalize(smoothed_pitch),
            normalize(smoothed_yaw),
            normalize(smoothed_roll)
        )