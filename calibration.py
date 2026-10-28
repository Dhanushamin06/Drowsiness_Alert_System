"""Calibration module for EAR baseline and neutral head pose."""
import time
import numpy as np
import cv2
from typing import Optional, Tuple
from config import CONFIG


class Calibration:
    """Handles startup calibration and recalibration."""
    def __init__(self):
        self.ear_values: list = []
        self.yaw_values: list = []
        self.pitch_values: list = []
        self.is_calibrating: bool = False
        self.calibration_start_time: float = 0.0
        self.ear_threshold: Optional[float] = None
        self.neutral_yaw: float = 0.0
        self.neutral_pitch: float = 0.0

    def start(self):
        """Start calibration process."""
        self.ear_values.clear()
        self.yaw_values.clear()
        self.pitch_values.clear()
        self.is_calibrating = True
        self.calibration_start_time = time.monotonic()

    def update(self, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> bool:
        """
        Update calibration with new frame data.
        Returns True if calibration is complete.
        """
        if not self.is_calibrating:
            return False

        ear = self._compute_raw_ear(landmarks)
        pitch, yaw, _ = self._estimate_raw_head_pose(landmarks, frame_shape)

        self.ear_values.append(ear)
        self.yaw_values.append(yaw)
        self.pitch_values.append(pitch)

        elapsed = time.monotonic() - self.calibration_start_time
        if elapsed >= CONFIG.calibration_duration_sec:
            self._finalize()
            return True

        return False

    def _compute_raw_ear(self, landmarks: np.ndarray) -> float:
        """Compute EAR without smoothing."""
        from features import calculate_ear
        right_ear = calculate_ear(landmarks, CONFIG.right_eye_indices)
        left_ear = calculate_ear(landmarks, CONFIG.left_eye_indices)
        return (right_ear + left_ear) / 2.0

    def _estimate_raw_head_pose(self, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[float, float, float]:
        """Estimate head pose without smoothing."""
        from features import estimate_head_pose
        return estimate_head_pose(landmarks, frame_shape)

    def _finalize(self):
        """Finalize calibration and compute thresholds."""
        if not self.ear_values:
            # Fallback defaults
            self.ear_threshold = 0.2
            self.neutral_yaw = 0.0
            self.neutral_pitch = 0.0
        else:
            mean_ear = np.mean(self.ear_values)
            self.ear_threshold = np.clip(
                mean_ear * CONFIG.ear_threshold_multiplier,
                CONFIG.ear_threshold_min,
                CONFIG.ear_threshold_max
            )
            self.neutral_yaw = float(np.mean(self.yaw_values))
            self.neutral_pitch = float(np.mean(self.pitch_values))

        self.is_calibrating = False

    def get_ear_threshold(self) -> float:
        if self.ear_threshold is None:
            return 0.2  # Default fallback
        return self.ear_threshold

    def get_neutral_pose(self) -> Tuple[float, float]:
        return self.neutral_yaw, self.neutral_pitch

    def is_active(self) -> bool:
        return self.is_calibrating

    def get_progress(self) -> float:
        """Return calibration progress 0.0-1.0."""
        if not self.is_calibrating:
            return 1.0
        elapsed = time.monotonic() - self.calibration_start_time
        return min(elapsed / CONFIG.calibration_duration_sec, 1.0)

    def draw_overlay(self, frame: np.ndarray):
        """Draw calibration overlay on frame."""
        if not self.is_calibrating:
            return

        progress = self.get_progress()
        h, w = frame.shape[:2]

        # Draw progress bar
        bar_width = int(w * 0.6)
        bar_height = 20
        x = (w - bar_width) // 2
        y = h - 80

        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (100, 100, 100), -1)
        cv2.rectangle(frame, (x, y), (x + int(bar_width * progress), y + bar_height), (0, 255, 0), -1)
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 2)

        # Draw text
        text = f"Calibrating... Look straight ahead, eyes open ({progress*100:.0f}%)"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = y - 10
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)