"""Timestamp-based driver state machine."""
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional
from config import CONFIG


class DriverState(Enum):
    ATTENTIVE = "ATTENTIVE"
    DISTRACTED = "DISTRACTED"
    DROWSY = "DROWSY"
    PHONE = "PHONE"
    DRIVER_MISSING = "DRIVER_MISSING"


@dataclass
class StateTransition:
    timestamp: float
    from_state: DriverState
    to_state: DriverState


@dataclass
class TimerState:
    """Tracks continuous condition durations."""
    drowsy_start: Optional[float] = None
    distracted_start: Optional[float] = None
    phone_start: Optional[float] = None
    missing_start: Optional[float] = None
    blink_start: Optional[float] = None
    last_alert_time: float = 0.0
    alert_hold_until: float = 0.0


class StateMachine:
    """Timestamp-based state machine for driver monitoring."""
    def __init__(self):
        self.current_state = DriverState.ATTENTIVE
        self.timers = TimerState()
        self.transitions: List[StateTransition] = []
        self.ear_threshold: float = 0.2
        self._last_face_detected: bool = True
        self._last_yaw: float = 0.0  # Tracks sustained yaw for distraction direction

    def set_ear_threshold(self, threshold: float):
        self.ear_threshold = threshold

    def update(
        self,
        ear: float,
        yaw: float,
        pitch: float,
        phone_detected: bool,
        face_detected: bool,
        timestamp: float
    ) -> DriverState:
        """Update state machine with current sensor readings."""
        self._update_timers(ear, yaw, pitch, phone_detected, face_detected, timestamp)
        new_state = self._evaluate_state(timestamp)

        if new_state != self.current_state:
            self._transition(new_state, timestamp)

        # Apply hysteresis: hold alert state for minimum duration
        if self.timers.alert_hold_until > timestamp:
            return self.current_state

        return self.current_state

    def _update_timers(
        self,
        ear: float,
        yaw: float,
        pitch: float,
        phone_detected: bool,
        face_detected: bool,
        timestamp: float
    ):
        """Update all condition timers."""
        # Face detection timer
        if not face_detected:
            if self.timers.missing_start is None:
                self.timers.missing_start = timestamp
        else:
            self.timers.missing_start = None

        # Track sustained yaw for distraction direction display
        self._last_yaw = yaw

        # Drowsiness timer (EAR below threshold)
        is_eye_closed = ear < self.ear_threshold
        if is_eye_closed:
            if self.timers.drowsy_start is None:
                self.timers.drowsy_start = timestamp
            # Check for blink (short eye closure)
            if self.timers.blink_start is None:
                self.timers.blink_start = timestamp
        else:
            self.timers.drowsy_start = None
            # Blink ended
            if self.timers.blink_start is not None:
                blink_duration = timestamp - self.timers.blink_start
                if CONFIG.blink_min_duration_sec <= blink_duration <= CONFIG.blink_max_duration_sec:
                    # Valid blink, don't count as drowsy
                    pass
                self.timers.blink_start = None

        # Distraction timer (head turned)
        is_distracted = abs(yaw) > CONFIG.yaw_threshold_deg
        if is_distracted:
            if self.timers.distracted_start is None:
                self.timers.distracted_start = timestamp
        else:
            self.timers.distracted_start = None

        # Phone timer
        if phone_detected:
            if self.timers.phone_start is None:
                self.timers.phone_start = timestamp
        else:
            self.timers.phone_start = None

    def _evaluate_state(self, timestamp: float) -> DriverState:
        """Evaluate current state based on timer durations."""
        # Priority order: DRIVER_MISSING > DROWSY > PHONE > DISTRACTED > ATTENTIVE

        # DRIVER_MISSING
        if self.timers.missing_start is not None:
            duration = timestamp - self.timers.missing_start
            if duration >= CONFIG.driver_missing_duration_sec:
                return DriverState.DRIVER_MISSING

        # DROWSY (but not if it was just a blink)
        if self.timers.drowsy_start is not None:
            duration = timestamp - self.timers.drowsy_start
            if duration >= CONFIG.drowsy_duration_sec:
                # Check if this might be a blink that just ended
                if self.timers.blink_start is None or (timestamp - self.timers.blink_start) > CONFIG.blink_max_duration_sec:
                    return DriverState.DROWSY

        # PHONE
        if self.timers.phone_start is not None:
            duration = timestamp - self.timers.phone_start
            if duration >= CONFIG.phone_duration_sec:
                return DriverState.PHONE

        # DISTRACTED
        if self.timers.distracted_start is not None:
            duration = timestamp - self.timers.distracted_start
            if duration >= CONFIG.distracted_duration_sec:
                direction = "LEFT" if self._get_last_yaw() < 0 else "RIGHT"
                return DriverState.DISTRACTED

        return DriverState.ATTENTIVE

    def _get_last_yaw(self) -> float:
        """Get the last sustained yaw for distraction direction display."""
        return self._last_yaw

    def _transition(self, new_state: DriverState, timestamp: float):
        """Record state transition."""
        transition = StateTransition(
            timestamp=timestamp,
            from_state=self.current_state,
            to_state=new_state
        )
        self.transitions.append(transition)
        self.current_state = new_state

        # Set alert hold timer for hysteresis
        self.timers.alert_hold_until = timestamp + CONFIG.hysteresis_duration_sec

    def get_active_timers(self, timestamp: float) -> dict:
        """Get current timer values for display."""
        return {
            'drowsy': timestamp - self.timers.drowsy_start if self.timers.drowsy_start else 0.0,
            'distracted': timestamp - self.timers.distracted_start if self.timers.distracted_start else 0.0,
            'phone': timestamp - self.timers.phone_start if self.timers.phone_start else 0.0,
            'missing': timestamp - self.timers.missing_start if self.timers.missing_start else 0.0,
        }

    def get_state_label(self) -> str:
        """Get display label for current state."""
        if self.current_state == DriverState.DISTRACTED:
            # Would need to track direction, simplified here
            return "DISTRACTED"
        return self.current_state.value

    def get_transitions(self) -> List[StateTransition]:
        return self.transitions

    def reset(self):
        """Reset state machine (e.g., after recalibration)."""
        self.current_state = DriverState.ATTENTIVE
        self.timers = TimerState()
        self.transitions.clear()