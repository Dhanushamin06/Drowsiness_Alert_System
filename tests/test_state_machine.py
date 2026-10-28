"""Unit tests for state machine using fake clock."""
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state_machine import StateMachine, DriverState
from config import CONFIG


class FakeClock:
    """Controllable clock for testing."""
    def __init__(self):
        self.time = 0.0

    def monotonic(self):
        return self.time

    def advance(self, seconds: float):
        self.time += seconds


class TestStateMachine(unittest.TestCase):
    """Test cases for timestamp-based state machine."""

    def setUp(self):
        self.clock = FakeClock()
        self.machine = StateMachine()
        self.machine.set_ear_threshold(0.2)

        # Monkey-patch time.monotonic for the state machine
        import state_machine
        self.original_monotonic = state_machine.time.monotonic
        state_machine.time.monotonic = self.clock.monotonic

    def tearDown(self):
        import state_machine
        state_machine.time.monotonic = self.original_monotonic

    def _update(self, ear=0.3, yaw=0.0, pitch=0.0, phone=False, face=True):
        """Helper to update state machine."""
        return self.machine.update(ear, yaw, pitch, phone, face, self.clock.monotonic())

    def test_initial_state_attentive(self):
        """Initial state should be ATTENTIVE."""
        state = self._update()
        self.assertEqual(state, DriverState.ATTENTIVE)

    def test_blink_does_not_trigger_drowsy(self):
        """Short eye closure (blink) should not trigger drowsy."""
        # Eyes closed for 0.2s (within blink range)
        self._update(ear=0.1)
        self.clock.advance(0.2)
        self._update(ear=0.1)

        # Eyes open
        self.clock.advance(0.1)
        state = self._update(ear=0.3)

        self.assertEqual(state, DriverState.ATTENTIVE)

    def test_sustained_eye_closure_triggers_drowsy(self):
        """Eyes closed for >= 1.5s should trigger DROWSY."""
        self._update(ear=0.1)  # Eyes closed
        self.clock.advance(1.5)
        state = self._update(ear=0.1)

        self.assertEqual(state, DriverState.DROWSY)

    def test_head_turn_triggers_distracted(self):
        """Head turned >25 deg for >= 2s should trigger DISTRACTED."""
        self._update(ear=0.3, yaw=30.0)  # Looking right
        self.clock.advance(2.0)
        state = self._update(ear=0.3, yaw=30.0)

        self.assertEqual(state, DriverState.DISTRACTED)

    def test_phone_detected_triggers_phone_state(self):
        """Phone detected for >= 1s should trigger PHONE."""
        self._update(ear=0.3, phone=True)
        self.clock.advance(1.0)
        state = self._update(ear=0.3, phone=True)

        self.assertEqual(state, DriverState.PHONE)

    def test_no_face_triggers_driver_missing(self):
        """No face for >= 3s should trigger DRIVER_MISSING."""
        self._update(ear=0.3, face=True)
        self.clock.advance(3.0)
        state = self._update(ear=0.3, face=False)

        self.assertEqual(state, DriverState.DRIVER_MISSING)

    def test_priority_driver_missing_over_drowsy(self):
        """DRIVER_MISSING should have priority over DROWSY."""
        # Start drowsy condition
        self._update(ear=0.1, face=True)
        self.clock.advance(1.5)
        self._update(ear=0.1, face=True)
        self.assertEqual(self.machine.current_state, DriverState.DROWSY)

        # Then face disappears for 3s (missing takes priority)
        self.clock.advance(3.0)
        state = self._update(ear=0.1, face=False)
        self.assertEqual(state, DriverState.DRIVER_MISSING)

    def test_priority_drowsy_over_phone(self):
        """DROWSY should have priority over PHONE."""
        # Start phone condition
        self._update(ear=0.3, phone=True)
        self.clock.advance(1.0)
        self._update(ear=0.3, phone=True)
        self.assertEqual(self.machine.current_state, DriverState.PHONE)

        # Then eyes close for 1.5s (drowsy takes priority)
        self.clock.advance(1.5)
        state = self._update(ear=0.1, phone=True)
        self.assertEqual(state, DriverState.DROWSY)

    def test_priority_phone_over_distracted(self):
        """PHONE should have priority over DISTRACTED."""
        # Start distracted condition
        self._update(ear=0.3, yaw=30.0)
        self.clock.advance(2.0)
        self._update(ear=0.3, yaw=30.0)
        self.assertEqual(self.machine.current_state, DriverState.DISTRACTED)

        # Then phone detected for 1s (phone takes priority)
        self.clock.advance(1.0)
        state = self._update(ear=0.3, yaw=30.0, phone=True)
        self.assertEqual(state, DriverState.PHONE)

    def test_hysteresis_holds_alert(self):
        """Alert should hold for hysteresis duration after condition clears."""
        # Trigger drowsy
        self._update(ear=0.1)
        self.clock.advance(1.5)
        self._update(ear=0.1)
        self.assertEqual(self.machine.current_state, DriverState.DROWSY)

        # Eyes open
        self.clock.advance(0.1)
        state = self._update(ear=0.3)

        # Should still be DROWSY due to hysteresis (0.5s)
        self.assertEqual(state, DriverState.DROWSY)

        # After hysteresis expires
        self.clock.advance(0.5)
        state = self._update(ear=0.3)
        self.assertEqual(state, DriverState.ATTENTIVE)

    def test_transition_logging(self):
        """State transitions should be logged with timestamps."""
        self._update(ear=0.1)
        self.clock.advance(1.5)
        self._update(ear=0.1)

        transitions = self.machine.get_transitions()
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].from_state, DriverState.ATTENTIVE)
        self.assertEqual(transitions[0].to_state, DriverState.DROWSY)
        self.assertAlmostEqual(transitions[0].timestamp, 1.5, places=1)


if __name__ == '__main__':
    unittest.main()