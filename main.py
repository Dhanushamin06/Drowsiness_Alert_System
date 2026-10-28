"""Main loop: webcam capture, pipeline, overlay, keyboard controls."""
import cv2
import time
import sys
import argparse
import platform
import numpy as np
from typing import Optional
from config import CONFIG
from features import FeatureExtractor
from calibration import Calibration
from state_machine import StateMachine, DriverState
from phone_detector import PhoneDetector, create_phone_detector
from profiler import Profiler


# Try to import winsound for Windows beep
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class DriverMonitor:
    """Main driver monitoring application."""
    def __init__(self, lite_mode: bool = False, video_path: Optional[str] = None):
        self.lite_mode = lite_mode
        self.video_path = video_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False

        # Configure resolution and processing rate
        if lite_mode:
            self.frame_width = CONFIG.lite_frame_width
            self.frame_height = CONFIG.lite_frame_height
            self.process_every_n = CONFIG.lite_process_every_n_frames
        else:
            self.frame_width = CONFIG.frame_width
            self.frame_height = CONFIG.frame_height
            self.process_every_n = CONFIG.process_every_n_frames

        # Components
        self.face_mesh = None
        self.feature_extractor = FeatureExtractor()
        self.calibration = Calibration()
        self.state_machine = StateMachine()
        self.phone_detector: Optional[PhoneDetector] = None
        self.profiler = Profiler()

        # State
        self.frame_count = 0
        self.process_frame_count = 0
        self.last_fps_time = time.monotonic()
        self.fps = 0.0
        self.show_debug = CONFIG.show_debug_overlay
        self.last_alert_time = 0.0
        self.ear_threshold = 0.2
        self.last_frame: Optional[np.ndarray] = None  # For screenshot (s key)

        # Calibration starts automatically
        self.calibration.start()

    def initialize(self) -> bool:
        """Initialize camera, MediaPipe, and phone detector."""
        # Camera
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            self.cap = cv2.VideoCapture(CONFIG.camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows

        if not self.cap.isOpened():
            print(f"Error: Could not open camera (index {CONFIG.camera_index})")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, CONFIG.target_fps)

        # MediaPipe FaceLandmarker (new Tasks API for MediaPipe >= 0.10)
        import mediapipe as mp
        import os
        model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
        if not os.path.exists(model_path):
            print(f"Error: MediaPipe model not found at {model_path}")
            print("Download it with:")
            print("  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -OutFile face_landmarker.task")
            return False

        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        BaseOptions = mp.tasks.BaseOptions

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=CONFIG.max_num_faces,
            min_face_detection_confidence=CONFIG.min_detection_confidence,
            min_face_presence_confidence=CONFIG.min_detection_confidence,
            min_tracking_confidence=CONFIG.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        # Phone detector
        if CONFIG.use_phone_detector:
            self.phone_detector = create_phone_detector()
            if not self.phone_detector.is_running():
                print("Warning: Phone detector failed to start, falling back to heuristic mode")
                CONFIG.use_phone_detector = False
                self.phone_detector = None

        # Profiler
        self.profiler.start_sampling()

        return True

    def run(self):
        """Main processing loop."""
        if not self.initialize():
            return

        self.running = True
        print("Driver Monitor started. Press ESC/q to quit, 'c' to recalibrate, 'd' debug, 'p' phone toggle, 's' screenshot")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.video_path:
                    # Loop video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("Error: Failed to read frame")
                break

            self.frame_count += 1
            current_time = time.monotonic()

            # Calculate FPS
            if current_time - self.last_fps_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time

            # Process every N frames
            process_this_frame = (self.process_frame_count % self.process_every_n == 0)
            self.process_frame_count += 1

            # Run pipeline
            annotated_frame, state = self.process_frame(frame, process_this_frame, current_time)

            # Draw HUD
            self.draw_hud(annotated_frame, state, current_time)

            # Show frame
            cv2.imshow('Driver Monitor', annotated_frame)
            self.last_frame = annotated_frame  # Store for screenshot

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            self.handle_key(key)

        self.cleanup()

    def process_frame(self, frame: np.ndarray, process: bool, timestamp: float) -> tuple:
        """Process a single frame through the pipeline."""
        h, w = frame.shape[:2]

        # Start profiling
        self.profiler.start_stage('total')

        # Preprocessing
        self.profiler.start_stage('preprocess')
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # Performance optimization
        self.profiler.end_stage('preprocess')

        # FaceMesh (new Tasks API)
        self.profiler.start_stage('face_mesh')
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.face_mesh.detect(mp_image)
        self.profiler.end_stage('face_mesh')

        # Default values
        ear = 0.0
        pitch = yaw = roll = 0.0
        face_detected = False
        phone_detected = False
        yolo_latency_ms = 0.0

        if results.face_landmarks:
            face_detected = True
            landmarks = results.face_landmarks[0]
            # Convert to numpy array (x, y in pixels)
            landmarks_np = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)

            # Calibration
            self.profiler.start_stage('calibration')
            if self.calibration.is_active():
                done = self.calibration.update(landmarks_np, frame.shape)
                if done:
                    self.ear_threshold = self.calibration.get_ear_threshold()
                    self.state_machine.set_ear_threshold(self.ear_threshold)
                    # Apply calibrated neutral pose to the shared feature extractor
                    neutral_yaw, neutral_pitch = self.calibration.get_neutral_pose()
                    self.feature_extractor.set_neutral_pose(neutral_yaw, neutral_pitch)
                    print(f"Calibration complete: EAR threshold = {self.ear_threshold:.3f}, "
                          f"neutral yaw={neutral_yaw:.1f}°, pitch={neutral_pitch:.1f}°")
            self.profiler.end_stage('calibration')

            # Feature extraction
            self.profiler.start_stage('features')
            ear, pitch, yaw, roll = self.feature_extractor.process(landmarks_np, frame.shape)
            self.profiler.end_stage('features')

            # Phone detection (submit frame to background thread)
            if self.phone_detector and process:
                self.phone_detector.submit_frame(frame)

            # Get phone detection result
            if self.phone_detector:
                phone_detected, yolo_latency_ms = self.phone_detector.get_result()
            else:
                # Heuristic: head pitch down for extended period
                phone_detected = pitch > CONFIG.pitch_down_threshold_deg

            # Draw landmarks (minimal)
            if self.show_debug:
                self.draw_minimal_landmarks(frame, landmarks_np)
        else:
            # No face - reset feature smoother
            self.feature_extractor.reset_smoothers()

        # State machine
        self.profiler.start_stage('state_machine')
        state = self.state_machine.update(ear, yaw, pitch, phone_detected, face_detected, timestamp)
        self.profiler.end_stage('state_machine')

        # Drawing
        self.profiler.start_stage('drawing')
        self.draw_alerts(frame, state, pitch, yaw)
        self.profiler.end_stage('drawing')

        # End frame profiling
        self.profiler.end_frame(yolo_latency_ms)

        return frame, state

    def draw_minimal_landmarks(self, frame: np.ndarray, landmarks: np.ndarray):
        """Draw only key landmarks: eye contours and nose tip."""
        # Eye contours
        for idx in CONFIG.right_eye_indices + CONFIG.left_eye_indices:
            x, y = int(landmarks[idx, 0]), int(landmarks[idx, 1])
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Nose tip
        x, y = int(landmarks[1, 0]), int(landmarks[1, 1])
        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

    def draw_alerts(self, frame: np.ndarray, state: DriverState, pitch: float, yaw: float):
        """Draw alert banners based on state."""
        h, w = frame.shape[:2]
        banner_height = 50

        if state == DriverState.DROWSY:
            color = (0, 0, 255)  # Red
            text = "CRITICAL: DROWSINESS ALERT"
            self.play_alert()
        elif state == DriverState.DISTRACTED:
            color = (0, 165, 255)  # Orange
            direction = "LEFT" if yaw < 0 else "RIGHT"
            text = f"DISTRACTION ALERT: Looking {direction}"
            self.play_alert()
        elif state == DriverState.PHONE:
            color = (0, 165, 255)  # Orange
            text = "DISTRACTION ALERT: Phone Use Detected"
            self.play_alert()
        elif state == DriverState.DRIVER_MISSING:
            color = (0, 255, 255)  # Yellow
            text = "WARNING: DRIVER MISSING"
            self.play_alert()
        else:
            return

        cv2.rectangle(frame, (0, 0), (w, banner_height), color, -1)
        cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    def draw_hud(self, frame: np.ndarray, state: DriverState, timestamp: float):
        """Draw heads-up display with metrics."""
        h, w = frame.shape[:2]
        x, y = 10, 30
        line_h = 22
        color = (255, 255, 255)

        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += line_h

        # EAR
        ear_val = self.feature_extractor.ear_smoother.values[-1] if self.feature_extractor.ear_smoother.values else 0.0
        cv2.putText(frame, f"EAR: {ear_val:.3f} / {self.ear_threshold:.3f}", 
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += line_h

        # Head pose (from feature extractor smoothers)
        pitch = self.feature_extractor.pitch_smoother.values[-1] if self.feature_extractor.pitch_smoother.values else 0.0
        yaw = self.feature_extractor.yaw_smoother.values[-1] if self.feature_extractor.yaw_smoother.values else 0.0
        roll = self.feature_extractor.roll_smoother.values[-1] if self.feature_extractor.roll_smoother.values else 0.0
        cv2.putText(frame, f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}  Roll: {roll:.1f}", 
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += line_h

        # State
        state_color = (0, 255, 0) if state == DriverState.ATTENTIVE else (0, 0, 255)
        cv2.putText(frame, f"State: {state.value}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        y += line_h

        # Timers
        timers = self.state_machine.get_active_timers(timestamp)
        for name, value in timers.items():
            if value > 0:
                cv2.putText(frame, f"  {name}: {value:.1f}s", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y += line_h

        # Latencies
        summary = self.profiler.get_summary()
        if summary and 'per_stage' in summary:
            cv2.putText(frame, "Latencies (ms):", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += line_h
            for stage_name, stats in summary['per_stage'].items():
                cv2.putText(frame, f"  {stage_name}: {stats['avg_ms']:.1f}", (x, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
                y += line_h

        # CPU
        if summary and 'cpu_percent' in summary:
            cv2.putText(frame, f"CPU: {summary['cpu_percent']['mean']:.1f}%", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Calibration overlay
        self.calibration.draw_overlay(frame)

    def play_alert(self):
        """Play audible alert with cooldown."""
        if not CONFIG.enable_audio_alerts:
            return
        current_time = time.monotonic()
        if current_time - self.last_alert_time < CONFIG.audio_alert_cooldown_sec:
            return
        self.last_alert_time = current_time

        if HAS_WINSOUND:
            winsound.Beep(1000, 500)  # 1kHz for 500ms
        else:
            # Cross-platform fallback: print bell character
            print('\a', end='', flush=True)

    def handle_key(self, key: int):
        """Handle keyboard input."""
        if key in (27, ord('q')):  # ESC or q
            self.running = False
        elif key == ord('c'):
            print("Recalibrating...")
            self.calibration.start()
            self.feature_extractor.reset_smoothers()
            self.state_machine.reset()
        elif key == ord('d'):
            self.show_debug = not self.show_debug
            print(f"Debug overlay: {'ON' if self.show_debug else 'OFF'}")
        elif key == ord('p'):
            CONFIG.use_phone_detector = not CONFIG.use_phone_detector
            print(f"Phone detector: {'ON' if CONFIG.use_phone_detector else 'OFF (heuristic mode)'}")
            if CONFIG.use_phone_detector and self.phone_detector is None:
                self.phone_detector = create_phone_detector()
            elif not CONFIG.use_phone_detector and self.phone_detector:
                self.phone_detector.stop()
                self.phone_detector = None
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            if self.last_frame is not None:
                cv2.imwrite(filename, self.last_frame)
                print(f"Saved {filename}")
            else:
                print("No frame available for screenshot")

    def cleanup(self):
        """Release resources."""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.face_mesh:
            self.face_mesh.close()
        if self.phone_detector:
            self.phone_detector.stop()
        self.profiler.stop_sampling()
        cv2.destroyAllWindows()
        print("Driver Monitor stopped.")


def main():
    parser = argparse.ArgumentParser(description='Driver Drowsiness & Distraction Detection')
    parser.add_argument('--lite', action='store_true', help='Lite mode: lower resolution, less frequent YOLO')
    parser.add_argument('--video', type=str, help='Path to video file instead of webcam')
    parser.add_argument('--no-phone', action='store_true', help='Disable phone detector (use heuristic)')
    args = parser.parse_args()

    if args.no_phone:
        CONFIG.use_phone_detector = False

    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"OpenCV: {cv2.__version__}")
    import mediapipe as mp
    print(f"MediaPipe: {mp.__version__}")
    import onnxruntime as ort
    print(f"ONNX Runtime: {ort.__version__}")

    monitor = DriverMonitor(lite_mode=args.lite, video_path=args.video)
    monitor.run()


if __name__ == '__main__':
    main()