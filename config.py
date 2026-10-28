"""Configuration dataclass for all thresholds and tunables."""
from dataclasses import dataclass, field


@dataclass
class Config:
    # Camera settings
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    process_every_n_frames: int = 2  # Process at ~15-20 FPS

    # Lite mode settings (activated via --lite flag)
    lite_frame_width: int = 480
    lite_frame_height: int = 360
    lite_process_every_n_frames: int = 3

    # Mediapipe FaceMesh settings
    max_num_faces: int = 1
    refine_landmarks: bool = False
    static_image_mode: bool = False
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # EAR settings
    right_eye_indices: tuple = (33, 160, 158, 133, 153, 144)
    left_eye_indices: tuple = (362, 385, 387, 263, 373, 380)
    ear_smoothing_frames: int = 5
    ear_threshold_multiplier: float = 0.72
    ear_threshold_min: float = 0.12
    ear_threshold_max: float = 0.25
    calibration_duration_sec: float = 3.0

    # Head pose settings
    head_pose_model_points: tuple = field(default_factory=lambda: (
        (0.0, 0.0, 0.0),          # Nose tip (1)
        (0.0, -63.6, -12.5),      # Chin (152)
        (-43.3, 32.7, -26.0),     # Left eye corner (33)
        (43.3, 32.7, -26.0),      # Right eye corner (263)
        (-28.9, -28.9, -24.1),    # Left mouth corner (61)
        (28.9, -28.9, -24.1),     # Right mouth corner (291)
    ))
    head_pose_smoothing_frames: int = 5
    yaw_threshold_deg: float = 25.0
    pitch_down_threshold_deg: float = 15.0  # For phone heuristic

    # State machine thresholds (seconds)
    drowsy_duration_sec: float = 1.5
    distracted_duration_sec: float = 2.0
    phone_duration_sec: float = 1.0
    driver_missing_duration_sec: float = 3.0
    hysteresis_duration_sec: float = 0.5
    blink_min_duration_sec: float = 0.1
    blink_max_duration_sec: float = 0.4

    # Phone detector settings
    use_phone_detector: bool = True
    yolo_imgsz: int = 320
    yolo_conf_threshold: float = 0.4
    yolo_nms_threshold: float = 0.45
    yolo_phone_class_id: int = 67  # COCO class for cell phone
    yolo_inference_interval_sec: float = 0.3  # Max every 300ms
    yolo_inference_every_n_frames: int = 5

    # ONNX Runtime settings
    ort_intra_op_num_threads: int = 2
    # ort_graph_optimization_level is now referenced directly as ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # in phone_detector.py to avoid passing a raw int where an enum is expected.

    # Profiling settings
    enable_profiling: bool = True
    cpu_sample_interval_sec: float = 0.1

    # Alert settings
    enable_audio_alerts: bool = True
    audio_alert_cooldown_sec: float = 2.0

    # Overlay settings
    show_debug_overlay: bool = True
    draw_all_landmarks: bool = False


CONFIG = Config()