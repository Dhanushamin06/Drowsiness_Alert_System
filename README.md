# Driver Drowsiness & Distraction Detection

A real-time desktop application for monitoring driver alertness using a standard webcam. Designed to run at 15+ FPS on low-end Intel Core i3 laptops (2 cores / 4 threads, 4-8 GB RAM, no GPU).

## Features

- **Drowsiness Detection**: Eye Aspect Ratio (EAR) monitoring with calibrated threshold
- **Distraction Detection**: Head pose estimation (yaw/pitch/roll) via solvePnP
- **Phone Use Detection**: YOLOv8n ONNX model running in background thread
- **Driver Missing Detection**: Face loss timeout
- **Timestamp-based State Machine**: FPS-independent alert timing with hysteresis
- **Startup Calibration**: Automatic EAR threshold and neutral head pose calibration
- **Low-CPU Design**: Optimized MediaPipe settings, minimal drawing, threaded inference
- **Profiling & Benchmarking**: Per-stage latency, CPU/RAM tracking, automated graphs

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Loop (15-20 FPS)                    │
├─────────────────────────────────────────────────────────────────┤
│  Camera Capture → Preprocess → MediaPipe FaceMesh → Features   │
│       (640×480)       (RGB)      (468 landmarks)   (EAR+Pose)  │
├─────────────────────────────────────────────────────────────────┤
│                    State Machine (timestamp-based)              │
│  DRIVER_MISSING > DROWSY > PHONE > DISTRACTED > ATTENTIVE      │
├─────────────────────────────────────────────────────────────────┤
│  Overlay & Alerts → Display (OpenCV window)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (background thread, ≤300ms interval)
┌─────────────────────────────────────────────────────────────────┐
│                  Phone Detector (YOLOv8n ONNX)                  │
│  Input: 320×320 → ONNX Runtime (2 threads) → Class 67 (phone)  │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Computer Vision | OpenCV | 4.8.1 |
| Face Landmarks | MediaPipe | 0.10.14 (legacy `mp.solutions.face_mesh`) |
| Numerical | NumPy | 1.24.3 |
| Phone Detection | ONNX Runtime | 1.16.0 |
| Model Export | Ultralytics | 8.0.196 (YOLOv8n export only) |
| System Monitoring | psutil | 5.9.5 |
| Plotting | Matplotlib | 3.7.2 |

**No PyTorch at runtime.** All inference uses ONNX Runtime (CPU) and MediaPipe (TFLite backend).

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Export YOLOv8n to ONNX (run once)
python export_onnx.py --quantize --verify
```

## Usage

```bash
# Standard mode (640×480, YOLO every 5 frames)
python main.py

# Lite mode (480×360, YOLO less frequent)
python main.py --lite

# Use video file instead of webcam
python main.py --video path/to/video.mp4

# Disable phone detector (use pitch-down heuristic)
python main.py --no-phone
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `ESC` / `q` | Quit |
| `c` | Recalibrate (look straight, eyes open) |
| `d` | Toggle debug overlay (landmarks) |
| `p` | Toggle phone detector on/off |
| `s` | Save screenshot |

## Calibration

On startup (and when pressing `c`), the system collects 3 seconds of data while you:
1. Look straight at the road (camera)
2. Keep eyes open normally

It computes:
- **EAR threshold** = 0.72 × mean open-eye EAR (clamped to 0.12–0.25)
- **Neutral yaw/pitch** = mean head angles during calibration

All subsequent measurements are relative to this baseline.

## Thresholds (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `drowsy_duration_sec` | 1.5 | Eyes closed continuously to trigger DROWSY |
| `distracted_duration_sec` | 2.0 | \|yaw\| > 25° continuously to trigger DISTRACTED |
| `phone_duration_sec` | 1.0 | Phone detected continuously to trigger PHONE |
| `driver_missing_duration_sec` | 3.0 | No face detected to trigger DRIVER_MISSING |
| `hysteresis_duration_sec` | 0.5 | Hold alert state after condition clears |
| `blink_min/max_duration_sec` | 0.1 / 0.4 | Blink duration range (ignored for drowsy) |
| `yaw_threshold_deg` | 25.0 | Head turn threshold for distraction |
| `pitch_down_threshold_deg` | 15.0 | Pitch down threshold for phone heuristic |
| `ear_threshold_multiplier` | 0.72 | Calibration multiplier for EAR threshold |

## Benchmarking

```bash
# Run benchmark (1000 frames, saves CSV + graphs to ./assets/)
python benchmark.py

# With video file for reproducibility
python benchmark.py --video path/to/video.mp4

# Lite mode benchmark
python benchmark.py --lite
```

Outputs:
- `assets/benchmark_results.csv` — Per-frame timings
- `assets/latency_histogram.png` — Total latency distribution (mean/P95/P99)
- `assets/stage_latencies.png` — Per-stage average latency
- `assets/cpu_over_time.png` — CPU utilization timeline
- `assets/yolo_comparison.png` — YOLO latency comparison (template)

## Benchmark Results

> **Fill in after running `benchmark.py`**

| Metric | Value |
|--------|-------|
| **CPU Model** | *e.g., Intel Core i3-8130U @ 2.20GHz* |
| **Resolution** | 640×480 (standard) / 480×360 (lite) |
| **Avg FPS** | ___ |
| **Core Pipeline Latency** | |
| &nbsp;&nbsp;Mean | ___ ms |
| &nbsp;&nbsp;Median | ___ ms |
| &nbsp;&nbsp;P95 | ___ ms |
| &nbsp;&nbsp;P99 | ___ ms |
| **Avg CPU Usage** | ___ % |
| **Avg RAM Usage** | ___ MB |
| **YOLO Latency (ONNX FP32)** | ___ ms |
| **YOLO Latency (ONNX INT8)** | ___ ms |

### Graphs
*(Generated in `./assets/` after benchmark)*

![Latency Histogram](assets/latency_histogram.png)
![Stage Latencies](assets/stage_latencies.png)
![CPU Over Time](assets/cpu_over_time.png)
![YOLO Comparison](assets/yolo_comparison.png)

## How It Works

### MediaPipe FaceMesh (TFLite)
- Runs on CPU via TensorFlow Lite delegate
- Configured for single face, no refinement, tracking mode
- `image.flags.writeable = False` avoids unnecessary copies

### Head Pose (solvePnP)
- 6 2D-3D point correspondences: nose tip, chin, eye corners, mouth corners
- Generic 3D face model (metric coordinates)
- Camera matrix: focal length = frame width, principal point = image center
- Euler angles extracted from rotation matrix, normalized to ±90°

### EAR Calculation
```
EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
```
Using 6 landmarks per eye (MediaPipe indices 33,160,158,133,153,144 for right; 362,385,387,263,373,380 for left).

### State Machine
- **Time-based**: Uses `time.monotonic()` for FPS-invariant behavior
- **Priority**: DRIVER_MISSING > DROWSY > PHONE > DISTRACTED > ATTENTIVE
- **Hysteresis**: 0.5s hold after condition clears to prevent flicker
- **Blink rejection**: 0.1–0.4s eye closures ignored

### Phone Detection
- YOLOv8n exported to ONNX (imgsz=320, opset=12, simplified)
- ONNX Runtime: 2 intra-op threads, all graph optimizations
- Runs in separate thread with bounded queue (size 1), drops stale frames
- Max inference rate: every 300ms or every 5th frame
- COCO class 67 (cell phone), conf ≥ 0.4, NMS 0.45
- INT8 dynamic quantization available for faster inference

## Known Limitations

- **Glasses/sunglasses**: May reduce EAR accuracy or block eye landmarks
- **Low light**: MediaPipe detection confidence drops significantly
- **Camera angle**: Assumes roughly frontal view; extreme angles degrade pose estimation
- **No IR**: Visible-light only; performance degrades in darkness
- **Single face**: Only monitors the largest/most confident face
- **Heuristic phone mode**: Pitch-down detection has false positives (looking at dashboard)

## Disclaimer

> **This is a demonstration/educational system. It is NOT a safety-certified driver monitoring system and must NOT be used in production vehicles or safety-critical applications. False positives and false negatives will occur. Always maintain full attention while driving.**

## Future Work

- [ ] IR camera support for night operation
- [ ] Trained drowsiness classifier (PERCLOS, blink rate, gaze)
- [ ] Raspberry Pi 4 / Jetson Nano deployment with TensorRT
- [ ] Multi-face support with driver identification
- [ ] CAN bus integration for vehicle speed/context
- [ ] Mobile app companion for alerts/logging

## License

MIT License — see LICENSE file for details.

## Acknowledgments

- MediaPipe team for FaceMesh
- Ultralytics for YOLOv8
- ONNX Runtime team for CPU inference