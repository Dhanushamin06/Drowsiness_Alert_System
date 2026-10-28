# Setup and Run Instructions

## 1. Create Virtual Environment & Install Dependencies

```powershell
cd E:\Drowsiness_detection_machine\driver-monitor

# Create venv
python -m venv venv

# Activate venv
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## 2. Export YOLOv8n to ONNX (run once)

```powershell
# Export FP32 ONNX (imgsz=320, opset=12, simplified)
python export_onnx.py --quantize --verify

# This creates:
# - yolov8n_phone.onnx (FP32)
# - yolov8n_phone_int8.onnx (INT8 quantized)
```

## 3. Run the Demo

```powershell
# Standard mode (640x480, webcam)
python main.py

# Lite mode (480x360, less frequent YOLO)
python main.py --lite

# Use video file instead of webcam
python main.py --video path\to\video.mp4

# Disable phone detector (use pitch-down heuristic)
python main.py --no-phone
```

### Keyboard Controls
| Key | Action |
|-----|--------|
| `ESC` / `q` | Quit |
| `c` | Recalibrate (look straight, eyes open 3s) |
| `d` | Toggle debug overlay |
| `p` | Toggle phone detector |
| `s` | Save screenshot |

## 4. Run Benchmark

```powershell
# Benchmark with webcam (1000 frames)
python benchmark.py

# Benchmark with video file (reproducible)
python benchmark.py --video path\to\video.mp4

# Lite mode benchmark
python benchmark.py --lite

# Disable phone detector for benchmark
python benchmark.py --no-phone
```

Outputs to `./assets/`:
- `benchmark_results.csv` — per-frame timings
- `latency_histogram.png` — latency distribution
- `stage_latencies.png` — per-stage breakdown
- `cpu_over_time.png` — CPU timeline
- `yolo_comparison.png` — YOLO latency comparison

## 5. Run Unit Tests

```powershell
# Run state machine tests
python -m pytest tests/test_state_machine.py -v

# Or with unittest
python -m unittest tests.test_state_machine -v
```

## Expected Files After Setup

```
driver-monitor/
├── venv/                    # Virtual environment
├── yolov8n_phone.onnx       # FP32 ONNX model
├── yolov8n_phone_int8.onnx  # INT8 quantized model
├── assets/
│   ├── benchmark_results.csv
│   ├── latency_histogram.png
│   ├── stage_latencies.png
│   ├── cpu_over_time.png
│   └── yolo_comparison.png
├── main.py
├── config.py
├── features.py
├── calibration.py
├── state_machine.py
├── phone_detector.py
├── profiler.py
├── export_onnx.py
├── benchmark.py
├── requirements.txt
├── README.md
├── RUN_INSTRUCTIONS.md
├── .gitignore
└── tests/
    └── test_state_machine.py
```

## Troubleshooting

### Camera not opening
- Try different camera index: edit `CONFIG.camera_index` in `config.py`
- On Windows, `CAP_DSHOW` backend is used by default

### Low FPS
- Use `--lite` flag
- Reduce `CONFIG.yolo_imgsz` to 256 in `config.py`
- Increase `CONFIG.process_every_n_frames`
- Disable phone detector: `--no-phone`

### MediaPipe errors
- Ensure mediapipe version is exactly 0.10.14
- Some newer versions changed the API

### ONNX Runtime errors
- Verify model exists: `yolov8n_phone.onnx`
- Re-run `export_onnx.py --verify` to validate

### Module import errors
- Ensure venv is activated: `venv\Scripts\activate`
- Reinstall: `pip install -r requirements.txt`