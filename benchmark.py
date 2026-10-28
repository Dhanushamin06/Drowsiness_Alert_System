"""Benchmark script: runs pipeline, saves CSV, generates graphs."""
import argparse
import csv
import time
import sys
import platform
import subprocess
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from config import CONFIG
from features import FeatureExtractor
from calibration import Calibration
from state_machine import StateMachine
from phone_detector import PhoneDetector, create_phone_detector
from profiler import Profiler
import mediapipe as mp


def get_cpu_model():
    """Get CPU model name."""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip() and "Name" not in line:
                    return line.strip()
        else:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown"


class BenchmarkRunner:
    """Runs benchmark and generates reports."""
    def __init__(self, video_path: str = None, frames: int = 1000, lite_mode: bool = False):
        self.video_path = video_path
        self.target_frames = frames
        self.lite_mode = lite_mode
        self.cpu_model = get_cpu_model()

        # Configure resolution
        if lite_mode:
            self.frame_width = CONFIG.lite_frame_width
            self.frame_height = CONFIG.lite_frame_height
            self.process_every_n = CONFIG.lite_process_every_n_frames
        else:
            self.frame_width = CONFIG.frame_width
            self.frame_height = CONFIG.frame_height
            self.process_every_n = CONFIG.process_every_n_frames

        # Components
        self.cap = None
        self.face_mesh = None
        self.feature_extractor = FeatureExtractor()
        self.calibration = Calibration()
        self.state_machine = StateMachine()
        self.phone_detector = None
        self.profiler = Profiler()
        self.ear_threshold: float = 0.2

        # Data
        self.frame_count = 0
        self.processed_count = 0
        self.start_time = 0

    def initialize(self) -> bool:
        """Initialize all components."""
        # Video source
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            self.cap = cv2.VideoCapture(CONFIG.camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print(f"Error: Could not open video source")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # MediaPipe FaceLandmarker (new Tasks API for MediaPipe >= 0.10)
        import os
        model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
        if not os.path.exists(model_path):
            print(f"Error: MediaPipe model not found at {model_path}")
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
                print("Phone detector failed, disabling")
                CONFIG.use_phone_detector = False
                self.phone_detector = None

        # Profiler
        self.profiler.start_sampling()

        # Quick calibration (1 second)
        self.calibration.start()
        CONFIG.calibration_duration_sec = 1.0

        return True

    def run(self):
        """Run benchmark loop."""
        if not self.initialize():
            return False

        self.start_time = time.monotonic()
        print(f"Running benchmark: {self.target_frames} frames...")
        print(f"CPU: {self.cpu_model}")
        print(f"Resolution: {self.frame_width}x{self.frame_height}")
        print(f"Process every {self.process_every_n} frames")

        while self.processed_count < self.target_frames:
            ret, frame = self.cap.read()
            if not ret:
                if self.video_path:
                    # Loop video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            self.frame_count += 1
            timestamp = time.monotonic()

            # Process every N frames
            if self.frame_count % self.process_every_n != 0:
                continue

            self.processed_count += 1
            self.process_frame(frame, timestamp)

            # Progress
            if self.processed_count % 100 == 0:
                elapsed = time.monotonic() - self.start_time
                fps = self.processed_count / elapsed
                print(f"  {self.processed_count}/{self.target_frames} frames ({fps:.1f} FPS)")

        elapsed = time.monotonic() - self.start_time
        print(f"\nCompleted {self.processed_count} frames in {elapsed:.1f}s ({self.processed_count/elapsed:.1f} FPS)")

        self.cleanup()
        return True

    def process_frame(self, frame: np.ndarray, timestamp: float):
        """Process single frame."""
        h, w = frame.shape[:2]

        # Preprocess
        self.profiler.start_stage('preprocess')
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        self.profiler.end_stage('preprocess')

        # FaceMesh (new Tasks API)
        self.profiler.start_stage('face_mesh')
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.face_mesh.detect(mp_image)
        self.profiler.end_stage('face_mesh')

        ear = pitch = yaw = roll = 0.0
        face_detected = False
        phone_detected = False
        yolo_latency_ms = 0.0

        if results.face_landmarks:
            face_detected = True
            landmarks = results.face_landmarks[0]
            landmarks_np = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)

            # Calibration (fast)
            if self.calibration.is_active():
                self.calibration.update(landmarks_np, frame.shape)
                if not self.calibration.is_active():
                    self.ear_threshold = self.calibration.get_ear_threshold()
                    self.state_machine.set_ear_threshold(self.ear_threshold)

            # Features
            self.profiler.start_stage('features')
            ear, pitch, yaw, roll = self.feature_extractor.process(landmarks_np, frame.shape)
            self.profiler.end_stage('features')

            # Phone detection
            if self.phone_detector:
                self.phone_detector.submit_frame(frame)
                phone_detected, yolo_latency_ms = self.phone_detector.get_result()

        # State machine
        self.profiler.start_stage('state_machine')
        self.state_machine.update(ear, yaw, pitch, phone_detected, face_detected, timestamp)
        self.profiler.end_stage('state_machine')

        # End frame
        self.profiler.end_frame(yolo_latency_ms)

    def cleanup(self):
        """Cleanup resources."""
        if self.cap:
            self.cap.release()
        if self.face_mesh:
            self.face_mesh.close()
        if self.phone_detector:
            self.phone_detector.stop()
        self.profiler.stop_sampling()

    def save_results(self, output_dir: Path):
        """Save CSV and generate graphs."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export CSV
        csv_path = output_dir / "benchmark_results.csv"
        self.profiler.export_csv(str(csv_path))
        print(f"CSV saved to {csv_path}")

        # Generate graphs
        self.generate_graphs(output_dir)

        # Print summary
        self.print_summary()

    def generate_graphs(self, output_dir: Path):
        """Generate benchmark graphs."""
        summary = self.profiler.get_summary()
        if not summary:
            print("No profiling data to graph")
            return

        # 1. Latency histogram
        self._plot_latency_histogram(output_dir)

        # 2. Per-stage bar chart
        self._plot_stage_latencies(output_dir, summary)

        # 3. CPU over time
        self._plot_cpu_over_time(output_dir)

        # 4. YOLO comparison (placeholder - would need separate runs)
        self._plot_yolo_comparison(output_dir)

        print(f"Graphs saved to {output_dir}")

    def _plot_latency_histogram(self, output_dir: Path):
        """Plot total latency histogram with percentiles."""
        total_times = [p.total_time * 1000 for p in self.profiler.frame_profiles]

        plt.figure(figsize=(10, 6))
        plt.hist(total_times, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        plt.axvline(np.mean(total_times), color='red', linestyle='--', label=f'Mean: {np.mean(total_times):.1f}ms')
        plt.axvline(np.percentile(total_times, 95), color='orange', linestyle='--', label=f'P95: {np.percentile(total_times, 95):.1f}ms')
        plt.axvline(np.percentile(total_times, 99), color='green', linestyle='--', label=f'P99: {np.percentile(total_times, 99):.1f}ms')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.title('Core Pipeline Latency Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "latency_histogram.png", dpi=150)
        plt.close()

    def _plot_stage_latencies(self, output_dir: Path, summary: dict):
        """Plot per-stage average latency bar chart."""
        stages = summary['per_stage']
        names = list(stages.keys())
        avgs = [stages[n]['avg_ms'] for n in names]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, avgs, color='steelblue', edgecolor='black')
        plt.xlabel('Pipeline Stage')
        plt.ylabel('Average Latency (ms)')
        plt.title('Per-Stage Average Latency')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, avg in zip(bars, avgs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{avg:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_dir / "stage_latencies.png", dpi=150)
        plt.close()

    def _plot_cpu_over_time(self, output_dir: Path):
        """Plot CPU utilization over time."""
        cpu_samples = self.profiler.cpu_samples
        if not cpu_samples:
            return

        times = np.arange(len(cpu_samples)) * CONFIG.cpu_sample_interval_sec

        plt.figure(figsize=(10, 6))
        plt.plot(times, cpu_samples, color='steelblue', linewidth=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel('CPU Usage (%)')
        plt.title('CPU Utilization Over Time')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max(100, max(cpu_samples) * 1.1))
        plt.tight_layout()
        plt.savefig(output_dir / "cpu_over_time.png", dpi=150)
        plt.close()

    def _plot_yolo_comparison(self, output_dir: Path):
        """Plot YOLO latency comparison (placeholder)."""
        # This would be filled in after running separate benchmarks
        # For now, create a template
        plt.figure(figsize=(8, 5))
        models = ['PyTorch .pt\n(not run)', 'ONNX FP32\n(measured)', 'ONNX INT8\n(not run)']
        latencies = [0, np.mean([p.yolo_latency_ms for p in self.profiler.frame_profiles if p.yolo_latency_ms > 0]) or 0, 0]
        colors = ['lightgray', 'steelblue', 'lightgray']

        bars = plt.bar(models, latencies, color=colors, edgecolor='black')
        plt.ylabel('Average Latency (ms)')
        plt.title('YOLO Phone Detection Latency Comparison')
        plt.grid(True, alpha=0.3, axis='y')

        for bar, lat in zip(bars, latencies):
            if lat > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{lat:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_dir / "yolo_comparison.png", dpi=150)
        plt.close()

    def print_summary(self):
        """Print benchmark summary to terminal."""
        summary = self.profiler.get_summary()
        if not summary:
            return

        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"CPU Model: {self.cpu_model}")
        print(f"Frames Processed: {summary['frame_count']}")
        print(f"Resolution: {self.frame_width}x{self.frame_height}")
        print(f"Avg FPS: {summary['fps']:.1f}")
        print()
        print("Core Pipeline Latency:")
        t = summary['total_time']
        print(f"  Mean:   {t['mean']:.1f} ms")
        print(f"  Median: {t['median']:.1f} ms")
        print(f"  P95:    {t['p95']:.1f} ms")
        print(f"  P99:    {t['p99']:.1f} ms")
        print()
        print("CPU Usage:")
        c = summary['cpu_percent']
        print(f"  Mean: {c['mean']:.1f}%")
        print(f"  Max:  {c['max']:.1f}%")
        print()
        print("RAM Usage:")
        r = summary['ram_mb']
        print(f"  Mean: {r['mean']:.1f} MB")
        print(f"  Max:  {r['max']:.1f} MB")
        print()
        if summary['yolo_latency_ms']['mean'] > 0:
            print("YOLO Inference Latency:")
            y = summary['yolo_latency_ms']
            print(f"  Mean: {y['mean']:.1f} ms")
            print(f"  P95:  {y['p95']:.1f} ms")
        print()
        print("Per-Stage Latency:")
        for name, stats in summary['per_stage'].items():
            print(f"  {name:20s}: {stats['avg_ms']:.1f} ms (avg)")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark driver monitoring pipeline")
    parser.add_argument("--video", type=str, help="Path to video file (for reproducibility)")
    parser.add_argument("--frames", type=int, default=1000, help="Number of frames to process")
    parser.add_argument("--lite", action="store_true", help="Lite mode (lower res, less YOLO)")
    parser.add_argument("--output", type=str, default="./assets", help="Output directory")
    parser.add_argument("--no-phone", action="store_true", help="Disable phone detector")
    args = parser.parse_args()

    if args.no_phone:
        CONFIG.use_phone_detector = False

    print("Driver Monitor Benchmark")
    print(f"OpenCV: {cv2.__version__}, NumPy: {np.__version__}")

    runner = BenchmarkRunner(
        video_path=args.video,
        frames=args.frames,
        lite_mode=args.lite
    )

    if runner.run():
        runner.save_results(Path(args.output))
    else:
        print("Benchmark failed")
        sys.exit(1)


if __name__ == "__main__":
    main()