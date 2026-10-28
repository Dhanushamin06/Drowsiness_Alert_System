"""Profiling module for per-stage timing and CPU/RAM sampling."""
import time
import psutil
import csv
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from config import CONFIG


@dataclass
class StageTiming:
    """Timing data for a single pipeline stage."""
    name: str
    total_time: float = 0.0
    count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0

    def update(self, elapsed: float):
        self.total_time += elapsed
        self.count += 1
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)

    def get_avg(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0

    def get_min(self) -> float:
        return self.min_time if self.count > 0 else 0.0

    def get_max(self) -> float:
        return self.max_time if self.count > 0 else 0.0


@dataclass
class FrameProfile:
    """Complete profile for a single frame."""
    timestamp: float
    stage_times: Dict[str, float]
    total_time: float
    cpu_percent: float
    ram_mb: float
    yolo_latency_ms: float = 0.0


class Profiler:
    """Profiles pipeline stages and system resources."""
    def __init__(self):
        self.stages: Dict[str, StageTiming] = {}
        self.frame_profiles: List[FrameProfile] = []
        self.current_frame_stages: Dict[str, float] = {}
        self.stage_start_times: Dict[str, float] = {}
        self.cpu_process = psutil.Process()
        self.cpu_process.cpu_percent(interval=None)  # Initialize
        self.sampling_thread: Optional[threading.Thread] = None
        self.sampling_running: bool = False
        self.cpu_samples: List[float] = []
        self.ram_samples: List[float] = []
        self._lock = threading.Lock()

    def start_stage(self, name: str):
        """Start timing a stage."""
        if not CONFIG.enable_profiling:
            return
        self.stage_start_times[name] = time.perf_counter()

    def end_stage(self, name: str):
        """End timing a stage."""
        if not CONFIG.enable_profiling:
            return
        if name in self.stage_start_times:
            elapsed = time.perf_counter() - self.stage_start_times[name]
            self.current_frame_stages[name] = elapsed
            if name not in self.stages:
                self.stages[name] = StageTiming(name)
            self.stages[name].update(elapsed)

    def end_frame(self, yolo_latency_ms: float = 0.0):
        """Finalize frame profile."""
        if not CONFIG.enable_profiling:
            return

        total_time = sum(self.current_frame_stages.values())
        cpu_percent = self.cpu_process.cpu_percent(interval=None)
        ram_mb = self.cpu_process.memory_info().rss / (1024 * 1024)

        profile = FrameProfile(
            timestamp=time.monotonic(),
            stage_times=self.current_frame_stages.copy(),
            total_time=total_time,
            cpu_percent=cpu_percent,
            ram_mb=ram_mb,
            yolo_latency_ms=yolo_latency_ms
        )

        with self._lock:
            self.frame_profiles.append(profile)
            self.cpu_samples.append(cpu_percent)
            self.ram_samples.append(ram_mb)

        self.current_frame_stages.clear()

    def start_sampling(self):
        """Start background CPU/RAM sampling."""
        if self.sampling_running:
            return
        self.sampling_running = True
        self.sampling_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self.sampling_thread.start()

    def stop_sampling(self):
        """Stop background sampling."""
        self.sampling_running = False
        if self.sampling_thread:
            self.sampling_thread.join(timeout=1.0)

    def _sampling_loop(self):
        """Background sampling loop."""
        while self.sampling_running:
            cpu = self.cpu_process.cpu_percent(interval=None)
            ram = self.cpu_process.memory_info().rss / (1024 * 1024)
            with self._lock:
                self.cpu_samples.append(cpu)
                self.ram_samples.append(ram)
            time.sleep(CONFIG.cpu_sample_interval_sec)

    def get_summary(self) -> Dict:
        """Get profiling summary statistics."""
        if not self.frame_profiles:
            return {}

        total_times = [p.total_time for p in self.frame_profiles]
        cpu_vals = [p.cpu_percent for p in self.frame_profiles]
        ram_vals = [p.ram_mb for p in self.frame_profiles]
        yolo_times = [p.yolo_latency_ms for p in self.frame_profiles if p.yolo_latency_ms > 0]

        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]

        return {
            'frame_count': len(self.frame_profiles),
            'total_time': {
                'mean': np.mean(total_times) * 1000,
                'median': np.median(total_times) * 1000,
                'p95': percentile(total_times, 95) * 1000,
                'p99': percentile(total_times, 99) * 1000,
                'min': min(total_times) * 1000,
                'max': max(total_times) * 1000,
            },
            'fps': 1.0 / np.mean(total_times) if np.mean(total_times) > 0 else 0,
            'cpu_percent': {
                'mean': np.mean(cpu_vals),
                'median': np.median(cpu_vals),
                'max': max(cpu_vals),
            },
            'ram_mb': {
                'mean': np.mean(ram_vals),
                'max': max(ram_vals),
            },
            'yolo_latency_ms': {
                'mean': np.mean(yolo_times) if yolo_times else 0,
                'median': np.median(yolo_times) if yolo_times else 0,
                'p95': percentile(yolo_times, 95) if yolo_times else 0,
            },
            'per_stage': {
                name: {
                    'avg_ms': stage.get_avg() * 1000,
                    'min_ms': stage.get_min() * 1000,
                    'max_ms': stage.get_max() * 1000,
                    'count': stage.count,
                }
                for name, stage in self.stages.items()
            }
        }

    def export_csv(self, filepath: str):
        """Export frame profiles to CSV."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            stage_names = sorted(self.stages.keys())
            header = ['timestamp', 'total_ms', 'cpu_percent', 'ram_mb', 'yolo_latency_ms'] + \
                     [f'{name}_ms' for name in stage_names]
            writer.writerow(header)

            for profile in self.frame_profiles:
                row = [
                    profile.timestamp,
                    profile.total_time * 1000,
                    profile.cpu_percent,
                    profile.ram_mb,
                    profile.yolo_latency_ms,
                ]
                for name in stage_names:
                    row.append(profile.stage_times.get(name, 0) * 1000)
                writer.writerow(row)

    def clear(self):
        """Clear all profiling data."""
        self.stages.clear()
        self.frame_profiles.clear()
        self.cpu_samples.clear()
        self.ram_samples.clear()
        self.current_frame_stages.clear()
        self.stage_start_times.clear()