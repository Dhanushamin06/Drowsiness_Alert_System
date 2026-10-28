"""ONNX Runtime YOLOv8n phone detector running in background thread."""
import threading
import queue
import time
import cv2
import numpy as np
import onnxruntime as ort
from typing import Optional, Tuple, List
from config import CONFIG


class PhoneDetector:
    """YOLOv8n phone detector using ONNX Runtime in a separate thread."""
    def __init__(self, model_path: str = "yolov8n_phone.onnx"):
        self.model_path = model_path
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_names: List[str] = []
        self.input_shape: Tuple[int, int] = (CONFIG.yolo_imgsz, CONFIG.yolo_imgsz)

        # Threading
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.result_queue: queue.Queue = queue.Queue(maxsize=1)
        self.worker_thread: Optional[threading.Thread] = None
        self.running: bool = False
        self.last_inference_time: float = 0.0
        self.last_result: Optional[List] = None
        self.enabled: bool = CONFIG.use_phone_detector

        # Latency tracking
        self.last_latency_ms: float = 0.0

    def initialize(self) -> bool:
        """Initialize ONNX Runtime session."""
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = CONFIG.ort_intra_op_num_threads
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=so,
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [out.name for out in self.session.get_outputs()]
            return True
        except Exception as e:
            print(f"Failed to initialize phone detector: {e}")
            return False

    def start(self):
        """Start background worker thread."""
        if not self.enabled or self.session is None:
            return
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """Stop background worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
            self.worker_thread = None

    def submit_frame(self, frame: np.ndarray):
        """Submit frame for processing (non-blocking, drops stale frames)."""
        if not self.enabled or not self.running:
            return
        try:
            self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            # Drop stale frame
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame.copy())
            except queue.Empty:
                pass

    def get_result(self) -> Tuple[bool, float]:
        """
        Get latest detection result.
        Returns: (phone_detected, latency_ms)
        """
        if not self.enabled:
            # Heuristic mode: return False, let state machine handle pitch-based detection
            return False, 0.0

        try:
            result = self.result_queue.get_nowait()
            self.last_result = result
            return result
        except queue.Empty:
            if self.last_result is not None:
                return self.last_result
            return False, 0.0

    def _worker_loop(self):
        """Background thread loop for inference."""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            current_time = time.monotonic()
            # Throttle inference rate
            if current_time - self.last_inference_time < CONFIG.yolo_inference_interval_sec:
                # Re-queue frame for later (but don't block)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
                time.sleep(0.01)
                continue

            self.last_inference_time = current_time
            start_time = time.perf_counter()

            try:
                detections = self._infer(frame)
                phone_detected = self._postprocess(detections, frame.shape[:2])
                latency_ms = (time.perf_counter() - start_time) * 1000
                self.last_latency_ms = latency_ms

                try:
                    self.result_queue.put_nowait((phone_detected, latency_ms))
                except queue.Full:
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait((phone_detected, latency_ms))
                    except queue.Empty:
                        pass
            except Exception as e:
                print(f"Phone detector inference error: {e}")

    def _infer(self, frame: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
        # Preprocess
        input_tensor = self._preprocess(frame)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return outputs[0]

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLOv8n (letterbox resize, normalize, NCHW)."""
        h, w = frame.shape[:2]
        target_h, target_w = self.input_shape

        # Letterbox resize
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        # Normalize and convert to NCHW
        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)   # Add batch dim
        return input_tensor

    def _postprocess(self, output: np.ndarray, orig_shape: Tuple[int, int]) -> bool:
        """
        Postprocess YOLOv8 output.
        Output shape: (1, 84, 8400) for YOLOv8n - (x, y, w, h, class_scores...)
        """
        # YOLOv8 output format: (batch, 84, 8400) where 84 = 4 (bbox) + 80 (classes)
        predictions = output[0]  # (84, 8400)
        predictions = predictions.T  # (8400, 84)

        # Extract boxes and scores
        boxes = predictions[:, :4]  # x_center, y_center, width, height (normalized to input size)
        class_scores = predictions[:, 4:]  # 80 classes

        # Get max class score and class ID for each prediction
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Filter by confidence threshold
        mask = max_scores >= CONFIG.yolo_conf_threshold
        boxes = boxes[mask]
        scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return False

        # Filter for phone class only
        phone_mask = class_ids == CONFIG.yolo_phone_class_id
        boxes = boxes[phone_mask]
        scores = scores[phone_mask]

        if len(boxes) == 0:
            return False

        # Convert to xyxy format in original image coordinates
        boxes_xyxy = self._xywh2xyxy(boxes)
        boxes_xyxy = self._scale_boxes(boxes_xyxy, self.input_shape, orig_shape)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(), scores.tolist(),
            CONFIG.yolo_conf_threshold, CONFIG.yolo_nms_threshold
        )

        if len(indices) > 0:
            return True
        return False

    def _xywh2xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert boxes from xywh to xyxy format."""
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return xyxy

    def _scale_boxes(self, boxes: np.ndarray, input_shape: Tuple[int, int], orig_shape: Tuple[int, int]) -> np.ndarray:
        """Scale boxes from input size to original image size."""
        input_h, input_w = input_shape
        orig_h, orig_w = orig_shape

        # Calculate scaling from letterbox
        scale = min(input_w / orig_w, input_h / orig_h)
        pad_x = (input_w - orig_w * scale) / 2
        pad_y = (input_h - orig_h * scale) / 2

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

        # Clip to image bounds
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        return boxes

    def is_running(self) -> bool:
        return self.running and self.enabled

    def get_latency_ms(self) -> float:
        return self.last_latency_ms


def create_phone_detector(model_path: str = "yolov8n_phone.onnx") -> PhoneDetector:
    """Factory function to create and initialize phone detector."""
    detector = PhoneDetector(model_path)
    if detector.initialize():
        detector.start()
    return detector