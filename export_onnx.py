"""Export YOLOv8n to ONNX format (FP32 and INT8 quantized)."""
import argparse
import os
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install ultralytics onnx onnxruntime onnxruntime-tools")
    sys.exit(1)


def export_yolov8n_onnx(
    model_name: str = "yolov8n.pt",
    output_path: str = "yolov8n_phone.onnx",
    imgsz: int = 320,
    opset: int = 12,
    simplify: bool = True
):
    """Export YOLOv8n to ONNX format."""
    print(f"Loading {model_name}...")
    model = YOLO(model_name)

    print(f"Exporting to ONNX (imgsz={imgsz}, opset={opset})...")
    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=False,
        half=False  # FP32
    )

    # Move to desired output path
    default_path = Path(model_name).with_suffix(".onnx")
    if default_path.exists() and default_path != Path(output_path):
        default_path.rename(output_path)
        print(f"Model saved to {output_path}")
    else:
        print(f"Model saved to {default_path}")


def quantize_onnx_model(
    input_path: str,
    output_path: str = "yolov8n_phone_int8.onnx"
):
    """Apply dynamic INT8 quantization to ONNX model."""
    print(f"Quantizing {input_path} to INT8...")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        op_types_to_quantize=['MatMul', 'Add', 'Conv'],
        weight_type=QuantType.QInt8,
        optimize_model=True
    )
    print(f"Quantized model saved to {output_path}")


def verify_onnx_model(model_path: str):
    """Verify ONNX model loads and runs."""
    print(f"Verifying {model_path}...")
    try:
        # Check with onnx
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print("  ONNX model structure: OK")

        # Check with ONNX Runtime
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, sess_options=so, providers=['CPUExecutionProvider'])

        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_names = [out.name for out in session.get_outputs()]

        print(f"  Input: {input_name} {input_shape}")
        print(f"  Outputs: {output_names}")

        # Test inference with dummy input
        import numpy as np
        dummy_input = np.random.randn(1, 3, input_shape[2], input_shape[3]).astype(np.float32)
        outputs = session.run(output_names, {input_name: dummy_input})
        print(f"  Test inference: OK (output shapes: {[o.shape for o in outputs]})")

        return True
    except Exception as e:
        print(f"  Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8n to ONNX for phone detection")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model to export")
    parser.add_argument("--output", default="yolov8n_phone.onnx", help="Output ONNX path")
    parser.add_argument("--imgsz", type=int, default=320, help="Input image size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--no-simplify", action="store_true", help="Disable ONNX simplification")
    parser.add_argument("--quantize", action="store_true", help="Also create INT8 quantized version")
    parser.add_argument("--verify", action="store_true", help="Verify exported models")
    args = parser.parse_args()

    # Export FP32
    export_yolov8n_onnx(
        model_name=args.model,
        output_path=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify
    )

    # Verify FP32
    if args.verify:
        verify_onnx_model(args.output)

    # Quantize to INT8
    if args.quantize:
        int8_path = args.output.replace(".onnx", "_int8.onnx")
        quantize_onnx_model(args.output, int8_path)
        if args.verify:
            verify_onnx_model(int8_path)

    print("\nDone! Use the ONNX model with phone_detector.py")


if __name__ == "__main__":
    main()