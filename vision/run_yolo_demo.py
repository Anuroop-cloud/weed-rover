"""
MODE A — ML SHOWCASE
Entrypoint for real-time YOLO weed detection demonstration.

Visualization ONLY:
- Captures frames from USB webcam using Camera abstraction
- Runs YOLO object detection via modular YOLODetector
- Overlays bounding boxes, class labels, confidence scores, target centroids, and FPS
- Does NOT contain robot movement logic, TargetController, serial communication, or ROS2
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path so modules can be imported when running from any directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
from vision.camera import Camera
from vision.detector import YOLODetector

try:
    from config.vision_config import (
        CAMERA_DEVICE_INDEX,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        CAMERA_FPS,
        CONFIDENCE_THRESHOLD,
    )
except ImportError:
    CAMERA_DEVICE_INDEX = 1
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    CAMERA_FPS = 30.0
    CONFIDENCE_THRESHOLD = 0.35


DEFAULT_MODEL_PATH = os.path.join("models", "best.pt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mode A: YOLO Weed Detection ML Showcase (Visualization Only)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to YOLO model weights (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--camera",
        type=lambda x: int(x) if x.isdigit() else x,
        default=CAMERA_DEVICE_INDEX,
        help=f"Camera device index (e.g. 0, 1) or phone stream URL (default: {CAMERA_DEVICE_INDEX})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence detection threshold (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=FRAME_WIDTH,
        help=f"Camera capture width in pixels (default: {FRAME_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=FRAME_HEIGHT,
        help=f"Camera capture height in pixels (default: {FRAME_HEIGHT})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=CAMERA_FPS,
        help=f"Target capture frame rate limit (default: {CAMERA_FPS})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw detection coordinates to terminal for debugging",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Scan and list available video capture device indices",
    )
    return parser.parse_args()


def check_model_exists(model_path: str) -> bool:
    """Verifies model file exists or is an official Ultralytics stock model name."""
    if os.path.exists(model_path):
        return True
    stock_models = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"}
    return os.path.basename(model_path).lower() in stock_models


def main():
    args = parse_args()

    # Handle camera discovery CLI option
    if args.list_cameras:
        print("[Camera Scan] Scanning available video capture devices...")
        cams = Camera.list_available_cameras(max_tested=6)
        if cams:
            print(f"Detected camera index(es): {cams}")
            print("Tip: Index 0 is typically the built-in webcam; Index 1+ is an external USB camera.")
        else:
            print("No video capture devices detected.")
        sys.exit(0)

    print("=" * 65)
    print("MODE A — ML SHOWCASE: YOLO WEED DETECTION")
    print("=" * 65)
    print(f"Model Path  : {args.model}")
    print(f"Camera Device: {args.camera}")
    print(f"Resolution   : {args.width}x{args.height} @ {args.fps} FPS")
    print(f"Confidence   : {args.conf}")
    print("-" * 65)

    # 1. Model Availability Check
    if not check_model_exists(args.model):
        print("\n" + "!" * 65)
        print(f"[ERROR] Custom YOLO model not found at: '{args.model}'")
        print("!" * 65)
        print("\nThe custom weed-detection model has not been trained yet.")
        print("To train the model on the prepared dataset:")
        print("  1. Dataset is prepared in: weed_dataset_yolo/data.yaml")
        print("  2. Train YOLO with Ultralytics, e.g.:")
        print("       yolo detect train data=weed_dataset_yolo/data.yaml model=yolov8n.pt epochs=50 imgsz=640")
        print(f"  3. Copy best weights to: '{args.model}'")
        print("\nAlternatively, you can test this showcase right now with a pretrained baseline:")
        print("  python vision/run_yolo_demo.py --model yolov8n.pt")
        print("=" * 65)
        sys.exit(1)

    # 2. Load Modular YOLO Detector
    try:
        detector = YOLODetector(model_path=args.model, conf_threshold=args.conf)
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO detector: {e}")
        sys.exit(1)

    # 3. Initialize Camera Stream
    camera = Camera(
        device_id=args.camera,
        width=args.width,
        height=args.height,
        fps_limit=args.fps,
    )
    if not camera.is_opened and args.camera != 0:
        print(f"[Camera Fallback] Camera index {args.camera} unavailable. Attempting fallback to built-in webcam (index 0)...")
        camera = Camera(
            device_id=0,
            width=args.width,
            height=args.height,
            fps_limit=args.fps,
        )

    if not camera.is_opened:
        print(f"[ERROR] Unable to open camera device (tested index {args.camera} and fallback 0).")
        print("Tip: Run with '--list-cameras' to scan available camera indices.")
        sys.exit(1)

    window_name = "Weed Rover - YOLO ML Showcase (Mode A)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print(f"\n[ML Showcase Ready] Streaming live video. Press 'q' or ESC to exit.\n")

    try:
        while True:
            # Capture frame
            success, frame = camera.read()
            if not success or frame is None:
                print("[WARNING] Frame capture failed. Exiting...")
                break

            # Run YOLO inference
            detections = detector.detect(frame)

            if args.debug and detections:
                for det in detections:
                    print(f"  [Detection] {det.to_debug_string()}")

            # Render bounding boxes, class names, confidence, and centroids
            annotated_frame = detector.draw_detections(frame, detections)

            # Overlay FPS and detection count
            Camera.draw_fps(
                annotated_frame,
                fps=camera.get_fps(),
                detection_count=len(detections),
            )

            # Display output window
            cv2.imshow(window_name, annotated_frame)

            # Handle exit keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("[ML Showcase] Exit requested by user.")
                break

    except KeyboardInterrupt:
        print("\n[ML Showcase] Interrupted by keyboard.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[ML Showcase] Camera released and windows closed. Done.")


if __name__ == "__main__":
    main()
