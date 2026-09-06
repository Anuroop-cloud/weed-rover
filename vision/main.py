"""
Main Vision Pipeline for weed-rover.
Orchestrates: Camera Capture -> YOLO Inference -> Visual Annotations -> Real-time Display.
"""

import argparse
import sys
import cv2

from camera import Camera
from detector import YOLODetector


def parse_args():
    parser = argparse.ArgumentParser(description="Weed Rover Vision Pipeline")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Camera capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height (default: 480)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model weights (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Starting Weed Rover Vision Pipeline")
    print(f"Device: {args.camera} | Resolution: {args.width}x{args.height} | Model: {args.model}")
    print("=" * 60)

    # 1. Initialize Camera Stream
    camera = Camera(device_id=args.camera, width=args.width, height=args.height)
    if not camera.is_opened:
        print("[Error] Failed to initialize camera. Exiting.")
        sys.exit(1)

    # 2. Initialize YOLO Detector
    try:
        detector = YOLODetector(model_path=args.model, conf_threshold=args.conf)
    except Exception as e:
        print(f"[Error] Failed to load detector model: {e}")
        camera.release()
        sys.exit(1)

    window_name = "Weed Rover - Object Detection Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n[Pipeline Ready] Streaming and running inference... Press 'q' or ESC to exit.\n")

    try:
        while True:
            # Step A: Capture OpenCV frame
            success, frame = camera.read()
            if not success or frame is None:
                print("[Warning] Frame capture failed. Exiting loop.")
                break

            # Step B: Run YOLO detection
            detections = detector.detect(frame)

            # Step C: Draw bounding boxes, labels, and target centroids
            annotated_frame = detector.draw_detections(frame, detections)

            # Step D: Overlay FPS
            Camera.draw_fps(annotated_frame, camera.get_fps())

            # Step E: Display output
            cv2.imshow(window_name, annotated_frame)

            # Step F: Handle quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[Pipeline] Exit requested by user.")
                break

    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted by keyboard.")
    finally:
        # Step G: Clean up resources
        camera.release()
        cv2.destroyAllWindows()
        print("[Pipeline] Shutdown complete.")


if __name__ == "__main__":
    main()
