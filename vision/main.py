"""
Main Vision Pipeline for weed-rover.
Orchestrates: Camera Capture -> YOLO Inference -> Visual Annotations -> Real-time Display.
"""

import argparse
import sys
import cv2

try:
    from vision.camera import Camera
    from vision.detector import YOLODetector
    from vision.target_detector import ColorTargetDetector
    from vision.dual_dot_detector import DualDotDetector
    from config.vision_config import (
        CAMERA_DEVICE_INDEX,
        BLACK_DOT_LOWER_HSV,
        BLACK_DOT_UPPER_HSV,
        BLACK_DOT_MIN_AREA,
        BLACK_DOT_MAX_AREA,
        BLUE_LOWER,
        BLUE_UPPER,
        SHOW_MASK_DEBUG_WINDOW,
        FARM_ROI,
        STABILITY_MIN_HITS,
        MAX_MISSED_FRAMES,
        TARGET_MATCH_DISTANCE,
    )
except ImportError:
    from camera import Camera
    from detector import YOLODetector
    from target_detector import ColorTargetDetector
    from dual_dot_detector import DualDotDetector
    CAMERA_DEVICE_INDEX = 1
    BLACK_DOT_LOWER_HSV = (0, 0, 0)
    BLACK_DOT_UPPER_HSV = (180, 255, 75)
    BLACK_DOT_MIN_AREA = 25.0
    BLACK_DOT_MAX_AREA = 4000.0
    BLUE_LOWER = (100, 50, 50)
    BLUE_UPPER = (135, 255, 255)
    SHOW_MASK_DEBUG_WINDOW = True
    FARM_ROI = (40, 30, 600, 450)
    STABILITY_MIN_HITS = 3
    MAX_MISSED_FRAMES = 3
    TARGET_MATCH_DISTANCE = 30.0


def parse_args():
    parser = argparse.ArgumentParser(description="Weed Rover Vision Pipeline")
    parser.add_argument(
        "--detector",
        choices=["dot", "target", "yolo"],
        default="dot",
        help="Active detector: 'dot' (black dots on grid), 'target' (color marker), or 'yolo'. Default: 'dot'"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_DEVICE_INDEX,
        help=f"Camera device index (default: {CAMERA_DEVICE_INDEX})"
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List all detected camera device indices on the system and exit"
    )
    parser.add_argument("--width", type=int, default=640, help="Camera capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height (default: 480)")
    parser.add_argument("--min-area", type=float, default=BLACK_DOT_MIN_AREA, help=f"Minimum dot area in pixels (default: {BLACK_DOT_MIN_AREA})")
    parser.add_argument("--max-area", type=float, default=BLACK_DOT_MAX_AREA, help=f"Maximum dot area in pixels (default: {BLACK_DOT_MAX_AREA})")
    parser.add_argument("--v-max", type=int, default=BLACK_DOT_UPPER_HSV[2], help=f"Maximum brightness/Value for black threshold (default: {BLACK_DOT_UPPER_HSV[2]})")
    parser.add_argument("--no-mask", action="store_true", help="Disable the separate threshold mask debug window")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model weights (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--debug", action="store_true", help="Print detection coordinates to terminal")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_cameras:
        print("[Camera Scan] Scanning available video devices...")
        cams = Camera.list_available_cameras(max_tested=6)
        if cams:
            print(f"Detected camera index(es): {cams}")
            print("Tip: Index 0 is typically the built-in laptop webcam; Index 1+ is your USB camera.")
        else:
            print("No cameras detected.")
        sys.exit(0)
    
    print("=" * 60)
    print("Starting Weed Rover Vision Pipeline")
    print(f"Mode: {args.detector.upper()} | Device: {args.camera} | Resolution: {args.width}x{args.height}")
    print("=" * 60)

    # 1. Initialize Camera Stream
    camera = Camera(device_id=args.camera, width=args.width, height=args.height)
    if not camera.is_opened:
        print("[Error] Failed to initialize camera. Exiting.")
        sys.exit(1)

    # 2. Initialize Active Detector
    black_mask_window = "Black Mask"
    blue_mask_window = "Blue Mask"
    show_mask = not args.no_mask and (SHOW_MASK_DEBUG_WINDOW or args.debug)

    if args.detector == "dot":
        lower_hsv = (0, 0, 0)
        upper_hsv = (180, 255, args.v_max)
        detector = DualDotDetector(
            black_lower_hsv=lower_hsv,
            black_upper_hsv=upper_hsv,
            blue_lower_hsv=BLUE_LOWER,
            blue_upper_hsv=BLUE_UPPER,
            min_area=args.min_area,
            max_area=args.max_area,
            roi=FARM_ROI,
            stability_min_hits=STABILITY_MIN_HITS,
            max_missed_frames=MAX_MISSED_FRAMES,
            match_distance=TARGET_MATCH_DISTANCE,
        )
        window_name = "Main Camera"
        if show_mask:
            cv2.namedWindow(black_mask_window, cv2.WINDOW_NORMAL)
            cv2.namedWindow(blue_mask_window, cv2.WINDOW_NORMAL)
    elif args.detector == "target":
        detector = ColorTargetDetector(min_area=args.min_area)
        window_name = "Weed Rover - Color Target Detector"
    else:
        try:
            detector = YOLODetector(model_path=args.model, conf_threshold=args.conf)
        except Exception as e:
            print(f"[Error] Failed to load detector model: {e}")
            camera.release()
            sys.exit(1)
        window_name = "Weed Rover - Object Detection Feed"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print(f"\n[Pipeline Ready] Active Mode: {args.detector.upper()}. Streaming... Press 'q' or ESC to exit.\n")

    try:
        while True:
            # Step A: Capture OpenCV frame
            success, frame = camera.read()
            if not success or frame is None:
                print("[Warning] Frame capture failed. Exiting loop.")
                break

            if args.detector == "dot":
                # Step B1: Run Dual Dot Detection (Black + Blue) with Temporal Stability & ROI
                targets, mask_black, mask_blue = detector.detect(frame)

                # Step C1: Debug terminal logging for stable targets
                if args.debug:
                    for target in targets:
                        if target.is_stable:
                            print(target.to_debug_string())

                # Step D1: Draw ROI boundary, Red/White boxes for Black dots, Blue boxes for Blue dots
                annotated_frame = detector.draw_detections(frame, targets)
                black_count = sum(1 for t in targets if t.is_stable and t.class_name == "black_dot")
                blue_count = sum(1 for t in targets if t.is_stable and t.class_name == "blue_dot")
                count = f"Black: {black_count} | Blue: {blue_count}"

                # Display two separate threshold mask windows
                if show_mask:
                    cv2.imshow(black_mask_window, mask_black)
                    cv2.imshow(blue_mask_window, mask_blue)

            elif args.detector == "target":
                # Step B2: Run Color Target Detection
                target = detector.detect(frame)

                if args.debug and target.detected:
                    print(target.to_debug_string())

                annotated_frame = detector.draw_target(frame, target)
                count = 1 if target.detected else 0

            else:
                # Step B3: Run YOLO detection
                detections = detector.detect(frame)

                if args.debug:
                    for det in detections:
                        print(det.to_debug_string())

                annotated_frame = detector.draw_detections(frame, detections)
                count = len(detections)

            # Step E: Overlay FPS and detection count on OpenCV window
            Camera.draw_fps(annotated_frame, camera.get_fps(), detection_count=count)

            # Step F: Display output
            cv2.imshow(window_name, annotated_frame)

            # Step G: Handle quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[Pipeline] Exit requested by user.")
                break

    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted by keyboard.")
    finally:
        # Step H: Clean up resources
        camera.release()
        cv2.destroyAllWindows()
        print("[Pipeline] Shutdown complete.")


if __name__ == "__main__":
    main()
