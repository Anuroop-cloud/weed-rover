"""
MODE B — ROBOT EXECUTION
Main Vision & Targeting Pipeline for weed-rover.

Orchestrates the complete robot demonstration loop:
USB Webcam
  -> Black/Color Dot Detector (DualDotDetector)
  -> Stable Target Tracking & ROI Filtering
  -> RobotGeometry (Pixel to 2D Ground Coordinates X_cm, Y_cm)
  -> TargetController (Lock-and-Execute State Machine with Blind-Spot Navigation & LED Fire Signal)

Note: This mode is completely independent of YOLO and does not require models/best.pt.
"""

import argparse
import os
import sys
import time

# Ensure repository root is in sys.path so vision and config are always importable
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2

try:
    from vision.camera import Camera
    from vision.crop_weed_detector import CropWeedDetector
    from vision.control_interface import WeedControlPipeline, LEDMatrixMapper
    from vision.robot_geometry import RobotGeometry
    from vision.continuous_executor import ContinuousWeedExecutor
    from vision.target_controller import TargetController, RoverState
    from config.vision_config import (
        CAMERA_DEVICE_INDEX,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        CAMERA_FPS,
        SHOW_MASK_DEBUG_WINDOW,
        FARM_ROI,
        STABILITY_MIN_HITS,
        MAX_MISSED_FRAMES,
        TARGET_MATCH_DISTANCE,
        CAMERA_HEIGHT_CM,
        CAMERA_TILT_DEG,
        CAMERA_FOV_HORIZONTAL_DEG,
        CAMERA_FOV_VERTICAL_DEG,
        CAMERA_TO_LED_FORWARD_CM,
        CAMERA_TO_LED_LATERAL_CM,
        LED_HEIGHT_CM,
        STATE_MACHINE_TARGET_CLASS,
        ALIGNMENT_TOLERANCE_XY_CM,
        CAMERA_BLIND_SPOT_ROW_PX,
        SIMULATED_APPROACH_SPEED_CM_S,
        LED_FIRE_DURATION_SEC,
        COMPLETED_TARGET_EXPIRY_SEC,
        MARKER_MIN_AREA,
        MARKER_MAX_AREA,
        MARKER_LOWER_HSV,
        MARKER_UPPER_HSV,
    )
except ImportError:
    from camera import Camera
    from crop_weed_detector import CropWeedDetector
    from control_interface import WeedControlPipeline, LEDMatrixMapper
    from robot_geometry import RobotGeometry
    from target_controller import TargetController, RoverState

    CAMERA_DEVICE_INDEX = 1
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    CAMERA_FPS = 30.0
    MARKER_LOWER_HSV = (0, 0, 0)
    MARKER_UPPER_HSV = (180, 255, 85)
    MARKER_MIN_AREA = 25.0
    MARKER_MAX_AREA = 8000.0
    SHOW_MASK_DEBUG_WINDOW = True
    FARM_ROI = (40, 30, 600, 450)
    STABILITY_MIN_HITS = 3
    MAX_MISSED_FRAMES = 3
    TARGET_MATCH_DISTANCE = 30.0
    CAMERA_HEIGHT_CM = 6.0
    CAMERA_TILT_DEG = 0.0
    CAMERA_FOV_HORIZONTAL_DEG = 70.0
    CAMERA_FOV_VERTICAL_DEG = 55.0
    CAMERA_TO_LED_FORWARD_CM = 5.0
    CAMERA_TO_LED_LATERAL_CM = 0.0
    LED_HEIGHT_CM = 10.0
    STATE_MACHINE_TARGET_CLASS = "weed"
    ALIGNMENT_TOLERANCE_XY_CM = 1.5
    CAMERA_BLIND_SPOT_ROW_PX = 430
    SIMULATED_APPROACH_SPEED_CM_S = 8.0
    LED_FIRE_DURATION_SEC = 1.5
    COMPLETED_TARGET_EXPIRY_SEC = 15.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Weed Rover Prototype Vision: Crop Dots (●) & Weed X Markers (X) with LED Matrix Mapping"
    )
    parser.add_argument(
        "--camera",
        type=lambda x: int(x) if x.isdigit() else x,
        default=CAMERA_DEVICE_INDEX,
        help=f"Camera device index (e.g. 0, 1) or phone stream URL (default: {CAMERA_DEVICE_INDEX})",
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
        help=f"Target frame rate limit (default: {CAMERA_FPS})",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=MARKER_MIN_AREA,
        help=f"Minimum marker area in pixels (default: {MARKER_MIN_AREA})",
    )
    parser.add_argument(
        "--max-area",
        type=float,
        default=MARKER_MAX_AREA,
        help=f"Maximum marker area in pixels (default: {MARKER_MAX_AREA})",
    )
    parser.add_argument(
        "--v-max",
        type=int,
        default=MARKER_UPPER_HSV[2],
        help=f"Maximum brightness/Value for dark marker threshold (default: {MARKER_UPPER_HSV[2]})",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable the separate threshold mask debug windows",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print state machine transitions and target coordinates to terminal",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List all detected camera device indices on the system and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Camera scan utility
    if args.list_cameras:
        print("[Camera Scan] Scanning available video devices...")
        cams = Camera.list_available_cameras(max_tested=6)
        if cams:
            print(f"Detected camera index(es): {cams}")
            print("Tip: Index 0 is typically the built-in laptop webcam; Index 1+ is your USB camera.")
        else:
            print("No cameras detected.")
        sys.exit(0)

    print("=" * 65)
    print("MODE B — ROBOT EXECUTION PIPELINE")
    print("=" * 65)
    print(f"Camera Device : {args.camera}")
    print(f"Resolution    : {args.width}x{args.height} @ {args.fps} FPS")
    print(f"Target Class  : {STATE_MACHINE_TARGET_CLASS}")
    print(f"Dot Area Filter: [{args.min_area}, {args.max_area}] px")
    print("-" * 65)

    # 1. Initialize Camera Stream
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
        print(f"[Error] Failed to initialize camera (tested index {args.camera} and fallback 0). Exiting.")
        sys.exit(1)

    # 2. Initialize Crop & Weed Marker Detector
    detector = CropWeedDetector(
        min_area=args.min_area,
        max_area=args.max_area,
        roi=FARM_ROI,
        stability_min_hits=STABILITY_MIN_HITS,
        max_missed_frames=MAX_MISSED_FRAMES,
        match_distance=TARGET_MATCH_DISTANCE,
    )

    # 3. Initialize Robot Geometry (Ground plane projection)
    geometry = RobotGeometry(
        camera_height_cm=CAMERA_HEIGHT_CM,
        camera_tilt_deg=CAMERA_TILT_DEG,
        image_width=args.width,
        image_height=args.height,
        fov_horizontal_deg=CAMERA_FOV_HORIZONTAL_DEG,
        fov_vertical_deg=CAMERA_FOV_VERTICAL_DEG,
        camera_to_led_forward_cm=CAMERA_TO_LED_FORWARD_CM,
        camera_to_led_lateral_cm=CAMERA_TO_LED_LATERAL_CM,
        led_height_cm=LED_HEIGHT_CM,
    )

    # 4. Initialize Continuous-Forward Weed Executor
    # (Replaces old TargetController: no target IDs, no target-locking, no stopping)
    executor = ContinuousWeedExecutor(
        geometry=geometry,
        matrix_mapper=LEDMatrixMapper(),
        firing_boundary_row_px=CAMERA_BLIND_SPOT_ROW_PX,
        rover_forward_speed_cm_s=SIMULATED_APPROACH_SPEED_CM_S,
        camera_to_led_forward_cm=CAMERA_TO_LED_FORWARD_CM,
        fire_duration_sec=LED_FIRE_DURATION_SEC,
    )

    window_name = "Weed Rover - Robot Execution (Mode B)"
    mask_window = "Marker Mask"
    show_mask = not args.no_mask and (SHOW_MASK_DEBUG_WINDOW or args.debug)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if show_mask:
        cv2.namedWindow(mask_window, cv2.WINDOW_NORMAL)

    print(f"\n[Pipeline Ready] Robot continuous-forward execution running. Press 'q' or ESC to exit.\n")

    last_time = time.time()

    try:
        while True:
            # Step A: Capture frame
            success, frame = camera.read()
            if not success or frame is None:
                print("[Warning] Frame capture failed. Exiting loop.")
                break

            now = time.time()
            dt = max(0.001, min(0.2, now - last_time))
            last_time = now

            # Step B: Run Crop & Weed Marker Detection
            detections, mask = detector.detect(frame)

            # Step C: Update Continuous-Forward Weed Executor
            status = executor.update(detections, delta_time=dt, current_time=now)
            payload = status.to_control_payload()

            if args.debug and status.is_firing:
                print(f"[CONTINUOUS EXECUTOR] Firing Columns: {status.firing_columns}")

            # Step D: Draw Clean HUD (No target IDs, no state locking)
            annotated_frame = executor.draw_hud(
                frame=frame,
                detections=detections,
                fps=camera.get_fps(),
                status=status,
            )

            # Step E: Display debug mask if enabled
            if show_mask:
                cv2.imshow(mask_window, mask)

            # Step F: Display main video stream
            cv2.imshow(window_name, annotated_frame)

            # Step G: Handle quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("[Pipeline] Exit requested by user.")
                break


    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted by keyboard.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[Pipeline] Shutdown complete.")


if __name__ == "__main__":
    main()
