"""
MODE B: LAPTOP -> ESP32 ROBOT INTEGRATION RUNNER
weed-rover continuous-forward execution with live USB serial control.

Pipeline Architecture:
USB Webcam
    ↓
CropWeedDetector (● Crops / X Weeds)
    ↓
Ignore Crops for Control
    ↓
ContinuousWeedExecutor (Spatial Debounce & Column Mapping 0..7)
    ↓
Firing Boundary Crossing & Forward Travel Delay
    ↓
ESP32 Serial Interface (115200 baud, newline-terminated):
    - Startup:  01 (LED OFF / MOTOR ON) -> Rover moves forward
    - Weed:     11 (LED ON  / MOTOR ON) -> Firing while driving
    - Resume:   01 (LED OFF / MOTOR ON) -> Forward motion continues
    - Shutdown: 00 (LED OFF / MOTOR OFF)-> Safe stop
"""

import argparse
import os
import sys
import time
import cv2

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config.vision_config import (
    CAMERA_DEVICE_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CAMERA_FPS,
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
    CAMERA_BLIND_SPOT_ROW_PX,
    SIMULATED_APPROACH_SPEED_CM_S,
    LED_FIRE_DURATION_SEC,
    MARKER_MIN_AREA,
    MARKER_MAX_AREA,
)
from vision.camera import Camera
from vision.crop_weed_detector import CropWeedDetector
from vision.robot_geometry import RobotGeometry
from vision.control_interface import LEDMatrixMapper
from vision.continuous_executor import ContinuousWeedExecutor
from vision.esp32_interface import ESP32Interface


def parse_args():
    parser = argparse.ArgumentParser(
        description="Weed Rover: Laptop to ESP32 Continuous-Forward Vision & Serial Integration"
    )
    parser.add_argument(
        "--camera",
        type=lambda x: int(x) if x.isdigit() else x,
        default=CAMERA_DEVICE_INDEX,
        help=f"Camera index (0, 1) or stream URL (default: {CAMERA_DEVICE_INDEX})",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port for ESP32 (e.g. COM3, COM5, /dev/ttyUSB0). If omitted, searches available ports.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Serial baud rate (default: 115200)",
    )
    parser.add_argument(
        "--mock-serial",
        action="store_true",
        help="Run with mock serial interface (for testing without physical ESP32 attached)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=FRAME_WIDTH,
        help=f"Capture frame width (default: {FRAME_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=FRAME_HEIGHT,
        help=f"Capture frame height (default: {FRAME_HEIGHT})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=CAMERA_FPS,
        help=f"Target frame rate limit (default: {CAMERA_FPS})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Configurable travel delay from firing boundary to LED matrix in seconds (default: 0.625s)",
    )
    parser.add_argument(
        "--fire-duration",
        type=float,
        default=LED_FIRE_DURATION_SEC,
        help=f"LED activation pulse duration in seconds (default: {LED_FIRE_DURATION_SEC}s)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose diagnostic terminal logging",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable binary threshold mask debug window",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 65)
    print("WEED ROVER: LAPTOP -> ESP32 VISION INTEGRATION")
    print("Continuous-Forward Physical Prototype Runner")
    print("=" * 65)

    # 1. Initialize Camera Stream
    print(f"[Camera] Opening camera source: {args.camera}...")
    camera = Camera(
        device_id=args.camera,
        width=args.width,
        height=args.height,
        fps_limit=args.fps,
    )
    if not camera.is_opened and args.camera != 0:
        print(f"[Camera Fallback] Camera {args.camera} failed. Falling back to built-in camera index 0...")
        camera = Camera(
            device_id=0,
            width=args.width,
            height=args.height,
            fps_limit=args.fps,
        )

    if not camera.is_opened:
        print("[Error] Camera capture could not be opened. Exiting.")
        sys.exit(1)

    # 2. Initialize Crop & Weed Marker Detector
    detector = CropWeedDetector(
        min_area=MARKER_MIN_AREA,
        max_area=MARKER_MAX_AREA,
        roi=FARM_ROI,
        stability_min_hits=STABILITY_MIN_HITS,
        max_missed_frames=MAX_MISSED_FRAMES,
        match_distance=TARGET_MATCH_DISTANCE,
    )

    # 3. Initialize Top-Down Robot Geometry & Matrix Column Mapper
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
    matrix_mapper = LEDMatrixMapper()

    # 4. Initialize Continuous-Forward Weed Executor
    executor = ContinuousWeedExecutor(
        geometry=geometry,
        matrix_mapper=matrix_mapper,
        firing_boundary_row_px=CAMERA_BLIND_SPOT_ROW_PX,
        rover_forward_speed_cm_s=SIMULATED_APPROACH_SPEED_CM_S,
        camera_to_led_forward_cm=CAMERA_TO_LED_FORWARD_CM,
        firing_delay_sec=args.delay,
        fire_duration_sec=args.fire_duration,
    )

    # 5. Initialize ESP32 Serial Interface
    port = args.port
    if not port and not args.mock_serial:
        avail = ESP32Interface.list_available_ports()
        if avail:
            port = avail[0]
            print(f"[ESP32 Auto-Detect] Found serial port: {port}")
        else:
            print("[Warning] No physical serial ports found. Falling back to mock serial mode (--mock-serial).")
            args.mock_serial = True

    esp32 = ESP32Interface(
        port=port,
        baud=args.baud,
        mock_mode=args.mock_serial,
    )

    print(f"[ESP32] Connecting to ESP32 (port={port or 'MOCK'}, baud={args.baud})...")
    connected = esp32.connect()
    if not connected:
        print("[Warning] Could not open physical ESP32 port. Switching to mock serial mode for visual demo.")
        esp32.mock_mode = True
        esp32.connect()

    print("\n" + "-" * 65)
    print("CONTROLS:")
    print("  'q' or ESC : Safe Shutdown (sends 00 to stop motor/LED and exits)")
    print("-" * 65 + "\n")

    window_name = "Weed Rover - Laptop to ESP32 Execution"
    mask_window = "Marker Mask"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if not args.no_mask:
        cv2.namedWindow(mask_window, cv2.WINDOW_NORMAL)

    last_time = time.time()

    try:
        while True:
            # Step A: Capture camera frame
            success, frame = camera.read()
            if not success or frame is None:
                print("[Warning] Camera frame read failure. Exiting loop.")
                break

            now = time.time()
            dt = max(0.001, min(0.2, now - last_time))
            last_time = now

            # Step B: Detect ● Crops and X Weeds
            detections, mask = detector.detect(frame)

            # Step C: Continuous-Forward Tracking & Delay Scheduling
            status = executor.update(detections, delta_time=dt, current_time=now)
            payload = status.to_control_payload()

            # Step D: Serial Command Transmission to ESP32
            # 11 = LED ON / MOTOR ON (firing while moving forward)
            # 01 = LED OFF / MOTOR ON (driving forward)
            if status.is_firing:
                active_col = status.column if status.column >= 0 else 3
                esp32.send_command(ESP32Interface.CMD_FIRE_WHILE_DRIVING, column_info=active_col)
            else:
                esp32.send_command(ESP32Interface.CMD_DRIVE_FORWARD)

            # Step E: Render Clean Live HUD
            annotated_frame = executor.draw_hud(
                frame=frame,
                detections=detections,
                fps=camera.get_fps(),
                status=status,
                serial_status=esp32.status_str,
                last_command=esp32.last_cmd_str,
            )

            # Step F: Display Stream
            if not args.no_mask:
                cv2.imshow(mask_window, mask)
            cv2.imshow(window_name, annotated_frame)

            # Step G: Handle Exit Key
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("[Shutdown] Exit requested by user.")
                break

    except KeyboardInterrupt:
        print("\n[Shutdown] Interrupted by keyboard (Ctrl+C).")

    finally:
        # Step H: Safe Shutdown Requirement
        # Immediately sends 00\n to turn off motor and LED before closing connection
        print("[Shutdown] Triggering safe robot shutdown...")
        esp32.safe_shutdown()
        camera.release()
        cv2.destroyAllWindows()
        print("[Shutdown] Robot halted safely. Done.")


if __name__ == "__main__":
    main()
