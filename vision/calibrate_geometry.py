"""
Interactive Camera Tilt & Geometry Calibration Utility.

Helps you find your camera's real physical tilt angle and verify ground measurements:
1. Measures the physical camera height with a ruler.
2. Places a single black or blue dot on the floor at a known measured forward distance.
3. Detects the dot's pixel position and solves for your exact mounting tilt angle.

Run via:
    python vision/calibrate_geometry.py --camera 1
"""

import argparse
import math
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
from vision.camera import Camera
from vision.dual_dot_detector import DualDotDetector
from vision.robot_geometry import RobotGeometry
from config.vision_config import (
    CAMERA_DEVICE_INDEX,
    CAMERA_HEIGHT_CM,
    CAMERA_TILT_DEG,
    CAMERA_FOV_HORIZONTAL_DEG,
    CAMERA_FOV_VERTICAL_DEG,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)


def main():
    parser = argparse.ArgumentParser(description="Camera Geometry Calibration Helper")
    parser.add_argument("--camera", type=int, default=CAMERA_DEVICE_INDEX, help="Camera index (default: 1)")
    parser.add_argument("--height", type=float, default=CAMERA_HEIGHT_CM, help="Measured height of camera lens center above floor in cm")
    parser.add_argument("--known-y", type=float, default=25.0, help="Known forward distance of calibration dot placed on floor in cm (default: 25.0)")
    args = parser.parse_args()

    print("=" * 70)
    print("WEED ROVER - CAMERA GEOMETRY CALIBRATION")
    print("=" * 70)
    print(f"1. Measure with a ruler from floor to center of camera lens: {args.height:.1f} cm")
    print(f"2. Place a black dot on the floor directly ahead of the camera at: {args.known_y:.1f} cm")
    print("3. Point the camera at the floor so the dot is clearly visible.")
    print("=" * 70)
    print("Press 'c' to capture & calibrate, or 'q' to exit.\n")

    camera = Camera(device_id=args.camera, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps_limit=30)
    if not camera.is_opened:
        print("[Error] Could not open camera.")
        sys.exit(1)

    detector = DualDotDetector()
    cy = FRAME_HEIGHT / 2.0
    half_fov_v = math.radians(CAMERA_FOV_VERTICAL_DEG / 2.0)
    fy = cy / math.tan(half_fov_v)

    window_name = "Geometry Calibration (Press 'c' to Calibrate, 'q' to Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                break

            targets, _, _ = detector.detect(frame)
            annotated = detector.draw_detections(frame, targets)

            # Draw center alignment guide
            cv2.line(annotated, (int(FRAME_WIDTH / 2), 0), (int(FRAME_WIDTH / 2), FRAME_HEIGHT), (0, 255, 255), 1)

            # If targets detected, show estimated values with current placeholder tilt
            geom_current = RobotGeometry(
                camera_height_cm=args.height,
                camera_tilt_deg=CAMERA_TILT_DEG,
                image_width=FRAME_WIDTH,
                image_height=FRAME_HEIGHT,
                fov_horizontal_deg=CAMERA_FOV_HORIZONTAL_DEG,
                fov_vertical_deg=CAMERA_FOV_VERTICAL_DEG,
            )

            status_text = "Place dot centered at known distance, press 'c' to calibrate"
            if targets:
                t = targets[0]
                pred = geom_current.pixel_to_ground(t.center_x, t.center_y)
                if pred:
                    status_text = f"Dot at pixel ({t.center_x},{t.center_y}) -> Model says Y={pred[1]:.1f}cm (Target: {args.known_y:.1f}cm)"

            cv2.putText(annotated, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                if not targets:
                    print("[Calibration] No dot detected in view! Make sure the dot is visible.")
                    continue

                best_target = min(targets, key=lambda t: abs(t.center_x - FRAME_WIDTH / 2))
                u = best_target.center_x
                v = best_target.center_y

                # Calculate true tilt angle
                # alpha = total downward angle of the ray = arctan(height / known_y)
                alpha_rad = math.atan2(args.height, args.known_y)
                # phi = angle of pixel relative to camera optical axis
                y_c = (v - cy) / fy
                phi_rad = math.atan(y_c)
                # tilt = alpha - phi
                computed_tilt_rad = alpha_rad - phi_rad
                computed_tilt_deg = math.degrees(computed_tilt_rad)

                print("\n" + "=" * 60)
                print("CALIBRATION RESULT")
                print("=" * 60)
                print(f"Detected dot pixel    : ({u}, {v})")
                print(f"Measured camera height: {args.height:.1f} cm")
                print(f"Known floor distance  : {args.known_y:.1f} cm")
                print(f"Current config tilt   : {CAMERA_TILT_DEG:.1f} deg")
                print(f"--> RECOMMENDED TILT  : {computed_tilt_deg:.1f} deg")
                print("-" * 60)
                print(f"Update config/vision_config.py with:")
                print(f"CAMERA_HEIGHT_CM = {args.height:.1f}")
                print(f"CAMERA_TILT_DEG = {computed_tilt_deg:.1f}")
                print("=" * 60 + "\n")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
