"""
Offline Demonstration and Validation Script for RobotGeometry.

Runs coordinate transformation across representative image pixels,
displaying the calculated ground-plane position in both the rover/camera reference frame
and the LED reference frame.

Run via:
    python vision/test_geometry.py
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.robot_geometry import RobotGeometry, CalibrationPoint
from config.vision_config import (
    CAMERA_HEIGHT_CM,
    CAMERA_TILT_DEG,
    CAMERA_RESOLUTION_WIDTH,
    CAMERA_RESOLUTION_HEIGHT,
    CAMERA_FOV_HORIZONTAL_DEG,
    CAMERA_FOV_VERTICAL_DEG,
    CAMERA_TO_LED_FORWARD_CM,
    CAMERA_TO_LED_LATERAL_CM,
    LED_HEIGHT_CM,
)


def run_geometry_demo():
    print("=" * 70)
    print("WEED ROVER - ROBOT GEOMETRY PROJECTION DEMO")
    print("=" * 70)
    print("Configured Parameters (INITIAL PLACEHOLDERS - calibrate on physical robot):")
    print(f"  Camera Height      : {CAMERA_HEIGHT_CM:.1f} cm")
    print(f"  Camera Tilt Down   : {CAMERA_TILT_DEG:.1f} deg")
    print(f"  Resolution         : {CAMERA_RESOLUTION_WIDTH}x{CAMERA_RESOLUTION_HEIGHT} px")
    print(f"  Field of View (H/V): {CAMERA_FOV_HORIZONTAL_DEG:.1f} deg / {CAMERA_FOV_VERTICAL_DEG:.1f} deg")
    print(f"  Camera-to-LED Offsets: Forward={CAMERA_TO_LED_FORWARD_CM:.1f} cm, Lateral={CAMERA_TO_LED_LATERAL_CM:.1f} cm")
    print(f"  LED Height         : {LED_HEIGHT_CM:.1f} cm")
    print("=" * 70)
    print("Coordinate Convention:")
    print("  +Y = Forward distance in front of rover")
    print("  +X = Lateral distance to the right of rover (-X = to the left)")
    print("=" * 70)

    geom = RobotGeometry(
        camera_height_cm=CAMERA_HEIGHT_CM,
        camera_tilt_deg=CAMERA_TILT_DEG,
        image_width=CAMERA_RESOLUTION_WIDTH,
        image_height=CAMERA_RESOLUTION_HEIGHT,
        fov_horizontal_deg=CAMERA_FOV_HORIZONTAL_DEG,
        fov_vertical_deg=CAMERA_FOV_VERTICAL_DEG,
        camera_to_led_forward_cm=CAMERA_TO_LED_FORWARD_CM,
        camera_to_led_lateral_cm=CAMERA_TO_LED_LATERAL_CM,
        led_height_cm=LED_HEIGHT_CM,
    )

    sample_pixels = [
        ("Image Center (Optical Axis)", 320, 240),
        ("Left of Center (Mid-ground)", 220, 300),
        ("Right of Center (Mid-ground)", 420, 300),
        ("Far Ahead (Upper in Image)", 320, 180),
        ("Near Field (Lower in Image)", 320, 360),
        ("Very Near Field (Near Bottom Edge)", 320, 440),
        ("Far Left Ground", 120, 320),
        ("Far Right Ground", 520, 320),
        ("Near Left Floor", 160, 420),
        ("Near Right Floor", 480, 420),
        ("Near Horizon (Far Distance)", 320, 80),
        ("Top Frame Edge (Distant Ground)", 320, 10),
    ]

    print(f"{'Description':<35} | {'Pixel (u, v)':<14} | {'Camera Ground (X, Y) cm':<24} | {'LED Frame (X, Y) cm':<22}")
    print("-" * 105)

    for desc, u, v in sample_pixels:
        ground_pt = geom.pixel_to_ground(u, v)
        led_pt = geom.pixel_to_led_ground(u, v)

        pixel_str = f"({u}, {v})"
        if ground_pt is not None:
            gx, gy = ground_pt
            lx, ly = led_pt
            sign_x = "+" if gx > 0 else ("-" if gx < 0 else " ")
            sign_lx = "+" if lx > 0 else ("-" if lx < 0 else " ")
            ground_str = f"({sign_x}{abs(gx):5.1f}, {gy:5.1f}) cm"
            led_str = f"({sign_lx}{abs(lx):5.1f}, {ly:5.1f}) cm"
        else:
            ground_str = "None (No ground hit)"
            led_str = "None"

        print(f"{desc:<35} | {pixel_str:<14} | {ground_str:<24} | {led_str:<22}")

    print("=" * 105)

    # Sample Calibration Evaluation demonstration
    print("\nCalibration Interface Demonstration:")
    sample_cal = [
        CalibrationPoint(pixel_x=320, pixel_y=240, known_x_cm=0.0, known_y_cm=28.5),
        CalibrationPoint(pixel_x=220, pixel_y=300, known_x_cm=-6.0, known_y_cm=20.0),
        CalibrationPoint(pixel_x=420, pixel_y=300, known_x_cm=6.0, known_y_cm=20.0),
    ]
    results = geom.evaluate_calibration(sample_cal)
    print(f"  Total validation points: {results['total_points']}")
    print(f"  Mean Absolute Error X  : {results['mae_x']} cm")
    print(f"  Mean Absolute Error Y  : {results['mae_y']} cm")
    print(f"  Mean Euclidean Error   : {results['mean_distance_error']} cm")
    print(f"  Max Error              : {results['max_distance_error']} cm")
    print("=" * 70)


if __name__ == "__main__":
    run_geometry_demo()
