"""
Unit tests for RobotGeometry ground-plane projection and calibration interface.
Tests coordinate conventions, symmetry, monotonicity, invalid rays, and LED frame offsets.
"""

import math
import os
import sys
import pytest

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
)


@pytest.fixture
def default_geometry() -> RobotGeometry:
    """Fixture initializing RobotGeometry with configured placeholder values."""
    return RobotGeometry(
        camera_height_cm=CAMERA_HEIGHT_CM,
        camera_tilt_deg=CAMERA_TILT_DEG,
        image_width=CAMERA_RESOLUTION_WIDTH,
        image_height=CAMERA_RESOLUTION_HEIGHT,
        fov_horizontal_deg=CAMERA_FOV_HORIZONTAL_DEG,
        fov_vertical_deg=CAMERA_FOV_VERTICAL_DEG,
        camera_to_led_forward_cm=CAMERA_TO_LED_FORWARD_CM,
        camera_to_led_lateral_cm=CAMERA_TO_LED_LATERAL_CM,
    )


def test_center_pixel_x_is_approximately_zero(default_geometry):
    """
    A pixel along the vertical center line (x = W / 2) should produce lateral X approx 0.
    """
    cx = default_geometry.image_width / 2.0
    cy = default_geometry.image_height / 2.0

    res = default_geometry.pixel_to_ground(cx, cy)
    assert res is not None
    x_cm, y_cm = res

    # Lateral distance along optical center column must be zero
    assert abs(x_cm) < 1e-2
    # Forward distance must be positive
    assert y_cm > 0


def test_left_right_symmetry(default_geometry):
    """
    Symmetric pixels to the left and right of image center at the same row
    should yield opposite X signs and identical forward Y distances.
    """
    cx = default_geometry.image_width / 2.0
    test_y = default_geometry.image_height / 2.0 + 50  # Lower half of image (firmly on ground)
    offset_x = 80.0

    left_pixel = (cx - offset_x, test_y)
    right_pixel = (cx + offset_x, test_y)

    res_left = default_geometry.pixel_to_ground(*left_pixel)
    res_right = default_geometry.pixel_to_ground(*right_pixel)

    assert res_left is not None
    assert res_right is not None

    x_left, y_left = res_left
    x_right, y_right = res_right

    # Left is negative X, Right is positive X
    assert x_left < 0, f"Expected X < 0 for left pixel, got {x_left}"
    assert x_right > 0, f"Expected X > 0 for right pixel, got {x_right}"

    # Magnitudes should be equal within precision
    assert abs(x_left + x_right) <= 0.05
    assert abs(y_left - y_right) <= 0.05


def test_forward_direction_is_positive(default_geometry):
    """
    All valid ground-plane pixels within the ground viewing region must produce positive Y forward distances.
    """
    cx = default_geometry.image_width / 2.0

    for py in [240, 300, 360, 420, 470]:
        res = default_geometry.pixel_to_ground(cx, py)
        assert res is not None
        x_cm, y_cm = res
        assert y_cm > 0, f"Ground distance at row {py} must be positive, got {y_cm}"


def test_monotonic_forward_behavior(default_geometry):
    """
    Points nearer to the robot appear lower in the image (larger pixel_y).
    Therefore, increasing pixel_y along the center column must yield strictly decreasing forward distance Y.
    """
    cx = default_geometry.image_width / 2.0

    # Test several rows from center (240) down to bottom edge (470)
    rows = [240, 280, 320, 360, 400, 450]
    y_distances = []

    for py in rows:
        res = default_geometry.pixel_to_ground(cx, py)
        assert res is not None
        y_distances.append(res[1])

    # Each distance should be strictly smaller than the previous one
    for i in range(len(y_distances) - 1):
        assert y_distances[i] > y_distances[i + 1], (
            f"Forward distance must decrease as pixel moves down (nearer). "
            f"Row {rows[i]} -> {y_distances[i]}cm vs Row {rows[i+1]} -> {y_distances[i+1]}cm"
        )


def test_invalid_and_sky_rays_handled_safely(default_geometry):
    """
    Pixels whose rays do not intersect the usable ground (e.g. above horizon, or NaN/Inf)
    should be handled safely and return None without throwing unhandled exceptions.
    """
    # 1. NaN and Inf inputs
    assert default_geometry.pixel_to_ground(float("nan"), 300.0) is None
    assert default_geometry.pixel_to_ground(320.0, float("inf")) is None

    # 2. Pixel way above horizon (e.g., pointing up at the sky if tilt was shallow)
    shallow_geom = RobotGeometry(camera_height_cm=20.0, camera_tilt_deg=10.0)
    # Row 0 with 10 deg tilt points well above horizontal
    res_sky = shallow_geom.pixel_to_ground(320.0, 0.0)
    assert res_sky is None

    # 3. Beyond max distance
    geom_strict = RobotGeometry(camera_height_cm=20.0, camera_tilt_deg=35.0, max_ground_distance_cm=25.0)
    # Center pixel (around 28cm) exceeds 25cm limit
    assert geom_strict.pixel_to_ground(320.0, 240.0) is None


def test_ground_point_from_pixel_alias(default_geometry):
    """Verify ground_point_from_pixel produces identical results to pixel_to_ground."""
    pt1 = default_geometry.pixel_to_ground(200.0, 350.0)
    pt2 = default_geometry.ground_point_from_pixel(200.0, 350.0)
    assert pt1 == pt2


def test_led_frame_transformation(default_geometry):
    """
    Verify transformation from camera frame to LED frame applies camera_to_led offsets correctly.
    """
    # CAMERA_TO_LED_FORWARD_CM = 5.0, CAMERA_TO_LED_LATERAL_CM = 0.0
    x_robot, y_robot = 4.0, 25.0
    x_led, y_led = default_geometry.ground_to_led_frame(x_robot, y_robot)

    assert x_led == x_robot - default_geometry.camera_to_led_lateral_cm
    assert y_led == y_robot - default_geometry.camera_to_led_forward_cm
    assert x_led == 4.0
    assert y_led == 20.0

    # Also verify pixel_to_led_ground convenience method
    res_robot = default_geometry.pixel_to_ground(320.0, 300.0)
    res_led = default_geometry.pixel_to_led_ground(320.0, 300.0)
    assert res_robot is not None
    assert res_led is not None
    assert res_led[0] == res_robot[0] - default_geometry.camera_to_led_lateral_cm
    assert res_led[1] == res_robot[1] - default_geometry.camera_to_led_forward_cm


def test_calibration_interface(default_geometry):
    """
    Verify CalibrationPoint interface and evaluate_calibration error calculation.
    """
    # Generate synthetic calibration points from known model output
    pt1 = default_geometry.pixel_to_ground(320.0, 280.0)
    pt2 = default_geometry.pixel_to_ground(250.0, 320.0)
    assert pt1 is not None and pt2 is not None

    cal_points = [
        CalibrationPoint(pixel_x=320.0, pixel_y=280.0, known_x_cm=pt1[0], known_y_cm=pt1[1]),
        CalibrationPoint(pixel_x=250.0, pixel_y=320.0, known_x_cm=pt2[0], known_y_cm=pt2[1]),
    ]

    metrics = default_geometry.evaluate_calibration(cal_points)
    assert metrics["total_points"] == 2
    assert metrics["valid_points"] == 2
    assert metrics["mae_x"] == 0.0
    assert metrics["mae_y"] == 0.0
    assert metrics["rmse_x"] == 0.0
    assert metrics["max_distance_error"] == 0.0

    # Introduce a 1 cm known offset and verify metrics reflect the delta
    perturbed_points = [
        CalibrationPoint(pixel_x=320.0, pixel_y=280.0, known_x_cm=pt1[0] + 1.0, known_y_cm=pt1[1]),
    ]
    p_metrics = default_geometry.evaluate_calibration(perturbed_points)
    assert p_metrics["valid_points"] == 1
    assert p_metrics["mae_x"] == 1.0
    assert p_metrics["mean_distance_error"] == 1.0
