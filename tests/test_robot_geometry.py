"""
Unit tests for RobotGeometry top-down floor projection model.
Tests all required coordinate conventions, directional signs, height scaling,
LED frame transformations, center symmetry, and confirms tilt math is eliminated.
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
    """Fixture initializing RobotGeometry with configured values."""
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


def test_image_center_maps_to_origin(default_geometry):
    """
    1. Verify image center (cx, cy) corresponds to approximately (0, 0)
    in camera ground coordinates directly below the camera optical center.
    """
    cx = default_geometry.cx
    cy = default_geometry.cy

    res = default_geometry.pixel_to_ground(cx, cy)
    assert res is not None
    x_cm, y_cm = res

    assert abs(x_cm) < 1e-3, f"Expected X approx 0, got {x_cm}"
    assert abs(y_cm) < 1e-3, f"Expected Y approx 0, got {y_cm}"


def test_pixel_right_of_center_is_positive_x(default_geometry):
    """
    2. Verify a pixel right of center (u > cx) produces positive X ground coordinate (+X = right).
    """
    cx = default_geometry.cx
    cy = default_geometry.cy

    res = default_geometry.pixel_to_ground(cx + 60.0, cy)
    assert res is not None
    x_cm, y_cm = res

    assert x_cm > 0.0, f"Expected positive X for pixel to right of center, got {x_cm}"
    assert abs(y_cm) < 1e-3, f"Expected Y approx 0 on center row, got {y_cm}"


def test_pixel_left_of_center_is_negative_x(default_geometry):
    """
    3. Verify a pixel left of center (u < cx) produces negative X ground coordinate (-X = left).
    """
    cx = default_geometry.cx
    cy = default_geometry.cy

    res = default_geometry.pixel_to_ground(cx - 60.0, cy)
    assert res is not None
    x_cm, y_cm = res

    assert x_cm < 0.0, f"Expected negative X for pixel to left of center, got {x_cm}"
    assert abs(y_cm) < 1e-3, f"Expected Y approx 0 on center row, got {y_cm}"


def test_pixel_above_center_is_positive_y(default_geometry):
    """
    4. Verify a pixel above center (v < cy, image TOP) produces positive Y ground coordinate (+Y = forward).
    """
    cx = default_geometry.cx
    cy = default_geometry.cy

    res = default_geometry.pixel_to_ground(cx, cy - 80.0)
    assert res is not None
    x_cm, y_cm = res

    assert y_cm > 0.0, f"Expected positive Y for pixel above center (image top), got {y_cm}"
    assert abs(x_cm) < 1e-3, f"Expected X approx 0 on center column, got {x_cm}"


def test_pixel_below_center_is_negative_y(default_geometry):
    """
    5. Verify a pixel below center (v > cy, image BOTTOM) produces negative Y ground coordinate (-Y = backward).
    """
    cx = default_geometry.cx
    cy = default_geometry.cy

    res = default_geometry.pixel_to_ground(cx, cy + 80.0)
    assert res is not None
    x_cm, y_cm = res

    assert y_cm < 0.0, f"Expected negative Y for pixel below center (image bottom), got {y_cm}"
    assert abs(x_cm) < 1e-3, f"Expected X approx 0 on center column, got {x_cm}"


def test_increasing_camera_height_scales_xy_proportionally():
    """
    6. Verify increasing camera height scales X and Y proportionally.
       Doubling H should exactly double X and Y for any non-center pixel.
    """
    geom_low = RobotGeometry(camera_height_cm=20.0)
    geom_high = RobotGeometry(camera_height_cm=40.0)  # 2x height

    test_u, test_v = 380.0, 160.0
    res_low = geom_low.pixel_to_ground(test_u, test_v)
    res_high = geom_high.pixel_to_ground(test_u, test_v)

    assert res_low is not None and res_high is not None
    x_low, y_low = res_low
    x_high, y_high = res_high

    assert math.isclose(x_high, 2.0 * x_low, rel_tol=1e-2), f"{x_high} vs 2 * {x_low}"
    assert math.isclose(y_high, 2.0 * y_low, rel_tol=1e-2), f"{y_high} vs 2 * {y_low}"


def test_camera_to_led_offset_sign_convention(default_geometry):
    """
    7. Verify camera-to-LED offset is applied with the documented sign convention:
       X_led = X_camera - lateral_offset
       Y_led = Y_camera - forward_offset

       Example from spec:
       If LED is physically 5 cm forward of camera (forward_offset = 5.0),
       a point directly below camera (0, 0) appears 5 cm behind the LED (Y_led = -5.0).
    """
    # Test ground_to_led_frame on the optical center
    x_led, y_led = default_geometry.ground_to_led_frame(0.0, 0.0)
    assert x_led == -default_geometry.camera_to_led_lateral_cm
    assert y_led == -default_geometry.camera_to_led_forward_cm
    assert y_led == -5.0, "Point under camera must be -5 cm relative to LED mounted +5 cm ahead"

    # Test with arbitrary non-zero offsets
    custom_geom = RobotGeometry(
        camera_height_cm=20.0,
        camera_to_led_forward_cm=8.0,
        camera_to_led_lateral_cm=3.0,
    )
    x_l, y_l = custom_geom.ground_to_led_frame(10.0, 15.0)
    assert x_l == 10.0 - 3.0  # 7.0
    assert y_l == 15.0 - 8.0  # 7.0

    # Test pixel_to_led_ground convenience wrapper
    cx, cy = custom_geom.cx, custom_geom.cy
    led_ground = custom_geom.pixel_to_led_ground(cx, cy)
    assert led_ground is not None
    assert led_ground[0] == -3.0
    assert led_ground[1] == -8.0


def test_symmetry_around_image_center(default_geometry):
    """
    8. Verify symmetry around the image center in both axes.
       Symmetric pixel offsets (du, dv) around (cx, cy) must yield coordinates with equal
       magnitudes and appropriate signs across all four quadrants.
    """
    cx = default_geometry.cx
    cy = default_geometry.cy
    du = 75.0
    dv = 55.0

    top_right = default_geometry.pixel_to_ground(cx + du, cy - dv)
    top_left = default_geometry.pixel_to_ground(cx - du, cy - dv)
    bottom_right = default_geometry.pixel_to_ground(cx + du, cy + dv)
    bottom_left = default_geometry.pixel_to_ground(cx - du, cy + dv)

    assert all(pt is not None for pt in [top_right, top_left, bottom_right, bottom_left])

    # X symmetry: Right is +X, Left is -X with equal magnitudes
    assert top_right[0] == -top_left[0]
    assert bottom_right[0] == -bottom_left[0]
    assert abs(top_right[0]) == abs(bottom_right[0])

    # Y symmetry: Top is +Y, Bottom is -Y with equal magnitudes
    assert top_right[1] == -bottom_right[1]
    assert top_left[1] == -bottom_left[1]
    assert abs(top_right[1]) == abs(top_left[1])


def test_no_angled_camera_tilt_math_used():
    """
    9. Verify no angled-camera tilt math is being used:
       Changing camera_tilt_deg parameter has zero effect on ground projection.
    """
    geom_tilt_0 = RobotGeometry(camera_tilt_deg=0.0)
    geom_tilt_35 = RobotGeometry(camera_tilt_deg=35.0)
    geom_tilt_75 = RobotGeometry(camera_tilt_deg=75.0)
    geom_tilt_none = RobotGeometry(camera_tilt_deg=None)

    test_pixels = [
        (320.0, 240.0),
        (200.0, 150.0),
        (450.0, 380.0),
        (100.0, 420.0),
        (540.0, 80.0),
    ]

    for u, v in test_pixels:
        pt_0 = geom_tilt_0.pixel_to_ground(u, v)
        pt_35 = geom_tilt_35.pixel_to_ground(u, v)
        pt_75 = geom_tilt_75.pixel_to_ground(u, v)
        pt_none = geom_tilt_none.pixel_to_ground(u, v)

        assert pt_0 == pt_35, f"Tilt changed projection at ({u}, {v}): {pt_0} vs {pt_35}"
        assert pt_0 == pt_75, f"Tilt changed projection at ({u}, {v}): {pt_0} vs {pt_75}"
        assert pt_0 == pt_none, f"Tilt None changed projection at ({u}, {v}): {pt_0} vs {pt_none}"

    # Confirm tilt trig terms do not exist as active attributes
    assert not hasattr(geom_tilt_0, "sin_tilt")
    assert not hasattr(geom_tilt_0, "cos_tilt")


def test_invalid_and_out_of_bounds_inputs_handled_safely(default_geometry):
    """
    Verify invalid numeric inputs (NaN, Inf) and coordinates beyond max distance return None safely.
    """
    assert default_geometry.pixel_to_ground(float("nan"), 240.0) is None
    assert default_geometry.pixel_to_ground(320.0, float("nan")) is None
    assert default_geometry.pixel_to_ground(float("inf"), 240.0) is None
    assert default_geometry.pixel_to_ground(320.0, float("-inf")) is None

    # Strict max distance check
    geom_strict = RobotGeometry(max_ground_distance_cm=2.0)
    # A pixel far from center will exceed 2 cm (at (50, 50) distance is ~4.32 cm)
    assert geom_strict.pixel_to_ground(50.0, 50.0) is None


def test_ground_point_from_pixel_alias(default_geometry):
    """Verify ground_point_from_pixel produces identical results to pixel_to_ground."""
    pt1 = default_geometry.pixel_to_ground(200.0, 350.0)
    pt2 = default_geometry.ground_point_from_pixel(200.0, 350.0)
    assert pt1 == pt2


def test_future_planar_calibration_hook(default_geometry):
    """
    Verify the future calibration hook: set_homography raises NotImplementedError
    when a homography matrix is set, and works normally when set to None.
    """
    assert default_geometry.homography_matrix is None
    assert default_geometry.pixel_to_ground(320.0, 240.0) == (0.0, 0.0)

    # Set dummy homography placeholder
    default_geometry.set_homography([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(NotImplementedError):
        default_geometry.pixel_to_ground(320.0, 240.0)

    # Restore None
    default_geometry.set_homography(None)
    assert default_geometry.pixel_to_ground(320.0, 240.0) == (0.0, 0.0)


def test_calibration_evaluation_interface(default_geometry):
    """Verify CalibrationPoint evaluation correctly computes error metrics."""
    cx, cy = default_geometry.cx, default_geometry.cy
    pt1 = default_geometry.pixel_to_ground(cx, cy - 50.0)
    pt2 = default_geometry.pixel_to_ground(cx + 50.0, cy)
    assert pt1 is not None and pt2 is not None

    cal_points = [
        CalibrationPoint(pixel_x=cx, pixel_y=cy - 50.0, known_x_cm=pt1[0], known_y_cm=pt1[1]),
        CalibrationPoint(pixel_x=cx + 50.0, pixel_y=cy, known_x_cm=pt2[0], known_y_cm=pt2[1]),
    ]

    metrics = default_geometry.evaluate_calibration(cal_points)
    assert metrics["total_points"] == 2
    assert metrics["valid_points"] == 2
    assert metrics["mae_x"] == 0.0
    assert metrics["mae_y"] == 0.0
    assert metrics["rmse_x"] == 0.0
    assert metrics["max_distance_error"] == 0.0

    # Perturb known coordinate to test delta measurement
    perturbed = [
        CalibrationPoint(pixel_x=cx, pixel_y=cy - 50.0, known_x_cm=pt1[0] + 2.0, known_y_cm=pt1[1]),
    ]
    p_metrics = default_geometry.evaluate_calibration(perturbed)
    assert p_metrics["valid_points"] == 1
    assert p_metrics["mae_x"] == 2.0
    assert p_metrics["mean_distance_error"] == 2.0
