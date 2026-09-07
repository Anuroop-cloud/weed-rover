"""
Unit tests for AI to Control Output Interface.

Verifies the 10 core requirements:
1. No detections -> weed_detected=0, column=-1
2. Weed in far-left matrix region -> column=0
3. Weed in each of columns 0 through 7 -> correct column
4. Weed exactly around the center -> correct center column according to defined boundaries
5. Weed slightly left/right of center -> correct neighboring column
6. X position outside the matrix -> safe clamping behavior
7. Camera-to-LED lateral offset -> correct column after applying offset
8. Multiple weeds -> closest valid weed to tool is selected deterministically
9. Confidence below threshold -> weed_detected=0
10. Valid weed -> weed_detected=1
"""

import math
import os
import sys
import numpy as np
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.detector import Detection
from vision.robot_geometry import RobotGeometry
from vision.control_interface import (
    LEDMatrixMapper,
    WeedControlOutput,
    WeedControlPipeline,
    x_to_matrix_column,
)


@pytest.fixture
def default_mapper() -> LEDMatrixMapper:
    """Fixture providing standard 8x8 LED matrix mapper (3.2 cm width, 8 columns)."""
    return LEDMatrixMapper(matrix_width_cm=3.2, num_columns=8)


@pytest.fixture
def test_pipeline() -> WeedControlPipeline:
    """Fixture providing pipeline configured with standard top-down geometry."""
    geom = RobotGeometry(
        camera_height_cm=20.0,
        camera_to_led_forward_cm=5.0,
        camera_to_led_lateral_cm=0.0,
    )
    return WeedControlPipeline(
        detector=None,
        geometry=geom,
        conf_threshold=0.35,
    )


# ==============================================================================
# TEST 1: NO DETECTIONS
# ==============================================================================
def test_no_detections_returns_zero_and_negative_one(test_pipeline):
    """
    1. When no weeds are detected:
       weed_detected = 0
       column = -1
       to_control_payload() returns exact contract payload.
    """
    out = test_pipeline.process_detections([])

    assert isinstance(out, WeedControlOutput)
    assert out.weed_detected == 0
    assert out.column == -1

    payload = out.to_control_payload()
    assert payload == {
        "weed_detected": 0,
        "column": -1,
    }


# ==============================================================================
# TEST 2: WEED IN FAR-LEFT MATRIX REGION
# ==============================================================================
def test_weed_in_far_left_matrix_region(default_mapper):
    """
    2. Weed positioned in the far-left region of the matrix (-1.6 to -1.2 cm):
       Maps strictly to column 0.
    """
    # Column 0 bounds are [-1.6, -1.2) cm, center is -1.4 cm
    assert default_mapper.x_to_column(-1.6) == 0
    assert default_mapper.x_to_column(-1.5) == 0
    assert default_mapper.x_to_column(-1.4) == 0
    assert default_mapper.x_to_column(-1.21) == 0


# ==============================================================================
# TEST 3: WEED IN EACH OF COLUMNS 0 THROUGH 7
# ==============================================================================
def test_weed_in_each_column_0_through_7(default_mapper):
    """
    3. Verify weed at each theoretical column center produces the exact column (0..7):
       col 0: -1.4 cm
       col 1: -1.0 cm
       col 2: -0.6 cm
       col 3: -0.2 cm
       col 4: +0.2 cm
       col 5: +0.6 cm
       col 6: +1.0 cm
       col 7: +1.4 cm
    """
    expected_centers = {
        0: -1.4,
        1: -1.0,
        2: -0.6,
        3: -0.2,
        4: 0.2,
        5: 0.6,
        6: 1.0,
        7: 1.4,
    }

    for col, center_x in expected_centers.items():
        assert math.isclose(default_mapper.column_center_x(col), center_x, abs_tol=1e-3)
        assert default_mapper.x_to_column(center_x) == col
        assert x_to_matrix_column(center_x) == col


# ==============================================================================
# TEST 4: WEED EXACTLY AROUND THE CENTER
# ==============================================================================
def test_weed_exactly_around_center(default_mapper):
    """
    4. Verify mapping at center boundaries:
       Center X=0.0 cm is the boundary between col 3 ([-0.4, 0.0)) and col 4 ([0.0, 0.4)).
       X = 0.0 cm maps to column 4.
       X = -0.2 cm (center of col 3) maps to column 3.
       X = +0.2 cm (center of col 4) maps to column 4.
    """
    assert default_mapper.x_to_column(0.0) == 4
    assert default_mapper.x_to_column(-0.2) == 3
    assert default_mapper.x_to_column(0.2) == 4


# ==============================================================================
# TEST 5: WEED SLIGHTLY LEFT/RIGHT OF CENTER
# ==============================================================================
def test_weed_slightly_left_and_right_of_center(default_mapper):
    """
    5. Verify neighboring column assignment for slight deviations around center:
       X = -0.01 cm (slightly left) -> column 3
       X = +0.01 cm (slightly right) -> column 4
       X = -0.10 cm (left of center) -> column 3
       X = +0.10 cm (right of center) -> column 4
    """
    assert default_mapper.x_to_column(-0.01) == 3
    assert default_mapper.x_to_column(0.01) == 4
    assert default_mapper.x_to_column(-0.10) == 3
    assert default_mapper.x_to_column(0.10) == 4


# ==============================================================================
# TEST 6: X POSITION OUTSIDE THE MATRIX (SAFE CLAMPING)
# ==============================================================================
def test_x_outside_matrix_clamps_safely(default_mapper):
    """
    6. Verify that coordinates beyond the physical matrix bounds clamp safely
       to column 0 (left) or column 7 (right) without raising exceptions.
    """
    # Far left out of bounds
    assert default_mapper.x_to_column(-1.61) == 0
    assert default_mapper.x_to_column(-2.5) == 0
    assert default_mapper.x_to_column(-10.0) == 0
    assert default_mapper.x_to_column(-100.0) == 0

    # Far right out of bounds
    assert default_mapper.x_to_column(1.61) == 7
    assert default_mapper.x_to_column(2.5) == 7
    assert default_mapper.x_to_column(10.0) == 7
    assert default_mapper.x_to_column(100.0) == 7


# ==============================================================================
# TEST 7: CAMERA-TO-LED LATERAL OFFSET
# ==============================================================================
def test_camera_to_led_lateral_offset():
    """
    7. Verify that camera-to-LED lateral offset is correctly applied:
       X_led = X_camera - camera_to_led_lateral_cm

       If a weed is centered in the camera image (X_cam = 0.0 cm):
       - If LED is offset by +1.0 cm (to the right of camera),
         X_led = 0.0 - 1.0 = -1.0 cm -> maps to column 1.
       - If LED is offset by -1.0 cm (to the left of camera),
         X_led = 0.0 - (-1.0) = +1.0 cm -> maps to column 6.
    """
    # Pipeline with LED mounted 1.0 cm to the right of camera
    geom_offset_right = RobotGeometry(
        camera_height_cm=20.0,
        camera_to_led_forward_cm=5.0,
        camera_to_led_lateral_cm=1.0,
    )
    pipeline_right = WeedControlPipeline(geometry=geom_offset_right)

    # Synthetic detection directly centered in camera frame (cx=320, cy=240 -> X_cam=0, Y_cam=0)
    det_center = Detection(
        class_id=0,
        class_name="weed",
        confidence=0.90,
        box=[300, 220, 340, 260],
        x1=300,
        y1=220,
        x2=340,
        y2=260,
        center_x=320,
        center_y=240,
    )

    out_right = pipeline_right.process_detections([det_center])
    assert out_right.weed_detected == 1
    assert out_right.x_camera_cm == 0.0
    assert out_right.x_led_cm == -1.0  # 0.0 - 1.0
    assert out_right.column == 1  # -1.0 cm is column 1

    # Pipeline with LED mounted 1.0 cm to the left of camera
    geom_offset_left = RobotGeometry(
        camera_height_cm=20.0,
        camera_to_led_forward_cm=5.0,
        camera_to_led_lateral_cm=-1.0,
    )
    pipeline_left = WeedControlPipeline(geometry=geom_offset_left)
    out_left = pipeline_left.process_detections([det_center])
    assert out_left.weed_detected == 1
    assert out_left.x_led_cm == 1.0  # 0.0 - (-1.0)
    assert out_left.column == 6  # +1.0 cm is column 6


# ==============================================================================
# TEST 8: MULTIPLE WEEDS (CLOSEST SELECTED DETERMINISTICALLY)
# ==============================================================================
def test_multiple_weeds_selects_closest_deterministically():
    """
    8. When multiple weeds are detected:
       The pipeline must deterministically select the one closest to the tool
       in ground coordinates, NOT simply the highest confidence detection.
    """
    geom = RobotGeometry(
        camera_height_cm=20.0,
        camera_to_led_forward_cm=0.0,
        camera_to_led_lateral_cm=0.0,
    )
    pipeline = WeedControlPipeline(geometry=geom)

    # Center is (320, 240).
    # Weed 1: High confidence (0.95), but FAR from tool (center_x=500, center_y=100)
    det_far_high_conf = Detection(
        class_id=0,
        class_name="weed",
        confidence=0.95,
        box=[480, 80, 520, 120],
        x1=480,
        y1=80,
        x2=520,
        y2=120,
        center_x=500,
        center_y=100,
    )

    # Weed 2: Lower confidence (0.45), but VERY CLOSE to tool (center_x=325, center_y=245)
    det_close_low_conf = Detection(
        class_id=0,
        class_name="weed",
        confidence=0.45,
        box=[310, 230, 340, 260],
        x1=310,
        y1=230,
        x2=340,
        y2=260,
        center_x=325,
        center_y=245,
    )

    # Feed in arbitrary order: far first, then close
    out = pipeline.process_detections([det_far_high_conf, det_close_low_conf])

    assert out.weed_detected == 1
    # Must select the CLOSEST weed (Weed 2) despite lower confidence!
    assert out.confidence == 0.45
    assert out.center_pixel == (325, 245)

    # Feed in reverse order: close first, then far
    out_rev = pipeline.process_detections([det_close_low_conf, det_far_high_conf])
    assert out_rev.confidence == 0.45
    assert out_rev.column == out.column


# ==============================================================================
# TEST 9: CONFIDENCE BELOW THRESHOLD
# ==============================================================================
def test_confidence_below_threshold_ignored(test_pipeline):
    """
    9. Detections below the confidence threshold (0.35) must be discarded:
       weed_detected = 0
       column = -1
    """
    det_low = Detection(
        class_id=0,
        class_name="weed",
        confidence=0.20,  # Below default 0.35 threshold
        box=[300, 200, 340, 240],
        x1=300,
        y1=200,
        x2=340,
        y2=240,
        center_x=320,
        center_y=220,
    )

    out = test_pipeline.process_detections([det_low])
    assert out.weed_detected == 0
    assert out.column == -1
    assert out.to_control_payload() == {"weed_detected": 0, "column": -1}


# ==============================================================================
# TEST 10: VALID WEED
# ==============================================================================
def test_valid_weed_produces_control_output(test_pipeline):
    """
    10. A valid weed (confidence >= threshold) produces:
        weed_detected = 1
        column = 0..7
        to_control_payload() returns valid dict.
    """
    # Detection at (320, 240) -> X_cam = 0.0 -> X_led = 0.0 -> column 4
    det_valid = Detection(
        class_id=0,
        class_name="weed",
        confidence=0.88,
        box=[300, 220, 340, 260],
        x1=300,
        y1=220,
        x2=340,
        y2=260,
        center_x=320,
        center_y=240,
    )

    out = test_pipeline.process_detections([det_valid])
    assert out.weed_detected == 1
    assert 0 <= out.column <= 7
    assert out.column == 4

    payload = out.to_control_payload()
    assert payload == {
        "weed_detected": 1,
        "column": 4,
    }


# ==============================================================================
# ADDITIONAL EDGE CASE & INTEGRATION TESTS
# ==============================================================================
def test_configurable_matrix_mapper():
    """Verify matrix mapper supports non-default width and column count."""
    # 10 columns across 5.0 cm (0.5 cm per column, half_width = 2.5 cm)
    custom_mapper = LEDMatrixMapper(matrix_width_cm=5.0, num_columns=10)
    assert custom_mapper.column_width_cm == 0.5
    assert custom_mapper.num_columns == 10
    # Left edge is -2.5 cm -> column 0
    assert custom_mapper.x_to_column(-2.5) == 0
    # Right edge is +2.5 cm -> clamped to 9
    assert custom_mapper.x_to_column(2.5) == 9
    # Center X=0.0 cm -> (0.0 + 2.5) / 0.5 = 5.0 -> column 5
    assert custom_mapper.x_to_column(0.0) == 5


def test_target_class_filtering():
    """Verify pipeline filters by target class if specified."""
    geom = RobotGeometry()
    pipeline = WeedControlPipeline(
        geometry=geom,
        target_classes=["weed"],
    )

    det_crop = Detection(
        class_id=1,
        class_name="crop",
        confidence=0.90,
        box=[300, 220, 340, 260],
        x1=300,
        y1=220,
        x2=340,
        y2=260,
        center_x=320,
        center_y=240,
    )

    out = pipeline.process_detections([det_crop])
    assert out.weed_detected == 0
    assert out.column == -1


def test_empty_or_none_frame_handling(test_pipeline):
    """Verify process_frame handles empty or None frames safely."""
    assert test_pipeline.process_frame(None).to_control_payload() == {"weed_detected": 0, "column": -1}
    assert test_pipeline.process_frame(np.zeros((0, 0, 3), dtype=np.uint8)).to_control_payload() == {"weed_detected": 0, "column": -1}
