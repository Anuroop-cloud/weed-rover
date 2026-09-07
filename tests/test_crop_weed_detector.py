"""
Unit tests for CropWeedDetector and Crop/Weed Perception-to-Control Pipeline.

Verifies the 12 core requirements:
1. Filled circular crop -> detected as crop.
2. X marker -> detected as weed.
3. Crop is NOT detected as weed.
4. X is NOT detected as crop.
5. Multiple crops + one weed -> exactly one weed target.
6. Multiple weeds -> deterministic target selection.
7. Weed pixel position converts correctly to X/Y.
8. Weed X position maps correctly to LED columns 0–7.
9. No weed -> weed_detected=0, column=-1.
10. Weed -> weed_detected=1, column=0..7.
11. 6 cm camera height is used by geometry.
12. Camera-to-LED offset is applied correctly.
"""

import math
import os
import sys
import cv2
import numpy as np
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.vision_config import (
    CAMERA_HEIGHT_CM,
    CAMERA_RESOLUTION_WIDTH,
    CAMERA_RESOLUTION_HEIGHT,
    CAMERA_FOV_HORIZONTAL_DEG,
    CAMERA_FOV_VERTICAL_DEG,
    LED_MATRIX_WIDTH_CM,
    LED_MATRIX_COLUMNS,
    LED_MATRIX_COLUMN_WIDTH_CM,
)
from vision.crop_weed_detector import CropWeedDetector, MarkerDetection
from vision.control_interface import (
    WeedControlPipeline,
    LEDMatrixMapper,
    WeedControlOutput,
    x_to_matrix_column,
)
from vision.robot_geometry import RobotGeometry


def create_canvas(w: int = 640, h: int = 480, bg_val: int = 255) -> np.ndarray:
    """Creates a white/light canvas representing the mock floor workspace."""
    return np.ones((h, w, 3), dtype=np.uint8) * bg_val


def draw_crop_dot(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    radius: int = 16,
    color: tuple = (0, 0, 0),
) -> np.ndarray:
    """Draws a filled circular crop dot (LED-sized)."""
    cv2.circle(canvas, (int(cx), int(cy)), int(radius), color, -1)
    return canvas


def draw_weed_x(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    arm_len: int = 22,
    thickness: int = 5,
    color: tuple = (0, 0, 0),
) -> np.ndarray:
    """Draws an X/cross weed marker."""
    pt1 = (int(cx - arm_len), int(cy - arm_len))
    pt2 = (int(cx + arm_len), int(cy + arm_len))
    pt3 = (int(cx - arm_len), int(cy + arm_len))
    pt4 = (int(cx + arm_len), int(cy - arm_len))
    cv2.line(canvas, pt1, pt2, color, thickness)
    cv2.line(canvas, pt3, pt4, color, thickness)
    return canvas


@pytest.fixture
def detector() -> CropWeedDetector:
    """Fixture providing CropWeedDetector with single-hit acquisition for direct test frames."""
    return CropWeedDetector(stability_min_hits=1, roi=None)


@pytest.fixture
def geometry() -> RobotGeometry:
    """Fixture providing RobotGeometry with 6 cm camera height."""
    return RobotGeometry(
        camera_height_cm=6.0,
        camera_to_led_forward_cm=5.0,
        camera_to_led_lateral_cm=0.0,
    )


@pytest.fixture
def control_pipeline(geometry) -> WeedControlPipeline:
    """Fixture providing WeedControlPipeline configured with 6 cm top-down geometry."""
    return WeedControlPipeline(
        geometry=geometry,
        matrix_mapper=LEDMatrixMapper(matrix_width_cm=3.2, num_columns=8),
        target_classes=["weed"],
        conf_threshold=0.35,
    )


# ==============================================================================
# REQUIREMENT 1: Filled circular crop -> detected as crop
# ==============================================================================
def test_filled_circular_crop_detected_as_crop(detector):
    """1. Verify that a filled circular dot is detected as 'crop'."""
    frame = create_canvas()
    draw_crop_dot(frame, 200, 200, radius=16)

    detections, _ = detector.detect(frame)
    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "crop"
    assert math.hypot(det.center_x - 200, det.center_y - 200) <= 2.0
    assert det.confidence >= 0.70


# ==============================================================================
# REQUIREMENT 2: X marker -> detected as weed
# ==============================================================================
def test_x_marker_detected_as_weed(detector):
    """2. Verify that an X marker is detected as 'weed'."""
    frame = create_canvas()
    draw_weed_x(frame, 320, 240, arm_len=22, thickness=5)

    detections, _ = detector.detect(frame)
    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "weed"
    assert math.hypot(det.center_x - 320, det.center_y - 240) <= 2.0
    assert det.confidence >= 0.70


# ==============================================================================
# REQUIREMENT 3: Crop is NOT detected as weed
# ==============================================================================
def test_crop_is_not_detected_as_weed(detector, control_pipeline):
    """
    3. Verify that circular crops are NEVER classified as weeds
       and NEVER trigger weed control output.
    """
    frame = create_canvas()
    draw_crop_dot(frame, 150, 150, radius=15)
    draw_crop_dot(frame, 350, 250, radius=18)

    detections, _ = detector.detect(frame)
    assert len(detections) == 2
    for d in detections:
        assert d.class_name == "crop"
        assert d.class_name != "weed"

    # Feed to downstream control layer: must produce NO weed
    out = control_pipeline.process_detections(detections)
    assert out.weed_detected == 0
    assert out.column == -1
    assert out.to_control_payload() == {"weed_detected": 0, "column": -1}


# ==============================================================================
# REQUIREMENT 4: X is NOT detected as crop
# ==============================================================================
def test_x_is_not_detected_as_crop(detector):
    """4. Verify that X markers are NEVER classified as crops."""
    frame = create_canvas()
    draw_weed_x(frame, 250, 200, arm_len=20, thickness=5)
    draw_weed_x(frame, 450, 300, arm_len=25, thickness=6)

    detections, _ = detector.detect(frame)
    assert len(detections) == 2
    for d in detections:
        assert d.class_name == "weed"
        assert d.class_name != "crop"


# ==============================================================================
# REQUIREMENT 5: Multiple crops + one weed -> exactly one weed target
# ==============================================================================
def test_multiple_crops_plus_one_weed_yields_one_weed_target(detector, control_pipeline):
    """
    5. In a mixed field with multiple crops and one weed:
       All crops are identified as crop, the weed is identified as weed,
       and the control output targets strictly the one weed.
    """
    frame = create_canvas()
    # 3 crops
    draw_crop_dot(frame, 150, 180, radius=14)
    draw_crop_dot(frame, 280, 320, radius=16)
    draw_crop_dot(frame, 460, 200, radius=15)
    # 1 weed
    draw_weed_x(frame, 320, 240, arm_len=22, thickness=5)

    detections, _ = detector.detect(frame)
    crops = [d for d in detections if d.class_name == "crop"]
    weeds = [d for d in detections if d.class_name == "weed"]

    assert len(crops) == 3
    assert len(weeds) == 1

    # Downstream control layer receives all detections but targets ONLY the weed
    out = control_pipeline.process_detections(detections)
    assert out.weed_detected == 1
    assert 0 <= out.column <= 7
    # Center weed at (320, 240) -> X_cam=0 -> X_led=0 -> column 4
    assert out.column == 4


# ==============================================================================
# REQUIREMENT 6: Multiple weeds -> deterministic target selection
# ==============================================================================
def test_multiple_weeds_deterministic_target_selection(detector, control_pipeline):
    """
    6. When multiple weeds are visible:
       Selects the weed closest to the tool deterministically in ground coordinates.
    """
    frame = create_canvas()
    # Weed 1: Center of image (X_cam=0.0 cm, Y_cam=0.0 cm -> dist to tool = 5.0 cm)
    draw_weed_x(frame, 320, 240, arm_len=20, thickness=5)
    # Weed 2: Farther to the right (X_cam ~ +1.6 cm, Y_cam=0.0 cm -> dist to tool ~ 5.24 cm)
    draw_weed_x(frame, 440, 240, arm_len=20, thickness=5)
    # Weed 3: Far down in the image (Y_cam ~ -2.1 cm -> dist to tool ~ 7.1 cm)
    draw_weed_x(frame, 320, 400, arm_len=20, thickness=5)

    detections, _ = detector.detect(frame)
    weeds = [d for d in detections if d.class_name == "weed"]
    assert len(weeds) == 3

    # Primary run
    out1 = control_pipeline.process_detections(detections)
    assert out1.weed_detected == 1
    assert out1.center_pixel == (320, 240)
    assert out1.column == 4

    # Run with reversed detection order: must select identical closest weed
    out2 = control_pipeline.process_detections(list(reversed(detections)))
    assert out2.weed_detected == 1
    assert out2.center_pixel == out1.center_pixel
    assert out2.column == out1.column


# ==============================================================================
# REQUIREMENT 7: Weed pixel position converts correctly to X/Y
# ==============================================================================
def test_weed_pixel_converts_correctly_to_xy(geometry):
    """
    7. Verifies pixel position converts correctly to ground X/Y using top-down geometry:
       X_camera = (u - cx) * H / fx
       Y_camera = (cy - v) * H / fy
    """
    cx, cy = geometry.cx, geometry.cy
    assert (cx, cy) == (320.0, 240.0)

    # 1. Center pixel
    pt_center = geometry.pixel_to_ground(cx, cy)
    assert pt_center == (0.0, 0.0)

    # 2. Pixel right of center (u = 380, v = 240) -> positive X, zero Y
    pt_right = geometry.pixel_to_ground(380.0, 240.0)
    assert pt_right is not None
    assert pt_right[0] > 0.0
    assert pt_right[1] == 0.0

    # 3. Pixel above center (u = 320, v = 160) -> zero X, positive Y (forward)
    pt_above = geometry.pixel_to_ground(320.0, 160.0)
    assert pt_above is not None
    assert pt_above[0] == 0.0
    assert pt_above[1] > 0.0

    # 4. Pixel below center (u = 320, v = 320) -> zero X, negative Y (backward)
    pt_below = geometry.pixel_to_ground(320.0, 320.0)
    assert pt_below is not None
    assert pt_below[0] == 0.0
    assert pt_below[1] < 0.0


# ==============================================================================
# REQUIREMENT 8: Weed X position maps correctly to LED columns 0–7
# ==============================================================================
def test_weed_x_position_maps_correctly_to_led_columns_0_through_7():
    """
    8. Verifies lateral ground X position maps to discrete LED columns 0 through 7:
       Column width: 0.4 cm, matrix width: 3.2 cm
       Column centers:
       0: -1.4 cm
       1: -1.0 cm
       2: -0.6 cm
       3: -0.2 cm
       4: +0.2 cm
       5: +0.6 cm
       6: +1.0 cm
       7: +1.4 cm
    """
    mapper = LEDMatrixMapper(matrix_width_cm=3.2, num_columns=8)

    centers = {
        0: -1.4,
        1: -1.0,
        2: -0.6,
        3: -0.2,
        4: 0.2,
        5: 0.6,
        6: 1.0,
        7: 1.4,
    }

    for expected_col, center_x in centers.items():
        assert mapper.x_to_column(center_x) == expected_col
        assert x_to_matrix_column(center_x) == expected_col


# ==============================================================================
# REQUIREMENT 9: No weed -> weed_detected=0, column=-1
# ==============================================================================
def test_no_weed_output_contract(control_pipeline):
    """
    9. When no weeds are detected:
       weed_detected = 0
       column = -1
       Payload returns {"weed_detected": 0, "column": -1}.
    """
    out = control_pipeline.process_detections([])
    assert out.weed_detected == 0
    assert out.column == -1
    assert out.to_control_payload() == {"weed_detected": 0, "column": -1}


# ==============================================================================
# REQUIREMENT 10: Weed -> weed_detected=1, column=0..7
# ==============================================================================
def test_weed_detected_output_contract(control_pipeline):
    """
    10. When a valid weed is detected:
        weed_detected = 1
        column = 0..7
        Payload returns {"weed_detected": 1, "column": <col>}.
    """
    weed_det = MarkerDetection(
        class_name="weed",
        target_id=1,
        center_x=320,
        center_y=240,
        area=500.0,
        confidence=0.90,
        box=[300, 220, 340, 260],
        x1=300,
        y1=220,
        x2=340,
        y2=260,
    )

    out = control_pipeline.process_detections([weed_det])
    assert out.weed_detected == 1
    assert out.column == 4  # (320, 240) is center X=0.0 -> column 4
    assert out.to_control_payload() == {
        "weed_detected": 1,
        "column": 4,
    }


# ==============================================================================
# REQUIREMENT 11: 6 cm camera height is used by geometry
# ==============================================================================
def test_six_cm_camera_height_used():
    """
    11. Verifies that the camera height is configured to 6.0 cm in config
        and by default in RobotGeometry.
    """
    assert CAMERA_HEIGHT_CM == 6.0

    geom_default = RobotGeometry()
    assert geom_default.camera_height_cm == 6.0

    # Verify scaling at 6 cm vs 12 cm
    geom_6 = RobotGeometry(camera_height_cm=6.0)
    geom_12 = RobotGeometry(camera_height_cm=12.0)

    res_6 = geom_6.pixel_to_ground(380.0, 200.0)
    res_12 = geom_12.pixel_to_ground(380.0, 200.0)

    assert res_6 is not None and res_12 is not None
    assert math.isclose(res_12[0], 2.0 * res_6[0], rel_tol=1e-2)
    assert math.isclose(res_12[1], 2.0 * res_6[1], rel_tol=1e-2)


# ==============================================================================
# REQUIREMENT 12: Camera-to-LED offset is applied correctly
# ==============================================================================
def test_camera_to_led_offset_applied_correctly():
    """
    12. Verifies camera-to-LED lateral and forward offsets are correctly applied:
        X_led = X_camera - camera_to_led_lateral_cm
        Y_led = Y_camera - camera_to_led_forward_cm
    """
    geom_offset = RobotGeometry(
        camera_height_cm=6.0,
        camera_to_led_forward_cm=4.0,
        camera_to_led_lateral_cm=1.0,
    )
    pipeline_offset = WeedControlPipeline(geometry=geom_offset)

    # Weed at center of camera (X_cam = 0.0, Y_cam = 0.0)
    weed_center = MarkerDetection(
        class_name="weed",
        target_id=1,
        center_x=320,
        center_y=240,
        area=500.0,
        confidence=0.90,
        box=[300, 220, 340, 260],
        x1=300,
        y1=220,
        x2=340,
        y2=260,
    )

    out = pipeline_offset.process_detections([weed_center])
    assert out.weed_detected == 1
    assert out.x_camera_cm == 0.0
    assert out.y_camera_cm == 0.0

    # In LED frame: X_led = 0.0 - 1.0 = -1.0 cm -> maps to Column 1!
    assert out.x_led_cm == -1.0
    assert out.y_led_cm == -4.0
    assert out.column == 1
    assert out.to_control_payload() == {
        "weed_detected": 1,
        "column": 1,
    }


# ==============================================================================
# VISUAL DEBUG OVERLAY TEST (Requirement 10)
# ==============================================================================
def test_visual_debug_overlay_renders_properly(detector):
    """Verify draw_visual_debug produces expected annotations and HUD banner without error."""
    frame = create_canvas()
    draw_crop_dot(frame, 150, 150, radius=15)
    draw_weed_x(frame, 320, 240, arm_len=20, thickness=5)

    detections, _ = detector.detect(frame)
    assert len(detections) == 2

    weed_det = next(d for d in detections if d.class_name == "weed")

    annotated = detector.draw_visual_debug(
        frame=frame,
        detections=detections,
        selected_weed=weed_det,
        selected_x_cm=0.0,
        selected_y_cm=2.5,
        selected_column=4,
        control_payload={"weed_detected": 1, "column": 4},
    )

    assert annotated.shape == (480, 640, 3)
    # Ensure overlay modified pixel values (not identical to raw canvas)
    assert not np.array_equal(annotated, frame)
