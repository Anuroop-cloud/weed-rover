"""
Unit tests for ContinuousWeedExecutor and the continuous-forward execution model.

Covers all 13 required test scenarios:
1. No weed -> no firing event.
2. One weed visible -> weed_detected = 1.
3. Same weed visible for many consecutive frames -> only ONE firing event.
4. Two weeds visible simultaneously -> both independently progress toward firing.
5. First weed reaches firing boundary -> its column is fired.
6. Second weed remains visible -> NOT discarded just because first weed fired.
7. Multiple weeds can produce sequential firing events.
8. Column is preserved from detection to firing.
9. Crop markers never produce firing events.
10. No target IDs are required.
11. No target-lock state is required.
12. Rover movement is never commanded by this perception layer.
13. Output remains weed_detected = 0/1, column = 0..7.
"""

from dataclasses import dataclass
import os
import sys
import pytest
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.robot_geometry import RobotGeometry
from vision.control_interface import LEDMatrixMapper
from vision.continuous_executor import ContinuousWeedExecutor, ContinuousStatus


@dataclass
class SimpleMarker:
    """Mock detection object matching the crop/weed detector contract."""
    class_name: str
    center_x: float
    center_y: float


@pytest.fixture
def executor() -> ContinuousWeedExecutor:
    """Fixture providing ContinuousWeedExecutor configured with fast 0.1s travel delay."""
    geom = RobotGeometry(
        camera_height_cm=6.0,
        camera_tilt_deg=0.0,
        camera_to_led_forward_cm=5.0,
        camera_to_led_lateral_cm=0.0,
        image_width=640,
        image_height=480,
    )
    return ContinuousWeedExecutor(
        geometry=geom,
        matrix_mapper=LEDMatrixMapper(),
        firing_boundary_row_px=430,
        rover_forward_speed_cm_s=8.0,
        camera_to_led_forward_cm=5.0,
        firing_delay_sec=0.1,      # Fast 100ms travel delay for testing
        fire_duration_sec=0.2,     # 200ms fire duration
        debounce_cooldown_sec=1.0,
    )


def test_1_no_weed_no_firing_event(executor):
    """Scenario 1: With no weeds in frame, weed_detected=0, column=-1, no firing occurs."""
    status = executor.update([], delta_time=0.033, current_time=1.0)
    assert status.weed_detected == 0
    assert status.column == -1
    assert status.is_firing is False
    assert len(status.firing_columns) == 0
    assert len(executor._firing_events) == 0


def test_2_one_weed_visible_detected(executor):
    """Scenario 2: One weed visible in frame produces weed_detected=1 and mapped column."""
    # Weed at center_x = 320 (center of camera -> Column 3 or 4), row 200 (above boundary)
    w = SimpleMarker(class_name="weed", center_x=320, center_y=200)
    status = executor.update([w], delta_time=0.033, current_time=1.0)

    assert status.weed_detected == 1
    assert status.column in (3, 4)
    assert status.is_firing is False  # Above firing boundary: NO FIRING YET
    assert len(executor._firing_events) == 0


def test_3_same_weed_20_frames_only_one_firing_event(executor):
    """
    Scenario 3: Same physical weed visible across 20 consecutive frames as rover moves
    forward must create exactly ONE firing event, NOT 20.
    """
    # Frames 1-15: weed moves downward from row 200 to row 420 (above boundary 430)
    t = 1.0
    for i in range(15):
        w = SimpleMarker(class_name="weed", center_x=320, center_y=200 + i * 14)
        status = executor.update([w], delta_time=0.033, current_time=t)
        assert status.weed_detected == 1
        assert status.is_firing is False
        assert len(executor._firing_events) == 0  # 0 fires while above boundary
        t += 0.033

    # Frame 16: weed crosses firing boundary (row 435 >= 430)
    w_cross = SimpleMarker(class_name="weed", center_x=320, center_y=435)
    executor.update([w_cross], delta_time=0.033, current_time=t)
    t += 0.033
    assert len(executor._firing_events) == 1  # Exactly ONE firing event scheduled!

    # Frames 17-25: weed continues moving near bottom border (row 445, 455, 465...)
    for i in range(9):
        w_near = SimpleMarker(class_name="weed", center_x=320, center_y=445 + i * 5)
        executor.update([w_near], delta_time=0.033, current_time=t)
        t += 0.033
        # Must STILL be exactly 1 firing event (no duplicate firing!)
        assert len(executor._firing_events) == 1


def test_4_two_weeds_visible_simultaneously_independent(executor):
    """
    Scenario 4: Two weeds visible simultaneously at different columns independently
    progress toward the firing boundary without locking onto one or discarding the other.
    """
    # Weed A at column 1 (left), Weed B at column 6 (right)
    # Weed A is at row 350, Weed B is higher at row 150
    w_a = SimpleMarker(class_name="weed", center_x=160, center_y=350)
    w_b = SimpleMarker(class_name="weed", center_x=480, center_y=150)

    status = executor.update([w_a, w_b], delta_time=0.033, current_time=1.0)

    assert status.weed_detected == 1
    assert status.active_weeds_count == 2
    # Both columns represented in active tracks
    assert len(status.active_weed_columns) == 2
    assert len(executor._tracks) == 2


def test_5_first_weed_crosses_boundary_fires_its_column(executor):
    """Scenario 5: When first weed crosses firing boundary, its column is scheduled and fired."""
    w_a = SimpleMarker(class_name="weed", center_x=160, center_y=435)  # Col 1, crossed boundary
    w_b = SimpleMarker(class_name="weed", center_x=480, center_y=200)  # Col 6, far above

    col_a = executor.matrix_mapper.x_to_column(executor.geometry.pixel_to_led_ground(160, 435)[0])

    # Crossing frame at t=1.0
    status = executor.update([w_a, w_b], delta_time=0.033, current_time=1.0)
    assert len(executor._firing_events) == 1
    assert executor._firing_events[0].column == col_a

    # Advance time past firing_delay_sec (0.1s) -> firing is active!
    status_firing = executor.update([w_b], delta_time=0.033, current_time=1.12)
    assert status_firing.is_firing is True
    assert col_a in status_firing.firing_columns


def test_6_second_weed_not_discarded_after_first_fires(executor):
    """
    Scenario 6: The second weed remains actively tracked and is NOT discarded
    just because the first weed crossed and fired.
    """
    w_a = SimpleMarker(class_name="weed", center_x=160, center_y=435)  # crossed
    w_b = SimpleMarker(class_name="weed", center_x=480, center_y=200)  # still in view

    executor.update([w_a, w_b], delta_time=0.033, current_time=1.0)

    # Weed A is now gone off-camera; Weed B advances to row 300
    w_b_next = SimpleMarker(class_name="weed", center_x=480, center_y=300)
    status = executor.update([w_b_next], delta_time=0.033, current_time=1.2)

    # Weed B is still active and tracked!
    assert status.weed_detected == 1
    assert any(tr.center_x == 480 and tr.has_fired is False for tr in executor._tracks)


def test_7_multiple_weeds_produce_sequential_firing_events(executor):
    """Scenario 7: Multiple weeds produce sequential firing events as each crosses the boundary."""
    # Weed 1 at col 1 crosses at t=1.0
    w1 = SimpleMarker(class_name="weed", center_x=160, center_y=435)
    executor.update([w1], delta_time=0.033, current_time=1.0)

    # Fire 1 occurs at t=1.1 to 1.3
    status1 = executor.update([], delta_time=0.033, current_time=1.15)
    assert status1.is_firing is True

    # Weed 2 at col 6 crosses at t=2.0
    w2 = SimpleMarker(class_name="weed", center_x=480, center_y=440)
    executor.update([w2], delta_time=0.033, current_time=2.0)

    # Fire 2 occurs at t=2.1 to 2.3
    status2 = executor.update([], delta_time=0.033, current_time=2.15)
    assert status2.is_firing is True
    col2 = executor.matrix_mapper.x_to_column(executor.geometry.pixel_to_led_ground(480, 440)[0])
    assert col2 in status2.firing_columns


def test_8_column_is_preserved_from_detection_to_firing(executor):
    """Scenario 8: Column 6 calculated at detection is identically preserved at firing."""
    # Target X=480 pixels maps to ground X ~ +1.0cm -> Column 6
    w = SimpleMarker(class_name="weed", center_x=480, center_y=200)
    status_init = executor.update([w], delta_time=0.033, current_time=1.0)
    detected_col = status_init.column

    # Move down to boundary
    w.center_y = 435
    executor.update([w], delta_time=0.033, current_time=1.1)

    # Advance to firing window
    status_fire = executor.update([], delta_time=0.033, current_time=1.22)
    assert status_fire.is_firing is True
    assert status_fire.firing_columns[0] == detected_col


def test_9_crop_markers_never_produce_firing_events(executor):
    """Scenario 9: Crop markers (●) are completely ignored for control and never trigger firing."""
    c1 = SimpleMarker(class_name="crop", center_x=320, center_y=200)
    c2 = SimpleMarker(class_name="crop", center_x=320, center_y=445)  # past boundary

    status = executor.update([c1, c2], delta_time=0.033, current_time=1.0)
    assert status.crops_count == 2
    assert status.weed_detected == 0
    assert status.column == -1
    assert status.is_firing is False
    assert len(executor._firing_events) == 0

    # Advance time: no firing ever occurs
    status_later = executor.update([], delta_time=0.033, current_time=1.5)
    assert status_later.is_firing is False
    assert status_later.weed_detected == 0


def test_10_no_target_ids_are_required_or_exposed(executor):
    """Scenario 10: Detections and control output require NO persistent target IDs."""
    w = SimpleMarker(class_name="weed", center_x=320, center_y=200)
    # Notice SimpleMarker has NO target_id attribute!
    assert not hasattr(w, "target_id")

    status = executor.update([w], delta_time=0.033, current_time=1.0)
    assert status.weed_detected == 1
    # Status contains no target ID fields
    assert not hasattr(status, "locked_target_id")
    assert not hasattr(status, "target_id")


def test_11_no_target_lock_state_required(executor):
    """Scenario 11: System does not use SEARCHING, LOCKED, APPROACHING, or STOPPED states."""
    assert not hasattr(executor, "state")
    w = SimpleMarker(class_name="weed", center_x=320, center_y=200)
    status = executor.update([w], delta_time=0.033, current_time=1.0)
    assert not hasattr(status, "state")


def test_12_rover_movement_never_commanded(executor):
    """Scenario 12: Perception layer only outputs weed detection & column, NEVER motion commands."""
    w = SimpleMarker(class_name="weed", center_x=320, center_y=200)
    status = executor.update([w], delta_time=0.033, current_time=1.0)
    payload = status.to_control_payload()

    # Payload MUST only contain 'weed_detected' and 'column'
    assert set(payload.keys()) == {"weed_detected", "column"}
    for forbidden in ("turn", "speed", "steer", "stop", "heading", "linear", "angular"):
        assert forbidden not in payload


def test_13_output_strictly_adheres_to_contract(executor):
    """Scenario 13: Output adheres strictly to weed_detected = 0/1 and column = 0..7 or -1."""
    # 1. No weed
    payload_empty = executor.update([], delta_time=0.033, current_time=1.0).to_control_payload()
    assert payload_empty == {"weed_detected": 0, "column": -1}

    # 2. Weed present
    w = SimpleMarker(class_name="weed", center_x=320, center_y=200)
    payload_weed = executor.update([w], delta_time=0.033, current_time=1.05).to_control_payload()
    assert payload_weed["weed_detected"] == 1
    assert 0 <= payload_weed["column"] <= 7


def test_hud_rendering_clean(executor):
    """Verify draw_hud renders without Target #XXX and without crashing."""
    frame = np.full((480, 640, 3), 220, dtype=np.uint8)
    w = SimpleMarker(class_name="weed", center_x=320, center_y=200)
    c = SimpleMarker(class_name="crop", center_x=150, center_y=150)

    status = executor.update([w, c], delta_time=0.033, current_time=1.0)
    hud = executor.draw_hud(frame, [w, c], fps=28.0, status=status)

    assert hud.shape == (480, 640, 3)
    assert not np.array_equal(hud, frame)
