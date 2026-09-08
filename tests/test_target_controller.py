"""
Unit tests for TargetController and Lock-and-Execute State Machine.
Tests target selection, strict locking, blind spot dead-reckoning, stop & fire cycle,
and return to search mode.
"""

import os
import sys
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from vision.dual_dot_detector import Target
from vision.robot_geometry import RobotGeometry
from vision.target_controller import TargetController, RoverState


@pytest.fixture
def test_controller() -> TargetController:
    """Fixture providing a TargetController with fast simulation times for unit testing."""
    geom = RobotGeometry(
        camera_height_cm=20.0,
        camera_tilt_deg=0.0,
        camera_to_led_forward_cm=-15.0,  # Tool mounted 15 cm behind camera along travel path
        camera_to_led_lateral_cm=0.0,
    )
    return TargetController(
        geometry=geom,
        target_class="black_dot",
        alignment_tolerance_cm=1.5,
        blind_spot_row_px=430,
        approach_speed_cm_s=10.0,
        fire_duration_sec=0.5,
        completed_expiry_sec=10.0,
    )


def test_initial_state_is_searching(test_controller):
    """Verify controller starts in SEARCHING state."""
    assert test_controller.state == RoverState.SEARCHING
    assert test_controller.locked_target_id is None
    assert test_controller.fire_active is False


def test_closest_stable_target_selected(test_controller):
    """Verify closest stable black dot is selected and locked."""
    # Far target at row 100 (farther ahead: Y ~ +6.1 cm)
    far_target = Target(
        class_name="black_dot",
        target_id=10,
        center_x=320,
        center_y=100,
        area=100.0,
        is_stable=True,
    )
    # Near target at row 220 (nearer ahead: Y ~ +0.9 cm)
    near_target = Target(
        class_name="black_dot",
        target_id=20,
        center_x=320,
        center_y=220,
        area=100.0,
        is_stable=True,
    )

    status = test_controller.update([far_target, near_target], delta_time=0.05)

    assert status.state == RoverState.LOCKED_ON_TARGET
    assert status.locked_target_id == 20  # Selected the closer dot!
    assert test_controller.locked_target_id == 20


def test_strict_lock_enforcement_refuses_switch(test_controller):
    """
    Once locked on Target #1, the controller must refuse to switch to a closer Target #2.
    """
    target1 = Target(
        class_name="black_dot",
        target_id=1,
        center_x=320,
        center_y=300,
        area=100.0,
        is_stable=True,
    )

    # Tick 1: Lock onto target 1
    test_controller.update([target1], delta_time=0.05)
    assert test_controller.locked_target_id == 1

    # Tick 2: Moves to APPROACHING
    test_controller.update([target1], delta_time=0.05)
    assert test_controller.state == RoverState.APPROACHING

    # Tick 3: A new, much closer dot (target 2 at row 400) appears
    target2 = Target(
        class_name="black_dot",
        target_id=2,
        center_x=320,
        center_y=400,
        area=150.0,
        is_stable=True,
    )
    test_controller.update([target1, target2], delta_time=0.05)

    # Must remain locked on Target #1!
    assert test_controller.locked_target_id == 1
    assert test_controller.state == RoverState.APPROACHING


def test_blind_spot_dead_reckoning_trigger_and_progress(test_controller):
    """
    When the locked target passes into the lower camera border (row >= 430),
    the controller transitions to BLIND_APPROACH and dead-reckons using last known coordinates.
    """
    target1 = Target(
        class_name="black_dot",
        target_id=1,
        center_x=320,
        center_y=300,
        area=100.0,
        is_stable=True,
    )

    # Lock and begin approach
    test_controller.update([target1], delta_time=0.05)
    test_controller.update([target1], delta_time=0.05)
    assert test_controller.state == RoverState.APPROACHING

    # Target reaches lower border row 440 >= 430
    near_border_target = Target(
        class_name="black_dot",
        target_id=1,
        center_x=320,
        center_y=440,
        area=120.0,
        is_stable=True,
    )
    status = test_controller.update([near_border_target], delta_time=0.05)

    assert status.state == RoverState.BLIND_APPROACH
    initial_remaining = test_controller.dead_reckon_remaining_cm
    assert initial_remaining > 0

    # Advance time: remaining distance should decrease by approach_speed * dt
    dt = 0.2
    test_controller.update([], delta_time=dt)  # target no longer visible in camera
    assert test_controller.state == RoverState.BLIND_APPROACH
    assert test_controller.dead_reckon_remaining_cm < initial_remaining


def test_full_execution_cycle_stop_fire_complete_resume(test_controller):
    """
    Test entire lifecycle:
    SEARCHING -> LOCKED -> APPROACHING -> BLIND_APPROACH -> STOPPED -> FIRING -> COMPLETED -> SEARCHING.
    """
    target = Target(
        class_name="black_dot",
        target_id=5,
        center_x=320,
        center_y=350,
        area=100.0,
        is_stable=True,
    )

    # 1. SEARCHING -> LOCKED_ON_TARGET
    test_controller.update([target], delta_time=0.05)
    assert test_controller.state == RoverState.LOCKED_ON_TARGET
    assert test_controller.locked_target_id == 5

    # 2. LOCKED_ON_TARGET -> APPROACHING
    test_controller.update([target], delta_time=0.05)
    assert test_controller.state == RoverState.APPROACHING

    # 3. Target exits view -> BLIND_APPROACH
    test_controller.update([], delta_time=0.05)
    assert test_controller.state == RoverState.BLIND_APPROACH
    assert test_controller.locked_target_id == 5

    # 4. Dead-reckon until alignment tolerance is reached -> STOPPED
    remaining_time = test_controller.dead_reckon_remaining_cm / test_controller.approach_speed_cm_s
    test_controller.update([], delta_time=remaining_time + 0.1)
    assert test_controller.state == RoverState.STOPPED
    assert test_controller.fire_active is False

    # 5. STOPPED -> FIRING (Step C)
    test_controller.update([], delta_time=0.05)
    assert test_controller.state == RoverState.FIRING
    assert test_controller.fire_active is True
    assert test_controller.fire_timer > 0

    # 6. Wait for fire_duration_sec to elapse -> COMPLETED
    test_controller.update([], delta_time=test_controller.fire_duration_sec + 0.1)
    assert test_controller.state == RoverState.COMPLETED
    assert test_controller.fire_active is False

    # 7. COMPLETED -> SEARCHING (Step D finish)
    test_controller.update([], delta_time=0.05)
    assert test_controller.state == RoverState.SEARCHING
    assert test_controller.locked_target_id is None
    assert 5 in test_controller.completed_targets


def test_completed_target_not_immediately_retargeted(test_controller):
    """
    Verify that Target #5 is ignored right after completion, even if still present in detections.
    """
    target5 = Target(
        class_name="black_dot",
        target_id=5,
        center_x=320,
        center_y=350,
        area=100.0,
        is_stable=True,
    )

    # Manually mark #5 completed
    test_controller.completed_targets[5] = 1000.0

    # Running search update at time 1001.0 (within 10s expiry)
    status = test_controller.update([target5], delta_time=0.05, current_time=1001.0)
    assert status.state == RoverState.SEARCHING
    assert status.locked_target_id is None


def test_hud_rendering_without_error(test_controller):
    """Verify draw_hud draws state badge and reticle without crashing."""
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    annotated = test_controller.draw_hud(frame)
    assert annotated.shape == (480, 640, 3)

    # Test during firing
    test_controller.state = RoverState.FIRING
    test_controller.fire_active = True
    test_controller.fire_timer = 1.0
    annotated_fire = test_controller.draw_hud(frame)
    assert annotated_fire.shape == (480, 640, 3)


def test_multiple_weeds_acknowledged_simultaneously(test_controller):
    """
    Verify that ALL detected weed markers are acknowledged as targets,
    tracked across their respective LED matrix columns, and visible in locked_target_ids.
    """
    t1 = Target(class_name="black_dot", target_id=101, center_x=120, center_y=200, area=100.0, is_stable=True)
    t2 = Target(class_name="black_dot", target_id=102, center_x=320, center_y=220, area=100.0, is_stable=True)
    t3 = Target(class_name="black_dot", target_id=103, center_x=520, center_y=210, area=100.0, is_stable=True)

    status = test_controller.update([t1, t2, t3], delta_time=0.05)

    assert status.state == RoverState.LOCKED_ON_TARGET
    assert 101 in status.locked_target_ids
    assert 102 in status.locked_target_ids
    assert 103 in status.locked_target_ids
    assert len(status.active_columns) >= 2
    assert status.fire_active is False


def test_no_firing_above_blind_spot_line(test_controller):
    """
    Verify that weeds visible above the camera blind spot threshold line (row < 430)
    remain in APPROACHING mode and NEVER trigger premature firing.
    """
    t1 = Target(class_name="black_dot", target_id=201, center_x=200, center_y=250, area=100.0, is_stable=True)
    t2 = Target(class_name="black_dot", target_id=202, center_x=440, center_y=320, area=100.0, is_stable=True)

    # Tick 1: Locked
    test_controller.update([t1, t2], delta_time=0.05)
    # Tick 2: Approaching
    status = test_controller.update([t1, t2], delta_time=0.05)

    assert status.state == RoverState.APPROACHING
    assert status.fire_active is False
    assert 201 in test_controller.locked_target_ids
    assert 202 in test_controller.locked_target_ids

    # Tick 3: Advance closer to row 410 (< 430 blind spot line)
    t1.center_y = 390
    t2.center_y = 410
    status = test_controller.update([t1, t2], delta_time=0.05)

    assert status.state == RoverState.APPROACHING
    assert status.fire_active is False  # Must NOT fire above blind spot line!


def test_simultaneous_multi_column_firing_after_crossing_blind_spot(test_controller):
    """
    Verify full multi-column execution cycle:
    1. Multiple weeds tracked across different columns
    2. Crossing blind spot line (row >= 430) initiates BLIND_APPROACH
    3. Dead-reckons to LED alignment
    4. Halts at STOPPED
    5. Fires ALL columns SIMULTANEOUSLY in FIRING mode
    6. Marks all target IDs as COMPLETED
    """
    t1 = Target(class_name="black_dot", target_id=301, center_x=160, center_y=300, area=100.0, is_stable=True)
    t2 = Target(class_name="black_dot", target_id=302, center_x=480, center_y=320, area=100.0, is_stable=True)

    # 1. Lock on targets
    test_controller.update([t1, t2], delta_time=0.05)
    test_controller.update([t1, t2], delta_time=0.05)
    assert test_controller.state == RoverState.APPROACHING

    # 2. Weeds cross blind spot line (row >= 430)
    t1.center_y = 435
    t2.center_y = 445
    status = test_controller.update([t1, t2], delta_time=0.05)

    assert status.state == RoverState.BLIND_APPROACH
    assert status.fire_active is False
    firing_cols = list(test_controller.firing_columns)
    assert len(firing_cols) >= 2  # Multiple distinct columns identified!

    # 3. Dead-reckon to LED alignment
    rem_time = test_controller.dead_reckon_remaining_cm / test_controller.approach_speed_cm_s
    test_controller.update([], delta_time=rem_time + 0.1)
    assert test_controller.state == RoverState.STOPPED
    assert test_controller.fire_active is False

    # 4. Trigger simultaneous firing
    status = test_controller.update([], delta_time=0.05)
    assert status.state == RoverState.FIRING
    assert status.fire_active is True
    assert status.firing_columns == firing_cols

    # 5. Elapse firing duration -> COMPLETED
    test_controller.update([], delta_time=test_controller.fire_duration_sec + 0.1)
    assert test_controller.state == RoverState.COMPLETED
    assert test_controller.fire_active is False

    # 6. Resume SEARCHING & verify both targets completed
    test_controller.update([], delta_time=0.05)
    assert test_controller.state == RoverState.SEARCHING
    assert 301 in test_controller.completed_targets
    assert 302 in test_controller.completed_targets
