"""
Unit tests for Laptop -> ESP32 Vision & Serial Integration.

Tests all 8 required integration behaviors using a mock serial interface:
1. Startup sends '01\\n' (LED OFF / MOTOR ON).
2. Weed detection creates an LED activation command event ('11\\n').
3. Crop detection creates no LED activation event.
4. Same weed across multiple frames does not spam commands.
5. Calculated column is preserved and logged.
6. Serial connection failure is handled safely without crashing.
7. Shutdown sends '00\\n' (LED OFF / MOTOR OFF).
8. No weed produces no firing command.
"""

from dataclasses import dataclass
import os
import sys
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.robot_geometry import RobotGeometry
from vision.control_interface import LEDMatrixMapper
from vision.continuous_executor import ContinuousWeedExecutor
from vision.esp32_interface import ESP32Interface


@dataclass
class MockMarker:
    """Mock detection object matching MarkerDetection interface."""
    class_name: str
    center_x: float
    center_y: float


@pytest.fixture
def esp32_client() -> ESP32Interface:
    """Fixture providing a mock ESP32Interface."""
    client = ESP32Interface(port="MOCK_COM", baud=115200, mock_mode=True)
    return client


@pytest.fixture
def executor() -> ContinuousWeedExecutor:
    """Fixture providing a ContinuousWeedExecutor with fast 0.05s firing delay."""
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
        firing_delay_sec=0.05,     # Fast 50ms delay for unit testing
        fire_duration_sec=0.15,    # 150ms fire duration
        debounce_cooldown_sec=1.0,
    )


def test_1_startup_sends_01_newline(esp32_client):
    """
    Requirement 1: Immediately after connection, startup sends '01\\n'
    (LED OFF / MOTOR ON). Must NOT send '00' on startup.
    """
    assert esp32_client.is_connected is False
    connected = esp32_client.connect()

    assert connected is True
    assert esp32_client.is_connected is True
    assert esp32_client.command_history[0] == "01"
    assert esp32_client.last_command == "01"
    assert "00" not in esp32_client.command_history


def test_2_weed_detection_creates_led_activation_command(esp32_client, executor):
    """
    Requirement 2: Weed detection crossing firing boundary triggers an LED
    activation command ('11\\n': LED ON / MOTOR ON).
    """
    esp32_client.connect()
    assert esp32_client.last_command == "01"

    # Weed crosses firing boundary (row 435 >= 430) at t=1.0
    w = MockMarker(class_name="weed", center_x=320, center_y=435)
    status = executor.update([w], delta_time=0.033, current_time=1.0)
    assert status.is_firing is False  # In physical travel delay

    # Advance past travel delay (0.05s) to t=1.08 -> firing window active!
    status_firing = executor.update([w], delta_time=0.033, current_time=1.08)
    assert status_firing.is_firing is True

    # Robot runner logic: if status.is_firing -> send 11
    if status_firing.is_firing:
        esp32_client.send_command(ESP32Interface.CMD_FIRE_WHILE_DRIVING, column_info=status_firing.column)

    assert esp32_client.last_command == "11"
    assert "11" in esp32_client.command_history


def test_3_crop_detection_creates_no_led_activation_event(esp32_client, executor):
    """
    Requirement 3: Crop detection (●) is ignored for control and never triggers
    an LED activation command.
    """
    esp32_client.connect()

    # Multiple crops present, some past boundary
    c1 = MockMarker(class_name="crop", center_x=200, center_y=200)
    c2 = MockMarker(class_name="crop", center_x=300, center_y=440)

    for i in range(10):
        status = executor.update([c1, c2], delta_time=0.033, current_time=1.0 + i * 0.033)
        assert status.is_firing is False
        if status.is_firing:
            esp32_client.send_command(ESP32Interface.CMD_FIRE_WHILE_DRIVING)
        else:
            esp32_client.send_command(ESP32Interface.CMD_DRIVE_FORWARD)

    # Must remain driving forward (01), '11' never sent
    assert esp32_client.last_command == "01"
    assert "11" not in esp32_client.command_history


def test_4_same_weed_across_multiple_frames_does_not_spam_commands(esp32_client, executor):
    """
    Requirement 4: Same physical weed visible across multiple frames must NOT
    spam '11' commands. It generates exactly ONE activation event.
    """
    esp32_client.connect()
    initial_cmd_count = len(esp32_client.command_history)

    # Weed moves across 20 consecutive frames
    w = MockMarker(class_name="weed", center_x=320, center_y=420)
    t = 1.0

    # Frame 1: crosses boundary
    w.center_y = 435
    status = executor.update([w], delta_time=0.033, current_time=t)
    if status.is_firing:
        esp32_client.send_command(ESP32Interface.CMD_FIRE_WHILE_DRIVING)
    else:
        esp32_client.send_command(ESP32Interface.CMD_DRIVE_FORWARD)

    # Frames 2-5: during travel delay
    for i in range(3):
        t += 0.015
        status = executor.update([w], delta_time=0.015, current_time=t)
        if status.is_firing:
            esp32_client.send_command(ESP32Interface.CMD_FIRE_WHILE_DRIVING)
        else:
            esp32_client.send_command(ESP32Interface.CMD_DRIVE_FORWARD)

    # Frames 6-10: during firing window
    for i in range(5):
        t += 0.02
        status = executor.update([w], delta_time=0.02, current_time=t)
        if status.is_firing:
            esp32_client.send_command(ESP32Interface.CMD_FIRE_WHILE_DRIVING)
        else:
            esp32_client.send_command(ESP32Interface.CMD_DRIVE_FORWARD)

    # Check how many times '11' was recorded in history
    fire_commands_sent = [c for c in esp32_client.command_history if c == "11"]
    assert len(fire_commands_sent) == 1  # Exactly ONE '11' event, NO spamming!


def test_5_calculated_column_is_preserved_and_logged(esp32_client, executor):
    """
    Requirement 5: Calculated LED column is preserved from perception to firing
    and passed for logging.
    """
    esp32_client.connect()

    # Weed crosses firing boundary
    w = MockMarker(class_name="weed", center_x=480, center_y=435)
    status = executor.update([w], delta_time=0.033, current_time=1.0)
    initial_column = status.column
    assert 0 <= initial_column <= 7

    # Advance to firing window
    status_firing = executor.update([w], delta_time=0.033, current_time=1.08)
    assert status_firing.is_firing is True
    # Column must be identically preserved!
    assert status_firing.column == initial_column

    # Send command with column info
    success = esp32_client.send_command(
        ESP32Interface.CMD_FIRE_WHILE_DRIVING,
        column_info=status_firing.column,
    )
    assert success is True
    assert esp32_client.last_command == "11"


def test_6_serial_connection_failure_handled_safely():
    """
    Requirement 6: Failure to open serial port is handled cleanly without crashing.
    """
    client = ESP32Interface(port="NON_EXISTENT_PORT_XYZ999", baud=115200, mock_mode=False)
    result = client.connect()

    # Must fail cleanly and report false, without raising unhandled exception
    assert result is False
    assert client.is_connected is False
    assert client.status_str == "DISCONNECTED"


def test_7_shutdown_sends_00_newline(esp32_client):
    """
    Requirement 7: On shutdown, '00\\n' (LED OFF / MOTOR OFF) is transmitted
    to ensure the rover does not remain running after laptop process exits.
    """
    esp32_client.connect()
    assert esp32_client.is_connected is True

    # Trigger safe shutdown
    esp32_client.safe_shutdown()

    assert esp32_client.is_connected is False
    assert esp32_client.last_command == "00"
    assert esp32_client.command_history[-1] == "00"


def test_8_no_weed_produces_no_firing_command(esp32_client, executor):
    """
    Requirement 8: When no weed is detected, robot continues driving forward ('01\\n')
    and never sends '11\\n'.
    """
    esp32_client.connect()

    for i in range(10):
        status = executor.update([], delta_time=0.033, current_time=1.0 + i * 0.033)
        assert status.is_firing is False
        if status.is_firing:
            esp32_client.send_command(ESP32Interface.CMD_FIRE_WHILE_DRIVING)
        else:
            esp32_client.send_command(ESP32Interface.CMD_DRIVE_FORWARD)

    assert esp32_client.last_command == "01"
    assert "11" not in esp32_client.command_history
