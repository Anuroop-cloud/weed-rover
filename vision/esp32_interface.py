"""
ESP32 Serial Communication Interface for weed-rover.

Implements the laptop-to-ESP32 command protocol over USB serial:
- 00 = LED OFF / MOTOR OFF (Safe Stop)
- 01 = LED OFF / MOTOR ON  (Continuous Forward Motion)
- 10 = LED ON  / MOTOR OFF (LED Standstill)
- 11 = LED ON  / MOTOR ON  (Firing while Driving Forward)

Serial configuration:
- 115200 baud
- Newline-terminated ASCII commands (e.g. b"01\\n")
"""

import logging
import os
import sys
import time
from typing import Any, List, Optional

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None


class ESP32Interface:
    """
    Manages serial connection, command transmission, state tracking,
    and safe shutdown for the ESP32-P4 firmware.
    """

    CMD_SAFE_STOP = "00"             # LED OFF / MOTOR OFF
    CMD_DRIVE_FORWARD = "01"         # LED OFF / MOTOR ON
    CMD_LED_STANDSTILL = "10"        # LED ON  / MOTOR OFF
    CMD_FIRE_WHILE_DRIVING = "11"    # LED ON  / MOTOR ON

    VALID_COMMANDS = {
        CMD_SAFE_STOP,
        CMD_DRIVE_FORWARD,
        CMD_LED_STANDSTILL,
        CMD_FIRE_WHILE_DRIVING,
    }

    def __init__(
        self,
        port: Optional[str] = None,
        baud: int = 115200,
        mock_mode: bool = False,
        timeout: float = 1.0,
    ):
        """
        Args:
            port: Serial port name (e.g. 'COM3', 'COM5', '/dev/ttyUSB0').
            baud: Baud rate (default: 115200).
            mock_mode: If True, uses an in-memory mock serial connection.
            timeout: Serial read/write timeout in seconds.
        """
        self.port = str(port) if port else None
        self.baud = int(baud)
        self.mock_mode = bool(mock_mode)
        self.timeout = float(timeout)

        self._ser = None
        self.is_connected: bool = False
        self.last_command: Optional[str] = None
        self.last_command_time: float = 0.0
        self.command_history: List[str] = []

    @classmethod
    def list_available_ports(cls) -> List[str]:
        """Returns list of available serial COM port names."""
        if serial is None or not hasattr(serial.tools, "list_ports"):
            return []
        ports = serial.tools.list_ports.comports()
        return [p.device for p in ports]

    def connect(self) -> bool:
        """
        Opens serial port and transmits startup command '01\\n' (LED OFF / MOTOR ON).
        Returns True if connection and startup succeed, False otherwise.
        """
        if self.mock_mode:
            self.is_connected = True
            print(f"[ESP32] Mock Serial initialized (baud={self.baud}).")
            self.send_command(self.CMD_DRIVE_FORWARD, force=True)
            return True

        if serial is None:
            print("[ESP32 Error] pyserial library is not available. Please install via 'pip install pyserial'.")
            self.is_connected = False
            return False

        if not self.port:
            available = self.list_available_ports()
            print(f"[ESP32 Error] No serial port specified. Available ports: {available if available else 'None found'}")
            self.is_connected = False
            return False

        try:
            self._ser = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
            # Give USB-UART a brief moment to stabilize DTR/RTS
            time.sleep(0.1)
            self.is_connected = True
            print(f"[ESP32] Connected to {self.port} @ {self.baud} baud.")

            # Startup Requirement: immediately send '01\n' (LED OFF / MOTOR ON)
            self.send_command(self.CMD_DRIVE_FORWARD, force=True)
            return True

        except Exception as e:
            print(f"[ESP32 Error] Failed to open serial port '{self.port}': {e}")
            self.is_connected = False
            self._ser = None
            return False

    def send_command(
        self,
        cmd: str,
        column_info: Optional[int] = None,
        force: bool = False,
    ) -> bool:
        """
        Sends a newline-terminated command to the ESP32.
        Prevents redundant spamming if the command hasn't changed.

        Args:
            cmd: Command string ('00', '01', '10', or '11').
            column_info: Optional calculated physical LED column (0..7) for diagnostic logging.
            force: If True, sends even if identical to last command.

        Returns:
            bool: True if transmission succeeded.
        """
        cmd_clean = cmd.strip()
        if cmd_clean not in self.VALID_COMMANDS:
            print(f"[ESP32 Error] Invalid command '{cmd_clean}'. Must be one of {self.VALID_COMMANDS}")
            return False

        # Anti-spam check: do not resend identical command unless forced
        if not force and cmd_clean == self.last_command:
            return True

        # Log with explicit column explanation
        if cmd_clean == self.CMD_FIRE_WHILE_DRIVING and column_info is not None:
            print(f"[SERIAL] WEED DETECTED -> COLUMN {column_info} -> LED COMMAND 11")
        elif cmd_clean == self.CMD_DRIVE_FORWARD:
            if self.last_command == self.CMD_FIRE_WHILE_DRIVING:
                print(f"[SERIAL] FIRING COMPLETE -> LED COMMAND 01 (LED OFF / MOTOR ON)")
            else:
                print(f"[SERIAL] STARTUP / DRIVE -> LED COMMAND 01 (LED OFF / MOTOR ON)")
        elif cmd_clean == self.CMD_SAFE_STOP:
            print(f"[SERIAL] SHUTDOWN -> LED COMMAND 00 (LED OFF / MOTOR OFF)")
        else:
            print(f"[SERIAL] Command sent: {cmd_clean}")

        # Transmit newline-terminated ASCII bytes
        payload = f"{cmd_clean}\n".encode("ascii")

        if self.mock_mode:
            self.last_command = cmd_clean
            self.last_command_time = time.time()
            self.command_history.append(cmd_clean)
            return True

        if not self.is_connected or self._ser is None:
            return False

        try:
            self._ser.write(payload)
            self._ser.flush()
            self.last_command = cmd_clean
            self.last_command_time = time.time()
            self.command_history.append(cmd_clean)
            return True
        except Exception as e:
            print(f"[ESP32 Error] Failed writing to serial port: {e}")
            return False

    def safe_shutdown(self) -> None:
        """
        Sends safe stop command '00\\n' (LED OFF / MOTOR OFF) and closes connection.
        Ensures the robot does not remain running after laptop process exits.
        """
        if self.is_connected:
            try:
                # Send '00\n' before closing
                self.send_command(self.CMD_SAFE_STOP, force=True)
            except Exception:
                pass

        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

        self.is_connected = False
        print("[ESP32] Serial connection safely closed.")

    @property
    def status_str(self) -> str:
        """Human-readable connection status string for HUD overlay."""
        if self.mock_mode:
            return "MOCK"
        if self.is_connected:
            return f"CONNECTED ({self.port})" if self.port else "CONNECTED"
        return "DISCONNECTED"

    @property
    def last_cmd_str(self) -> str:
        """Last transmitted command for HUD overlay."""
        return self.last_command if self.last_command is not None else "--"
