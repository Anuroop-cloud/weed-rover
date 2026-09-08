"""
ESP32 Serial Communication Module for weed-rover.

Manages the serial connection to the ESP32-P4 and implements the
two-character command protocol:

    00  =  LED OFF  / Motor OFF
    01  =  LED OFF  / Motor ON   (rover moving, no weed)
    10  =  LED ON   / Motor OFF  (weed spotted, rover stopped)
    11  =  LED ON   / Motor ON   (weed detected while moving — fire + drive)

Commands are sent as ASCII strings terminated by a newline character (\\n).

Usage example:
    from vision.esp32_serial import ESP32Serial

    with ESP32Serial(port="/dev/ttyUSB0", baud=115200) as esp:
        esp.send_move()           # 01\\n — rover moving
        esp.send_weed_detected()  # 11\\n — weed found, trigger LED matrix
        esp.send_stop()           # 00\\n — full stop
"""

import time
import threading
from typing import Optional

try:
    import serial
    import serial.serialutil
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Protocol constants (match ESP32 firmware exactly)
# ──────────────────────────────────────────────────────────────────────────────
CMD_LED_OFF_MOTOR_OFF = b"00\n"   # LED OFF, Motor OFF
CMD_LED_OFF_MOTOR_ON  = b"01\n"   # LED OFF, Motor ON   ← startup default
CMD_LED_ON_MOTOR_OFF  = b"10\n"   # LED ON,  Motor OFF
CMD_LED_ON_MOTOR_ON   = b"11\n"   # LED ON,  Motor ON   ← weed trigger


class ESP32Serial:
    """
    Thread-safe serial wrapper for the ESP32-P4 rover firmware.

    Parameters
    ----------
    port : str
        Serial device path, e.g. ``/dev/ttyUSB0`` or ``/dev/ttyACM0``.
    baud : int
        Baud rate matching the firmware (default: 115200).
    timeout : float
        Read timeout in seconds (default: 1.0).
    cooldown_sec : float
        Minimum seconds between identical consecutive commands to avoid
        spamming the ESP32 on every frame (default: 0.5).
    dry_run : bool
        When True, prints commands instead of sending them over serial.
        Useful for testing without hardware connected.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud: int = 115200,
        timeout: float = 1.0,
        cooldown_sec: float = 0.5,
        dry_run: bool = False,
    ) -> None:
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.cooldown_sec = float(cooldown_sec)
        self.dry_run = dry_run

        self._ser: Optional["serial.Serial"] = None   # type: ignore[name-defined]
        self._lock = threading.Lock()
        self._last_cmd: Optional[bytes] = None
        self._last_send_time: float = 0.0
        self._connected: bool = False

        if not dry_run:
            self._open()

    # ── Connection Management ──────────────────────────────────────────────────

    def _open(self) -> None:
        """Opens the serial port; prints a clear error if unavailable."""
        if not SERIAL_AVAILABLE:
            print(
                "[ESP32Serial ERROR] pyserial is not installed.\n"
                "  Install with:  pip install pyserial"
            )
            return

        try:
            self._ser = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=self.timeout,
            )
            # Allow ESP32 to boot / reset after DTR toggle
            time.sleep(2.0)
            self._connected = True
            print(
                f"[ESP32Serial] Connected to {self.port} @ {self.baud} baud."
            )
        except serial.serialutil.SerialException as exc:
            print(
                f"[ESP32Serial ERROR] Cannot open {self.port}: {exc}\n"
                "  Check:\n"
                "    • Is the USB cable plugged in?\n"
                "    • Run: ls /dev/tty* to find the correct port.\n"
                "    • Run: sudo usermod -aG dialout $USER  (then log out/in)\n"
                "    • Set SERIAL_PORT in config/vision_config.py"
            )

    @property
    def is_connected(self) -> bool:
        """True if the serial port is open and ready."""
        if self.dry_run:
            return True
        return self._connected and self._ser is not None and self._ser.is_open

    def close(self) -> None:
        """Closes the serial connection gracefully."""
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.close()
                self._connected = False
                print("[ESP32Serial] Connection closed.")

    # ── Context Manager Support ────────────────────────────────────────────────

    def __enter__(self) -> "ESP32Serial":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Raw Command Sender ─────────────────────────────────────────────────────

    def send_command(self, cmd: bytes, force: bool = False) -> bool:
        """
        Sends a raw command byte string over serial, with cooldown protection.

        Parameters
        ----------
        cmd : bytes
            Raw command bytes (e.g. ``b"11\\n"``).
        force : bool
            If True, bypasses cooldown and sends immediately.

        Returns
        -------
        bool
            True if the command was sent, False if suppressed by cooldown.
        """
        now = time.monotonic()
        with self._lock:
            # Cooldown: suppress identical commands sent too rapidly
            if (
                not force
                and cmd == self._last_cmd
                and (now - self._last_send_time) < self.cooldown_sec
            ):
                return False

            if self.dry_run:
                print(f"[ESP32Serial DRY-RUN] → {cmd.decode().strip()}")
                self._last_cmd = cmd
                self._last_send_time = now
                return True

            if not self.is_connected:
                print("[ESP32Serial WARNING] Not connected — command dropped.")
                return False

            try:
                self._ser.write(cmd)  # type: ignore[union-attr]
                self._ser.flush()     # type: ignore[union-attr]
                self._last_cmd = cmd
                self._last_send_time = now
                return True
            except serial.serialutil.SerialException as exc:
                print(f"[ESP32Serial ERROR] Write failed: {exc}")
                self._connected = False
                return False

    # ── Convenience Command Wrappers ───────────────────────────────────────────

    def send_move(self) -> bool:
        """
        Send ``01\\n`` — LED OFF, Motor ON.
        Call this on startup and whenever no weed is detected.
        """
        return self.send_command(CMD_LED_OFF_MOTOR_ON)

    def send_weed_detected(self) -> bool:
        """
        Send ``11\\n`` — LED ON, Motor ON.
        Call this when a cross/weed marker is confirmed.
        """
        return self.send_command(CMD_LED_ON_MOTOR_ON)

    def send_led_only(self) -> bool:
        """
        Send ``10\\n`` — LED ON, Motor OFF.
        Stops rover and activates LED matrix only.
        """
        return self.send_command(CMD_LED_ON_MOTOR_OFF)

    def send_stop(self) -> bool:
        """
        Send ``00\\n`` — LED OFF, Motor OFF.
        Emergency / full stop command.
        """
        return self.send_command(CMD_LED_OFF_MOTOR_OFF)

    def send_startup(self) -> bool:
        """
        Sends the mandatory startup command (``01\\n``) to set the rover moving.
        Forces immediate delivery regardless of cooldown.
        """
        result = self.send_command(CMD_LED_OFF_MOTOR_ON, force=True)
        if result:
            print("[ESP32Serial] Startup command sent: 01 (motor ON, LED OFF)")
        return result
