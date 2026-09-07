"""
Lock-and-Execute State Machine and Target Controller for weed-rover.

Enforces a strict single-target engagement lifecycle:
1. SEARCHING: Scans floor for candidate target dots, selecting closest stable dot.
2. LOCKED_ON_TARGET: Enforces target lock; ignores all other detections.
3. APPROACHING (Step A): Tracks ground coordinates towards the LED target position.
4. BLIND_APPROACH (Blind Spot Handling): When the dot exits the lower camera border,
   dead-reckons the final short distance to the LED using last-known coordinates.
5. STOPPED (Step B): Stops all rover movement when aligned under LED.
6. FIRING (Step C): Triggers prototype LED/laser firing action for configurable duration.
7. COMPLETED (Step D): Marks target completed, clears lock, and resumes SEARCHING.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from vision.dual_dot_detector import Target
    from vision.robot_geometry import RobotGeometry
except ImportError:
    from dual_dot_detector import Target
    from robot_geometry import RobotGeometry


class RoverState(str, Enum):
    SEARCHING = "SEARCHING"
    LOCKED_ON_TARGET = "LOCKED_ON_TARGET"
    APPROACHING = "APPROACHING"
    BLIND_APPROACH = "BLIND_APPROACH"
    STOPPED = "STOPPED"
    FIRING = "FIRING"
    COMPLETED = "COMPLETED"


@dataclass
class ControllerStatus:
    """Snapshot of the controller's current execution status."""
    state: RoverState
    locked_target_id: Optional[int]
    target_ground_camera: Optional[Tuple[float, float]]
    target_ground_led: Optional[Tuple[float, float]]
    fire_active: bool
    remaining_distance_cm: float
    state_message: str


class TargetController:
    """
    State machine controller managing target locking, blind-spot navigation,
    and the prototype execution lifecycle.
    """

    def __init__(
        self,
        geometry: Optional[RobotGeometry] = None,
        target_class: str = "weed",
        alignment_tolerance_cm: float = 1.5,
        blind_spot_row_px: int = 430,
        approach_speed_cm_s: float = 8.0,
        fire_duration_sec: float = 1.5,
        completed_expiry_sec: float = 15.0,
    ):
        """
        Args:
            geometry: RobotGeometry instance for pixel-to-ground translation.
            target_class: Color class to engage ('black_dot' or 'blue_dot').
            alignment_tolerance_cm: Stopping distance tolerance relative to LED in cm.
            blind_spot_row_px: Pixel row near bottom of image where target enters blind spot.
            approach_speed_cm_s: Simulated forward speed for blind dead-reckoning.
            fire_duration_sec: Duration to maintain LED/laser fire signal in seconds.
            completed_expiry_sec: Cooldown duration before a completed target ID can be re-targeted.
        """
        self.geometry = geometry if geometry is not None else RobotGeometry()
        self.target_class = str(target_class)
        self.alignment_tolerance_cm = float(alignment_tolerance_cm)
        self.blind_spot_row_px = int(blind_spot_row_px)
        self.approach_speed_cm_s = float(approach_speed_cm_s)
        self.fire_duration_sec = float(fire_duration_sec)
        self.completed_expiry_sec = float(completed_expiry_sec)

        # State tracking
        self.state: RoverState = RoverState.SEARCHING
        self.locked_target_id: Optional[int] = None
        self.target_pixel: Optional[Tuple[int, int]] = None
        self.target_ground_camera: Optional[Tuple[float, float]] = None
        self.target_ground_led: Optional[Tuple[float, float]] = None
        self.last_known_ground_pos: Optional[Tuple[float, float]] = None

        # Dead-reckoning & firing state
        self.dead_reckon_remaining_cm: float = 0.0
        self.fire_timer: float = 0.0
        self.fire_active: bool = False

        # Completed targets blacklist {target_id: timestamp_completed}
        self.completed_targets: Dict[int, float] = {}

        # Logging / diagnostics
        self.last_state_change_msg: str = "Controller initialized in SEARCHING mode."
        self.state_history: List[str] = []

    def reset(self) -> None:
        """Resets the controller to initial SEARCHING state."""
        self.state = RoverState.SEARCHING
        self.locked_target_id = None
        self.target_pixel = None
        self.target_ground_camera = None
        self.target_ground_led = None
        self.last_known_ground_pos = None
        self.dead_reckon_remaining_cm = 0.0
        self.fire_timer = 0.0
        self.fire_active = False
        self.completed_targets.clear()
        self._record_state_change("Reset to SEARCHING mode.")

    def _record_state_change(self, message: str) -> None:
        """Logs and records state transition."""
        self.last_state_change_msg = message
        self.state_history.append(message)
        if len(self.state_history) > 50:
            self.state_history.pop(0)

    def _clean_expired_completed(self, current_time: float) -> None:
        """Removes expired targets from completed blacklist."""
        expired = [
            tid for tid, t_comp in self.completed_targets.items()
            if (current_time - t_comp) > self.completed_expiry_sec
        ]
        for tid in expired:
            del self.completed_targets[tid]

    def update(
        self,
        targets: List[Target],
        delta_time: float = 0.05,
        current_time: Optional[float] = None,
    ) -> ControllerStatus:
        """
        Executes one step of the Lock-and-Execute State Machine.

        Args:
            targets: List of Target objects from detector.
            delta_time: Time elapsed since last update in seconds.
            current_time: Optional current timestamp (defaults to time.time()).

        Returns:
            ControllerStatus containing current state, target coordinates, and action signals.
        """
        now = current_time if current_time is not None else time.time()
        self._clean_expired_completed(now)
        dt = max(0.001, float(delta_time))

        # ----------------------------------------------------------------------
        # STATE 1: SEARCHING
        # ----------------------------------------------------------------------
        if self.state == RoverState.SEARCHING:
            self.fire_active = False

            # Filter valid candidates: stable, correct class, not completed
            candidates = []
            for t in targets:
                if t.class_name != self.target_class:
                    continue
                if not t.is_stable:
                    continue
                if t.target_id in self.completed_targets:
                    continue

                ground_pos = self.geometry.pixel_to_ground(t.center_x, t.center_y)
                if ground_pos is not None:
                    # Euclidean distance in front of rover
                    dist = math.hypot(ground_pos[0], ground_pos[1])
                    candidates.append((dist, t, ground_pos))

            if candidates:
                # Pick the candidate closest to rover
                candidates.sort(key=lambda c: c[0])
                best_dist, best_target, best_ground = candidates[0]

                self.locked_target_id = best_target.target_id
                self.target_pixel = (best_target.center_x, best_target.center_y)
                self.target_ground_camera = best_ground
                self.target_ground_led = self.geometry.ground_to_led_frame(*best_ground)
                self.last_known_ground_pos = self.target_ground_led

                self.state = RoverState.LOCKED_ON_TARGET
                self._record_state_change(
                    f"LOCKED_ON_TARGET: Locked onto Target #{self.locked_target_id} "
                    f"at Camera=(X={best_ground[0]}cm, Y={best_ground[1]}cm) "
                    f"LED=(X={self.target_ground_led[0]}cm, Y={self.target_ground_led[1]}cm)"
                )

        # ----------------------------------------------------------------------
        # STATE 2: LOCKED_ON_TARGET -> Transition to APPROACHING
        # ----------------------------------------------------------------------
        elif self.state == RoverState.LOCKED_ON_TARGET:
            # Immediate transition into Step A: Move to align
            self.state = RoverState.APPROACHING
            self._record_state_change(
                f"APPROACHING: Beginning alignment toward Target #{self.locked_target_id}"
            )

        # ----------------------------------------------------------------------
        # STATE 3: APPROACHING (Step A)
        # ----------------------------------------------------------------------
        elif self.state == RoverState.APPROACHING:
            # Locate locked target among detections (strict lock: ignore all other targets)
            locked_t = next(
                (t for t in targets if t.target_id == self.locked_target_id and t.class_name == self.target_class),
                None,
            )

            if locked_t is not None:
                # Target is currently visible in camera view
                self.target_pixel = (locked_t.center_x, locked_t.center_y)
                ground_cam = self.geometry.pixel_to_ground(locked_t.center_x, locked_t.center_y)

                if ground_cam is not None:
                    self.target_ground_camera = ground_cam
                    self.target_ground_led = self.geometry.ground_to_led_frame(*ground_cam)
                    self.last_known_ground_pos = self.target_ground_led

                # Check if already aligned with LED
                if self.target_ground_led is not None:
                    led_dist = math.hypot(self.target_ground_led[0], self.target_ground_led[1])
                    if led_dist <= self.alignment_tolerance_cm:
                        # Aligned! Step B: Stop movement
                        self.state = RoverState.STOPPED
                        self._record_state_change(
                            f"STOPPED: Target #{self.locked_target_id} aligned under LED "
                            f"(dist={led_dist:.1f}cm). Stopping all movement."
                        )
                        return self._get_status()

                # Check if target is about to enter lower blind spot
                if locked_t.center_y >= self.blind_spot_row_px:
                    # Target is at the south/bottom border of camera view
                    if self.target_ground_led is not None:
                        self.dead_reckon_remaining_cm = max(0.0, self.target_ground_led[1])
                    else:
                        self.dead_reckon_remaining_cm = self.geometry.camera_to_led_forward_cm

                    self.state = RoverState.BLIND_APPROACH
                    self._record_state_change(
                        f"BLIND_APPROACH: Target #{self.locked_target_id} entered lower camera border "
                        f"(pixel_y={locked_t.center_y}). Dead-reckoning remaining {self.dead_reckon_remaining_cm:.1f}cm."
                    )

            else:
                # Target was NOT detected in camera frame during approach!
                # Because the camera is mounted ahead of the LED, the dot has passed below the camera view.
                if self.last_known_ground_pos is not None:
                    self.dead_reckon_remaining_cm = max(0.0, self.last_known_ground_pos[1])
                    self.state = RoverState.BLIND_APPROACH
                    self._record_state_change(
                        f"BLIND_APPROACH: Target #{self.locked_target_id} out of camera view. "
                        f"Dead-reckoning final {self.dead_reckon_remaining_cm:.1f}cm to LED."
                    )
                else:
                    # Lost target with no prior ground estimate; reset to search
                    self.state = RoverState.SEARCHING
                    self.locked_target_id = None
                    self._record_state_change("Lost target during approach with no coordinates. Resuming SEARCHING.")

        # ----------------------------------------------------------------------
        # STATE 4: BLIND_APPROACH (Blind Spot Dead-Reckoning)
        # ----------------------------------------------------------------------
        elif self.state == RoverState.BLIND_APPROACH:
            # Advance simulated rover motion through blind spot
            dist_step = self.approach_speed_cm_s * dt
            self.dead_reckon_remaining_cm = max(0.0, self.dead_reckon_remaining_cm - dist_step)

            # Update simulated position relative to LED
            lateral_x = self.last_known_ground_pos[0] if self.last_known_ground_pos else 0.0
            self.target_ground_led = (round(lateral_x, 2), round(self.dead_reckon_remaining_cm, 2))

            if self.dead_reckon_remaining_cm <= self.alignment_tolerance_cm:
                # Step B: Stop all movement completely
                self.state = RoverState.STOPPED
                self._record_state_change(
                    f"STOPPED: Target #{self.locked_target_id} reached LED alignment position. "
                    f"All movement halted."
                )

        # ----------------------------------------------------------------------
        # STATE 5: STOPPED -> Transition to FIRING
        # ----------------------------------------------------------------------
        elif self.state == RoverState.STOPPED:
            # Step C: Trigger the prototype LED/laser firing action
            self.state = RoverState.FIRING
            self.fire_timer = self.fire_duration_sec
            self.fire_active = True
            self._record_state_change(
                f"FIRING: Triggering LED action on Target #{self.locked_target_id} "
                f"for {self.fire_duration_sec:.1f}s."
            )

        # ----------------------------------------------------------------------
        # STATE 6: FIRING (Step C in progress)
        # ----------------------------------------------------------------------
        elif self.state == RoverState.FIRING:
            self.fire_active = True
            self.fire_timer -= dt

            if self.fire_timer <= 0.0:
                self.fire_active = False
                self.state = RoverState.COMPLETED
                self._record_state_change(
                    f"COMPLETED: LED action finished on Target #{self.locked_target_id}."
                )

        # ----------------------------------------------------------------------
        # STATE 7: COMPLETED -> Reset and Resume Search (Step D)
        # ----------------------------------------------------------------------
        elif self.state == RoverState.COMPLETED:
            if self.locked_target_id is not None:
                self.completed_targets[self.locked_target_id] = now

            completed_id = self.locked_target_id
            self.locked_target_id = None
            self.target_pixel = None
            self.target_ground_camera = None
            self.target_ground_led = None
            self.last_known_ground_pos = None
            self.dead_reckon_remaining_cm = 0.0
            self.fire_active = False

            # Resume SEARCHING mode to pick next dot
            self.state = RoverState.SEARCHING
            self._record_state_change(
                f"SEARCHING: Target #{completed_id} marked COMPLETED. Lock cleared, resuming search."
            )

        return self._get_status()

    def _get_status(self) -> ControllerStatus:
        """Builds immutable status snapshot."""
        rem_dist = 0.0
        if self.state in (RoverState.LOCKED_ON_TARGET, RoverState.APPROACHING):
            if self.target_ground_led is not None:
                rem_dist = math.hypot(self.target_ground_led[0], self.target_ground_led[1])
        elif self.state == RoverState.BLIND_APPROACH:
            rem_dist = self.dead_reckon_remaining_cm

        return ControllerStatus(
            state=self.state,
            locked_target_id=self.locked_target_id,
            target_ground_camera=self.target_ground_camera,
            target_ground_led=self.target_ground_led,
            fire_active=self.fire_active,
            remaining_distance_cm=round(rem_dist, 2),
            state_message=self.last_state_change_msg,
        )

    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """
        Draws state machine HUD and target tracking reticles onto the frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. State-dependent colors & labels
        state_colors = {
            RoverState.SEARCHING: ((180, 100, 0), (255, 255, 255)),        # Dark Blue / White text
            RoverState.LOCKED_ON_TARGET: ((0, 180, 255), (0, 0, 0)),       # Gold / Black text
            RoverState.APPROACHING: ((0, 140, 255), (255, 255, 255)),     # Orange / White text
            RoverState.BLIND_APPROACH: ((0, 200, 255), (0, 0, 0)),         # Yellow / Black text
            RoverState.STOPPED: ((0, 0, 220), (255, 255, 255)),           # Red / White text
            RoverState.FIRING: ((200, 0, 255), (255, 255, 255)),          # Bright Purple-Magenta
            RoverState.COMPLETED: ((0, 200, 0), (255, 255, 255)),         # Green / White text
        }

        badge_bg, text_color = state_colors.get(
            self.state, ((50, 50, 50), (255, 255, 255))
        )

        # 2. Main State Banner (Top-Right)
        if self.state == RoverState.SEARCHING:
            banner_text = f"STATE: SEARCHING ({self.target_class})"
        elif self.state == RoverState.LOCKED_ON_TARGET:
            banner_text = f"STATE: LOCKED ON TARGET #{self.locked_target_id}"
        elif self.state == RoverState.APPROACHING:
            rem = self.target_ground_led[1] if self.target_ground_led else 0.0
            banner_text = f"APPROACHING #{self.locked_target_id} | Dist: {rem:.1f}cm"
        elif self.state == RoverState.BLIND_APPROACH:
            banner_text = f"BLIND DEAD-RECKONING | Rem: {self.dead_reckon_remaining_cm:.1f}cm"
        elif self.state == RoverState.STOPPED:
            banner_text = f"STOPPED (ALIGNED UNDER LED)"
        elif self.state == RoverState.FIRING:
            banner_text = f"🔥 FIRING LED: {max(0.0, self.fire_timer):.1f}s remaining"
        elif self.state == RoverState.COMPLETED:
            banner_text = f"TARGET COMPLETED -> RESUMING"
        else:
            banner_text = f"STATE: {self.state.value}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (bw, bh), _ = cv2.getTextSize(banner_text, font, font_scale, thickness)

        bx2 = w - 12
        bx1 = bx2 - bw - 16
        by1 = 12
        by2 = by1 + bh + 14

        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), badge_bg, -1)
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
        cv2.putText(
            annotated,
            banner_text,
            (bx1 + 8, by1 + bh + 6),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

        # 3. Target Reticle if target pixel is known and visible
        if self.target_pixel is not None and self.state in (
            RoverState.LOCKED_ON_TARGET,
            RoverState.APPROACHING,
        ):
            tx, ty = self.target_pixel
            # Reticle corners
            r_size = 18
            cv2.circle(annotated, (tx, ty), 4, (0, 220, 255), -1)
            cv2.line(annotated, (tx - r_size, ty - r_size), (tx - r_size + 8, ty - r_size), (0, 220, 255), 2)
            cv2.line(annotated, (tx - r_size, ty - r_size), (tx - r_size, ty - r_size + 8), (0, 220, 255), 2)
            cv2.line(annotated, (tx + r_size, ty - r_size), (tx + r_size - 8, ty - r_size), (0, 220, 255), 2)
            cv2.line(annotated, (tx + r_size, ty - r_size), (tx + r_size, ty - r_size + 8), (0, 220, 255), 2)
            cv2.line(annotated, (tx - r_size, ty + r_size), (tx - r_size + 8, ty + r_size), (0, 220, 255), 2)
            cv2.line(annotated, (tx - r_size, ty + r_size), (tx - r_size, ty + r_size - 8), (0, 220, 255), 2)
            cv2.line(annotated, (tx + r_size, ty + r_size), (tx + r_size - 8, ty + r_size), (0, 220, 255), 2)
            cv2.line(annotated, (tx + r_size, ty + r_size), (tx + r_size, ty + r_size - 8), (0, 220, 255), 2)

            # Ground coordinates badge
            if self.target_ground_camera:
                gx, gy = self.target_ground_camera
                lx, ly = self.target_ground_led if self.target_ground_led else (0.0, 0.0)
                coord_text = f"Ground: ({gx:+.1f}, {gy:.1f})cm | LED: ({lx:+.1f}, {ly:.1f})cm"
                cv2.putText(
                    annotated,
                    coord_text,
                    (max(10, tx - 100), max(20, ty - 24)),
                    font,
                    0.42,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 4. Blind Spot Boundary Line (South border warning)
        cv2.line(
            annotated,
            (0, self.blind_spot_row_px),
            (w, self.blind_spot_row_px),
            (0, 140, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            "CAMERA BLIND SPOT THRESHOLD",
            (10, self.blind_spot_row_px - 6),
            font,
            0.35,
            (0, 140, 255),
            1,
            cv2.LINE_AA,
        )

        # 5. Full Screen Warning if Firing
        if self.fire_active:
            cv2.rectangle(annotated, (0, 0), (w, h), (200, 0, 255), 4)

        return annotated
