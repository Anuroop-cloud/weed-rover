"""
Lock-and-Execute State Machine and Target Controller for weed-rover.

Enforces a multi-column target engagement lifecycle:
1. SEARCHING: Scans floor for candidate target dots/weeds across all columns.
2. LOCKED_ON_TARGET: Enforces target lock for all detected weeds across columns.
3. APPROACHING (Step A): Tracks ground coordinates towards the LED target position.
   Strictly prevents premature firing while weeds are visible above the camera blind spot line.
4. BLIND_APPROACH (Blind Spot Handling): When weeds reach or cross the blind spot threshold (row >= 430),
   dead-reckons the remaining distance to the LED array under the chassis.
5. STOPPED (Step B): Stops all rover movement when aligned under LED array.
6. FIRING (Step C): Triggers prototype LED/laser firing action simultaneously across ALL
   active weed columns for the configured duration.
7. COMPLETED (Step D): Marks all fired targets completed, clears lock, and resumes SEARCHING.
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
    from vision.control_interface import x_to_matrix_column
except ImportError:
    from dual_dot_detector import Target
    from robot_geometry import RobotGeometry
    try:
        from control_interface import x_to_matrix_column
    except ImportError:
        def x_to_matrix_column(x_cm: float) -> int:
            col = int(math.floor(round((float(x_cm) + 1.6) / 0.4, 6)))
            return max(0, min(7, col))


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
    locked_target_ids: List[int] = field(default_factory=list)
    active_columns: List[int] = field(default_factory=list)
    firing_columns: List[int] = field(default_factory=list)


class TargetController:
    """
    State machine controller managing multi-target locking, blind-spot navigation,
    and simultaneous multi-column prototype execution lifecycle.
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
            target_class: Target class to engage ('weed', 'black_dot', etc.).
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

        # Multi-target state tracking
        self.state: RoverState = RoverState.SEARCHING
        self.locked_target_id: Optional[int] = None
        self.locked_target_ids: List[int] = []
        self.active_columns: List[int] = []
        self.firing_target_ids: List[int] = []
        self.firing_columns: List[int] = []
        self.target_records: Dict[int, Dict[str, Any]] = {}

        # Primary target coordinates
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
        self.locked_target_ids = []
        self.active_columns = []
        self.firing_target_ids = []
        self.firing_columns = []
        self.target_records.clear()
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

    def _get_target_info(self, t: Target) -> Optional[Dict[str, Any]]:
        """Calculates ground coordinates, LED frame coords, and LED matrix column for a target."""
        g_cam = self.geometry.pixel_to_ground(t.center_x, t.center_y)
        if g_cam is None:
            return None
        g_led = self.geometry.ground_to_led_frame(*g_cam)
        col = x_to_matrix_column(g_led[0])
        dist = math.hypot(g_cam[0], g_cam[1])
        return {
            "target_id": t.target_id,
            "pixel": (int(t.center_x), int(t.center_y)),
            "ground_cam": g_cam,
            "ground_led": g_led,
            "column": col,
            "dist": dist,
            "target": t,
        }

    def update(
        self,
        targets: List[Target],
        delta_time: float = 0.05,
        current_time: Optional[float] = None,
    ) -> ControllerStatus:
        """
        Executes one step of the Multi-Column Target Execution State Machine.

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

                info = self._get_target_info(t)
                if info is not None:
                    candidates.append(info)

            if candidates:
                # Sort candidates by distance (closest to rover first)
                candidates.sort(key=lambda c: c["dist"])
                primary = candidates[0]

                self.locked_target_id = primary["target_id"]
                self.target_pixel = primary["pixel"]
                self.target_ground_camera = primary["ground_cam"]
                self.target_ground_led = primary["ground_led"]
                self.last_known_ground_pos = primary["ground_led"]

                # Acknowledge ALL detected weeds as active targets
                self.locked_target_ids = [c["target_id"] for c in candidates]
                self.active_columns = sorted(list(set(c["column"] for c in candidates)))
                self.target_records = {c["target_id"]: c for c in candidates}

                self.state = RoverState.LOCKED_ON_TARGET
                cols_str = str(self.active_columns)
                self._record_state_change(
                    f"LOCKED_ON_TARGET: Acknowledged {len(self.locked_target_ids)} target(s) #{self.locked_target_ids} "
                    f"across Columns {cols_str} (Primary #{self.locked_target_id})"
                )

        # ----------------------------------------------------------------------
        # STATE 2: LOCKED_ON_TARGET -> Transition to APPROACHING
        # ----------------------------------------------------------------------
        elif self.state == RoverState.LOCKED_ON_TARGET:
            self.state = RoverState.APPROACHING
            cols_str = str(self.active_columns)
            self._record_state_change(
                f"APPROACHING: Beginning alignment toward {len(self.locked_target_ids)} target(s) #{self.locked_target_ids} "
                f"across Columns {cols_str}"
            )

        # ----------------------------------------------------------------------
        # STATE 3: APPROACHING (Step A)
        # ----------------------------------------------------------------------
        elif self.state == RoverState.APPROACHING:
            # Refresh coordinates for all tracked targets currently visible
            visible_targets = [
                t for t in targets
                if t.class_name == self.target_class and t.target_id not in self.completed_targets
            ]

            # Update records for currently visible targets
            for t in visible_targets:
                info = self._get_target_info(t)
                if info is not None:
                    self.target_records[t.target_id] = info
                    if t.target_id not in self.locked_target_ids:
                        self.locked_target_ids.append(t.target_id)

            # Recompute active columns from all active records
            if self.target_records:
                self.active_columns = sorted(list(set(r["column"] for r in self.target_records.values())))

            # Update primary target reference
            if self.locked_target_id in self.target_records:
                primary = self.target_records[self.locked_target_id]
                self.target_pixel = primary["pixel"]
                self.target_ground_camera = primary["ground_cam"]
                self.target_ground_led = primary["ground_led"]
                self.last_known_ground_pos = primary["ground_led"]

            # TARGET CANNOT FIRE WHILE IN VIEW ABOVE THE BLIND SPOT LINE (row < 430).
            # Check if any tracked target reaches or crosses the camera blind spot threshold (row >= 430)
            crossing_targets = [
                rec for rec in self.target_records.values()
                if rec["pixel"][1] >= self.blind_spot_row_px
            ]
            primary_out_of_view = (
                self.locked_target_id is not None
                and self.locked_target_id not in [t.target_id for t in visible_targets]
            )

            if crossing_targets or (primary_out_of_view and self.locked_target_ids):
                # Targets are crossing into the blind spot!
                self.firing_target_ids = list(self.locked_target_ids)
                self.firing_columns = list(self.active_columns) if self.active_columns else [
                    x_to_matrix_column(self.target_ground_led[0]) if self.target_ground_led else 3
                ]

                blind_dist = (
                    abs(self.target_ground_led[1])
                    if self.target_ground_led and abs(self.target_ground_led[1]) > 0
                    else self.geometry.camera_to_led_forward_cm
                )
                self.dead_reckon_remaining_cm = max(self.alignment_tolerance_cm + 0.5, blind_dist)

                self.state = RoverState.BLIND_APPROACH
                self._record_state_change(
                    f"BLIND_APPROACH: Target(s) #{self.firing_target_ids} entered blind spot. "
                    f"Dead-reckoning {self.dead_reckon_remaining_cm:.1f}cm through blind spot to LED array for Columns {self.firing_columns}."
                )

            elif not visible_targets and not self.last_known_ground_pos:
                # Lost all targets with no prior ground estimate; reset to search
                self.state = RoverState.SEARCHING
                self.locked_target_id = None
                self.locked_target_ids = []
                self.active_columns = []
                self.target_records.clear()
                self._record_state_change("Lost all targets during approach with no coordinates. Resuming SEARCHING.")

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
                    f"STOPPED: Target(s) #{self.firing_target_ids} reached LED alignment position. "
                    f"Simultaneous Columns: {self.firing_columns}. All movement halted."
                )

        # ----------------------------------------------------------------------
        # STATE 5: STOPPED -> Transition to FIRING
        # ----------------------------------------------------------------------
        elif self.state == RoverState.STOPPED:
            # Step C: Trigger simultaneous multi-column LED/laser firing action
            self.state = RoverState.FIRING
            self.fire_timer = self.fire_duration_sec
            self.fire_active = True
            self._record_state_change(
                f"FIRING: Triggering simultaneous LED action on Columns {self.firing_columns} "
                f"for Targets #{self.firing_target_ids} for {self.fire_duration_sec:.1f}s."
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
                    f"COMPLETED: Simultaneous LED action finished on Columns {self.firing_columns} "
                    f"for Targets #{self.firing_target_ids}."
                )

        # ----------------------------------------------------------------------
        # STATE 7: COMPLETED -> Reset and Resume Search (Step D)
        # ----------------------------------------------------------------------
        elif self.state == RoverState.COMPLETED:
            for tid in self.firing_target_ids:
                self.completed_targets[tid] = now
            if self.locked_target_id is not None:
                self.completed_targets[self.locked_target_id] = now

            completed_ids = list(self.firing_target_ids) if self.firing_target_ids else (
                [self.locked_target_id] if self.locked_target_id else []
            )
            completed_cols = list(self.firing_columns)

            # Clear locks and buffers
            self.locked_target_id = None
            self.locked_target_ids = []
            self.active_columns = []
            self.firing_target_ids = []
            self.firing_columns = []
            self.target_records.clear()
            self.target_pixel = None
            self.target_ground_camera = None
            self.target_ground_led = None
            self.last_known_ground_pos = None
            self.dead_reckon_remaining_cm = 0.0
            self.fire_active = False

            # Resume SEARCHING mode
            self.state = RoverState.SEARCHING
            self._record_state_change(
                f"SEARCHING: Targets #{completed_ids} (Columns {completed_cols}) marked COMPLETED. Locks cleared, resuming search."
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
            locked_target_ids=list(self.locked_target_ids),
            active_columns=list(self.active_columns),
            firing_columns=list(self.firing_columns),
        )

    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """
        Draws state machine HUD, multi-target tracking reticles, column annotations,
        and simultaneous firing overlays onto the frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. State-dependent badge colors
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
        cols_disp = self.active_columns if self.state in (
            RoverState.SEARCHING, RoverState.LOCKED_ON_TARGET, RoverState.APPROACHING
        ) else self.firing_columns

        if self.state == RoverState.SEARCHING:
            banner_text = f"STATE: SEARCHING ({self.target_class.upper()})"
        elif self.state == RoverState.LOCKED_ON_TARGET:
            banner_text = f"LOCKED {len(self.locked_target_ids)} TARGETS | Cols: {cols_disp}"
        elif self.state == RoverState.APPROACHING:
            rem = self.target_ground_led[1] if self.target_ground_led else 0.0
            if len(self.locked_target_ids) > 1:
                banner_text = f"APPROACHING {len(self.locked_target_ids)} WEEDS | Cols: {cols_disp} | Dist: {rem:.1f}cm"
            else:
                banner_text = f"APPROACHING #{self.locked_target_id} | Col: {cols_disp} | Dist: {rem:.1f}cm"
        elif self.state == RoverState.BLIND_APPROACH:
            banner_text = f"BLIND DEAD-RECKONING | Cols: {self.firing_columns} | Rem: {self.dead_reckon_remaining_cm:.1f}cm"
        elif self.state == RoverState.STOPPED:
            banner_text = f"STOPPED (ALIGNED UNDER LED) | Cols: {self.firing_columns}"
        elif self.state == RoverState.FIRING:
            banner_text = f"🔥 FIRING COLUMNS: {self.firing_columns} ({max(0.0, self.fire_timer):.1f}s remaining)"
        elif self.state == RoverState.COMPLETED:
            banner_text = f"COLUMNS {self.firing_columns} COMPLETED -> RESUMING"
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

        # 3. Target Reticles on ALL acknowledged weeds
        if self.state in (RoverState.LOCKED_ON_TARGET, RoverState.APPROACHING):
            for tid, record in self.target_records.items():
                tx, ty = record["pixel"]
                col = record["column"]
                is_primary = (tid == self.locked_target_id)
                reticle_color = (0, 220, 255) if is_primary else (255, 200, 0)
                r_size = 18

                # Corner brackets
                cv2.circle(annotated, (tx, ty), 4, reticle_color, -1)
                cv2.line(annotated, (tx - r_size, ty - r_size), (tx - r_size + 8, ty - r_size), reticle_color, 2)
                cv2.line(annotated, (tx - r_size, ty - r_size), (tx - r_size, ty - r_size + 8), reticle_color, 2)
                cv2.line(annotated, (tx + r_size, ty - r_size), (tx + r_size - 8, ty - r_size), reticle_color, 2)
                cv2.line(annotated, (tx + r_size, ty - r_size), (tx + r_size, ty - r_size + 8), reticle_color, 2)
                cv2.line(annotated, (tx - r_size, ty + r_size), (tx - r_size + 8, ty + r_size), reticle_color, 2)
                cv2.line(annotated, (tx - r_size, ty + r_size), (tx - r_size, ty + r_size - 8), reticle_color, 2)
                cv2.line(annotated, (tx + r_size, ty + r_size), (tx + r_size - 8, ty + r_size), reticle_color, 2)
                cv2.line(annotated, (tx + r_size, ty + r_size), (tx + r_size, ty + r_size - 8), reticle_color, 2)

                # Ground / LED coordinates badge for every weed
                gx, gy = record["ground_cam"]
                lx, ly = record["ground_led"]
                badge_title = f"TARGET #{tid} | COL={col}" + (" [PRI]" if is_primary else "")
                coord_text = f"Ground:({gx:+.1f},{gy:.1f})cm LED:({lx:+.1f},{ly:.1f})cm"

                cv2.putText(
                    annotated,
                    badge_title,
                    (max(10, tx - 70), max(18, ty - 26)),
                    font,
                    0.42,
                    reticle_color,
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    coord_text,
                    (max(10, tx - 100), max(30, ty - 12)),
                    font,
                    0.38,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 4. Camera Blind Spot Threshold Line
        cv2.line(
            annotated,
            (0, self.blind_spot_row_px),
            (w, self.blind_spot_row_px),
            (0, 140, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            "CAMERA BLIND SPOT THRESHOLD (APPROACH ZONE -> NO FIRING ABOVE THIS LINE)",
            (10, self.blind_spot_row_px - 6),
            font,
            0.36,
            (0, 140, 255),
            1,
            cv2.LINE_AA,
        )

        # 5. Full Screen Warning Banner if Firing Simultaneously
        if self.fire_active:
            cv2.rectangle(annotated, (0, 0), (w, h), (200, 0, 255), 6)
            fire_label = f"🔥 SIMULTANEOUS FIRING - COLUMNS {self.firing_columns} 🔥"
            (fl_w, fl_h), _ = cv2.getTextSize(fire_label, font, 0.75, 2)
            cv2.rectangle(
                annotated,
                (w // 2 - fl_w // 2 - 10, h - 50),
                (w // 2 + fl_w // 2 + 10, h - 14),
                (200, 0, 255),
                -1,
            )
            cv2.putText(
                annotated,
                fire_label,
                (w // 2 - fl_w // 2, h - 22),
                font,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated
