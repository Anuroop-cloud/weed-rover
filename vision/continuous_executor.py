"""
Continuous-Forward Weed Execution Engine for weed-rover.

Implements the continuous-forward physical execution model:
- The rover continuously moves forward (never turns toward individual weeds).
- No target locking, no persistent target IDs, no approach/stop/fire state machine.
- Frame-based crop and weed detection.
- Crops (●) are ignored for control; only weeds (X) are processed.
- Physical X position mapped to physical LED matrix column (0..7).
- When a weed reaches/crosses the spatial FIRING BOUNDARY in the camera image:
    - Exactly ONE firing event is scheduled per physical weed (temporal debounce).
    - LED firing occurs after the configured forward travel delay.
    - Rover keeps moving throughout the entire process.
- Multiple weeds progress independently toward firing across different columns.
- Control output contract:
    weed_detected: 0 or 1
    column: -1 or 0..7
"""

from dataclasses import dataclass, field
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
    from config.vision_config import (
        CAMERA_BLIND_SPOT_ROW_PX,
        CAMERA_TO_LED_FORWARD_CM,
        SIMULATED_APPROACH_SPEED_CM_S,
        LED_FIRE_DURATION_SEC,
    )
    from vision.robot_geometry import RobotGeometry
    from vision.control_interface import LEDMatrixMapper, x_to_matrix_column
except ImportError:
    CAMERA_BLIND_SPOT_ROW_PX = 430
    CAMERA_TO_LED_FORWARD_CM = 5.0
    SIMULATED_APPROACH_SPEED_CM_S = 8.0
    LED_FIRE_DURATION_SEC = 0.5
    from robot_geometry import RobotGeometry
    try:
        from control_interface import LEDMatrixMapper, x_to_matrix_column
    except ImportError:
        class LEDMatrixMapper:
            def x_to_column(self, x_cm: float) -> int:
                col = int(math.floor(round((float(x_cm) + 1.6) / 0.4, 6)))
                return max(0, min(7, col))
        def x_to_matrix_column(x_cm: float) -> int:
            return LEDMatrixMapper().x_to_column(x_cm)


@dataclass
class WeedTrack:
    """Internal spatial-temporal track for a weed to prevent repeated firing."""
    track_id: int               # Internal tracking index (never shown on HUD)
    center_x: float
    center_y: float
    ground_x_cm: float
    ground_y_cm: float
    column: int                 # Physical LED matrix column (0..7)
    last_seen_time: float
    first_seen_time: float
    has_fired: bool = False     # Set to True once firing event is scheduled
    crossed_boundary: bool = False


@dataclass
class FiringEvent:
    """Represents a scheduled or active LED column firing event."""
    column: int
    scheduled_fire_time: float  # Epoch timestamp when LED turns on
    fire_end_time: float        # Epoch timestamp when LED turns off
    trigger_pixel_y: float
    created_time: float

    @property
    def is_active(self) -> bool:
        """True if current time falls within firing window."""
        now = time.time()
        return self.scheduled_fire_time <= now <= self.fire_end_time


@dataclass
class ContinuousStatus:
    """Control output snapshot for the continuous-forward prototype."""
    weed_detected: int                  # 0 = no weed, 1 = weed detected
    column: int                         # -1 = none, 0..7 = active/target column
    is_firing: bool                     # True if any LED column is actively firing
    firing_columns: List[int]           # All columns actively firing right now
    scheduled_columns: List[int]        # Columns scheduled to fire after travel delay
    active_weed_columns: List[int]      # Columns of all visible weeds in frame
    active_weeds_count: int             # Number of weeds currently visible
    crops_count: int                    # Number of crops currently visible
    closest_weed_coords: Optional[Tuple[float, float]] = None  # (X_cm, Y_cm)

    def to_control_payload(self) -> Dict[str, int]:
        """Strict control output contract: weed_detected (0/1) and column (0..7 or -1)."""
        return {
            "weed_detected": int(self.weed_detected),
            "column": int(self.column),
        }


class ContinuousWeedExecutor:
    """
    Manages continuous-forward weed perception and column firing for Mode B.

    Eliminates target locking, persistent target IDs, and stop-to-fire cycles.
    """

    def __init__(
        self,
        geometry: Optional[RobotGeometry] = None,
        matrix_mapper: Optional[LEDMatrixMapper] = None,
        firing_boundary_row_px: int = CAMERA_BLIND_SPOT_ROW_PX,
        rover_forward_speed_cm_s: float = SIMULATED_APPROACH_SPEED_CM_S,
        camera_to_led_forward_cm: float = CAMERA_TO_LED_FORWARD_CM,
        firing_delay_sec: Optional[float] = None,
        fire_duration_sec: float = LED_FIRE_DURATION_SEC,
        debounce_cooldown_sec: float = 1.2,
        match_dist_x_px: float = 45.0,
        match_dist_y_px: float = 90.0,
    ):
        """
        Args:
            geometry: RobotGeometry instance for pixel-to-ground translation.
            matrix_mapper: LEDMatrixMapper instance for lateral X to column 0..7 mapping.
            firing_boundary_row_px: Pixel row Y representing the firing boundary.
            rover_forward_speed_cm_s: Continuous rover forward speed in cm/s.
            camera_to_led_forward_cm: Physical forward separation between camera and LED array.
            firing_delay_sec: Delay between crossing boundary and firing. Defaults to distance / speed.
            fire_duration_sec: How long LED column remains active during firing event.
            debounce_cooldown_sec: Time after crossing boundary before same column/position can re-fire.
            match_dist_x_px: Max lateral pixel difference to match weed between frames.
            match_dist_y_px: Max vertical pixel advance to match weed between frames.
        """
        self.geometry = geometry if geometry is not None else RobotGeometry()
        self.matrix_mapper = matrix_mapper if matrix_mapper is not None else LEDMatrixMapper()
        self.firing_boundary_row_px = int(firing_boundary_row_px)
        self.rover_forward_speed_cm_s = float(rover_forward_speed_cm_s)
        self.camera_to_led_forward_cm = float(camera_to_led_forward_cm)

        # Configurable physical travel delay
        if firing_delay_sec is not None:
            self.firing_delay_sec = float(firing_delay_sec)
        else:
            speed = max(0.1, self.rover_forward_speed_cm_s)
            self.firing_delay_sec = round(self.camera_to_led_forward_cm / speed, 3)

        self.fire_duration_sec = float(fire_duration_sec)
        self.debounce_cooldown_sec = float(debounce_cooldown_sec)
        self.match_dist_x_px = float(match_dist_x_px)
        self.match_dist_y_px = float(match_dist_y_px)

        # Internal spatial tracking & event queue
        self._next_internal_id: int = 1
        self._tracks: List[WeedTrack] = []
        self._firing_events: List[FiringEvent] = []
        self._fired_history: List[Tuple[float, int, float]] = []  # (timestamp, column, x_px)

    def reset(self) -> None:
        """Resets all internal tracks and scheduled events."""
        self._next_internal_id = 1
        self._tracks.clear()
        self._firing_events.clear()
        self._fired_history.clear()

    def update(
        self,
        detections: List[Any],
        delta_time: float = 0.033,
        current_time: Optional[float] = None,
    ) -> ContinuousStatus:
        """
        Processes current frame detections, associates weeds across frames,
        schedules firing events when weeds cross the firing boundary, and returns
        control output.

        Args:
            detections: List of detection objects (must have class_name, center_x, center_y).
            delta_time: Time elapsed since last frame.
            current_time: Epoch timestamp (defaults to time.time()).

        Returns:
            ContinuousStatus containing weed_detected (0/1), column (0..7 or -1), and firing states.
        """
        now = current_time if current_time is not None else time.time()

        # Clean expired fired history (for cooldown debounce)
        self._fired_history = [
            (t, col, x) for (t, col, x) in self._fired_history
            if (now - t) < self.debounce_cooldown_sec
        ]

        # 1. Filter detections: CROPS ARE IGNORED FOR CONTROL
        weed_dets = [d for d in detections if getattr(d, "class_name", "") == "weed"]
        crops_count = sum(1 for d in detections if getattr(d, "class_name", "") == "crop")

        matched_track_indices = set()
        current_frame_tracks: List[WeedTrack] = []

        # 2. Process each detected weed in current frame
        for det in weed_dets:
            cx = float(getattr(det, "center_x", 0))
            cy = float(getattr(det, "center_y", 0))

            # Ground coordinates & LED column
            g_cam = self.geometry.pixel_to_ground(cx, cy)
            if g_cam is not None:
                gx, gy = g_cam
                g_led = self.geometry.ground_to_led_frame(gx, gy)
                col = self.matrix_mapper.x_to_column(g_led[0])
            else:
                gx, gy = 0.0, 0.0
                col = 3

            # Associate with existing active track (spatial proximity)
            best_idx = None
            best_dist = float("inf")

            for idx, tr in enumerate(self._tracks):
                if idx in matched_track_indices:
                    continue
                dx = abs(cx - tr.center_x)
                dy = cy - tr.center_y
                # Rover moves forward: weeds move downward (dy >= -15 px for minor jitter)
                if dx <= self.match_dist_x_px and -15.0 <= dy <= self.match_dist_y_px:
                    dist = math.hypot(dx, dy * 0.5)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx

            if best_idx is not None:
                matched_track_indices.add(best_idx)
                tr = self._tracks[best_idx]
                tr.center_x = cx
                tr.center_y = cy
                tr.ground_x_cm = gx
                tr.ground_y_cm = gy
                tr.column = col
                tr.last_seen_time = now

                # Check if weed reached / crossed the firing boundary
                if not tr.has_fired and cy >= self.firing_boundary_row_px:
                    self._schedule_fire(tr, now)

                current_frame_tracks.append(tr)

            else:
                # New physical weed detected in frame
                is_past_boundary = (cy >= self.firing_boundary_row_px)
                # Check cooldown to prevent re-triggering just-fired weeds near boundary
                in_cooldown = any(
                    col == c and abs(cx - px) < self.match_dist_x_px
                    for (_, c, px) in self._fired_history
                )

                new_track = WeedTrack(
                    track_id=self._next_internal_id,
                    center_x=cx,
                    center_y=cy,
                    ground_x_cm=gx,
                    ground_y_cm=gy,
                    column=col,
                    last_seen_time=now,
                    first_seen_time=now,
                    has_fired=is_past_boundary or in_cooldown,
                    crossed_boundary=is_past_boundary,
                )
                self._next_internal_id += 1

                if is_past_boundary and not in_cooldown:
                    self._schedule_fire(new_track, now)

                self._tracks.append(new_track)
                current_frame_tracks.append(new_track)

        # 3. Check tracks that disappeared off bottom border (blind spot exit)
        for idx, tr in enumerate(self._tracks):
            if idx not in matched_track_indices:
                # If it was close to the boundary when last seen, it crossed off-camera into blind zone
                if (
                    not tr.has_fired
                    and tr.center_y >= (self.firing_boundary_row_px - 40.0)
                    and (now - tr.last_seen_time) < 0.3
                ):
                    self._schedule_fire(tr, now)

        # 4. Clean up tracks that have expired or finished firing
        self._tracks = [
            tr for tr in self._tracks
            if (now - tr.last_seen_time) < (self.debounce_cooldown_sec + 0.5)
        ]

        # 5. Evaluate active and scheduled firing events
        active_fires: List[FiringEvent] = []
        scheduled_fires: List[FiringEvent] = []
        remaining_events: List[FiringEvent] = []

        for event in self._firing_events:
            if event.scheduled_fire_time <= now <= event.fire_end_time:
                active_fires.append(event)
                remaining_events.append(event)
            elif now < event.scheduled_fire_time:
                scheduled_fires.append(event)
                remaining_events.append(event)
            # Events past fire_end_time are discarded

        self._firing_events = remaining_events

        firing_columns = sorted(list(set(e.column for e in active_fires)))
        scheduled_columns = sorted(list(set(e.column for e in scheduled_fires)))
        active_weed_columns = sorted(list(set(tr.column for tr in current_frame_tracks)))

        # 6. Build Control Output (weed_detected = 0/1, column = 0..7 or -1)
        is_firing = len(firing_columns) > 0
        closest_coords = None

        if is_firing:
            # Active firing takes highest precedence for control actuation
            weed_detected = 1
            control_column = firing_columns[0]
        elif len(current_frame_tracks) > 0:
            # Report the weed closest to the firing boundary (largest Y)
            weed_detected = 1
            lead_track = max(current_frame_tracks, key=lambda t: t.center_y)
            control_column = lead_track.column
            closest_coords = (lead_track.ground_x_cm, lead_track.ground_y_cm)
        elif len(scheduled_columns) > 0:
            # Weed crossed boundary and is traveling under rover toward LED
            weed_detected = 1
            control_column = scheduled_columns[0]
        else:
            # No weed detected
            weed_detected = 0
            control_column = -1

        return ContinuousStatus(
            weed_detected=weed_detected,
            column=control_column,
            is_firing=is_firing,
            firing_columns=firing_columns,
            scheduled_columns=scheduled_columns,
            active_weed_columns=active_weed_columns,
            active_weeds_count=len(weed_dets),
            crops_count=crops_count,
            closest_weed_coords=closest_coords,
        )

    def _schedule_fire(self, track: WeedTrack, now: float) -> None:
        """Schedules a single firing event after the configured physical delay."""
        track.has_fired = True
        track.crossed_boundary = True

        fire_start = now + self.firing_delay_sec
        fire_end = fire_start + self.fire_duration_sec

        event = FiringEvent(
            column=track.column,
            scheduled_fire_time=fire_start,
            fire_end_time=fire_end,
            trigger_pixel_y=track.center_y,
            created_time=now,
        )
        self._firing_events.append(event)
        self._fired_history.append((now, track.column, track.center_x))

    def draw_hud(
        self,
        frame: np.ndarray,
        detections: List[Any],
        fps: float = 30.0,
        status: Optional[ContinuousStatus] = None,
        serial_status: Optional[str] = None,
        last_command: Optional[str] = None,
    ) -> np.ndarray:
        """
        Renders the clean, modern HUD for the continuous-forward prototype:
        - ● CROP in green
        - X WEED in magenta with Column label (NO TARGET IDs)
        - FIRING BOUNDARY line
        - FIRING -> COLUMN X alert when active
        - Clean top banner with optional serial status and last command
        """
        out = frame.copy()
        h, w = out.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 1. Draw Detections
        weed_dets = [d for d in detections if getattr(d, "class_name", "") == "weed"]
        crop_dets = [d for d in detections if getattr(d, "class_name", "") == "crop"]

        # Draw crops (●)
        for d in crop_dets:
            cx, cy = int(getattr(d, "center_x", 0)), int(getattr(d, "center_y", 0))
            area = float(getattr(d, "area", 50))
            radius = max(6, int(math.sqrt(area / math.pi)))
            cv2.circle(out, (cx, cy), radius, (0, 230, 0), 2)
            cv2.circle(out, (cx, cy), 2, (0, 255, 0), -1)
            cv2.putText(out, "CROP", (cx - 18, cy - radius - 6), font, 0.45, (0, 230, 0), 1, cv2.LINE_AA)

        # Identify lead weed (closest to boundary / largest Y)
        lead_weed = max(weed_dets, key=lambda d: getattr(d, "center_y", 0)) if weed_dets else None

        # Draw weeds (X)
        for d in weed_dets:
            cx, cy = int(getattr(d, "center_x", 0)), int(getattr(d, "center_y", 0))
            box = getattr(d, "box", None)
            if box is not None and len(box) == 4:
                bx1, by1, bx2, by2 = [int(v) for v in box]
                w_half = max(10, (bx2 - bx1) // 2)
                h_half = max(10, (by2 - by1) // 2)
            else:
                bx1, by1, bx2, by2 = cx - 14, cy - 14, cx + 14, cy + 14
                w_half, h_half = 14, 14

            is_lead = (lead_weed is not None and d is lead_weed)
            x_color = (0, 220, 255) if is_lead else (255, 0, 255)
            thickness = 3 if is_lead else 2

            # Prominent X marker
            cv2.line(out, (cx - w_half, cy - h_half), (cx + w_half, cy + h_half), x_color, thickness)
            cv2.line(out, (cx - w_half, cy + h_half), (cx + w_half, cy - h_half), x_color, thickness)
            cv2.rectangle(out, (bx1, by1), (bx2, by2), x_color, 1)

            # Determine Column
            g_cam = self.geometry.pixel_to_ground(cx, cy)
            if g_cam is not None:
                gx, gy = g_cam
                g_led = self.geometry.ground_to_led_frame(gx, gy)
                col = self.matrix_mapper.x_to_column(g_led[0])
                coord_str = f"({gx:+.1f}, {gy:.1f})cm"
            else:
                col = 3
                coord_str = ""

            # Clean label (NO TARGET IDs!)
            tag_str = f"WEED  Col: {col}" + (" [LEAD]" if is_lead else "")
            cv2.putText(
                out,
                tag_str,
                (max(5, bx1 - 8), max(16, by1 - 6)),
                font,
                0.46,
                x_color,
                1,
                cv2.LINE_AA,
            )

            if coord_str:
                cv2.putText(
                    out,
                    coord_str,
                    (max(5, bx1 - 8), min(h - 8, by2 + 14)),
                    font,
                    0.38,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 2. Firing Boundary Line
        cv2.line(
            out,
            (0, self.firing_boundary_row_px),
            (w, self.firing_boundary_row_px),
            (0, 140, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            "FIRING BOUNDARY",
            (10, self.firing_boundary_row_px - 8),
            font,
            0.45,
            (0, 140, 255),
            1,
            cv2.LINE_AA,
        )

        # 3. Top HUD Banner
        hud_bg = np.zeros((36, w, 3), dtype=np.uint8)
        hud_bg[:] = (20, 20, 20)
        out[0:36, 0:w] = cv2.addWeighted(out[0:36, 0:w], 0.3, hud_bg, 0.7, 0)

        # Build clean banner text
        cur_col = status.column if status else (lead_weed.column if lead_weed and hasattr(lead_weed, 'column') else -1)
        cur_weed = status.weed_detected if status else (1 if weed_dets else 0)

        banner_items = [
            f"FPS: {fps:.0f}",
            f"Crops: {len(crop_dets)}",
            f"Weeds: {len(weed_dets)}",
            f"Column: {cur_col if cur_col >= 0 else '-'}",
        ]
        if serial_status:
            banner_items.append(f"ESP32: {serial_status}")
        if last_command:
            banner_items.append(f"LAST CMD: {last_command}")

        hud_text = "  |  ".join(banner_items)
        color_hud = (0, 255, 255) if cur_weed == 1 else (180, 180, 180)
        cv2.putText(out, hud_text, (12, 24), font, 0.48, color_hud, 1, cv2.LINE_AA)

        # 4. Firing Status Alert
        if status and status.is_firing:
            # Active firing banner & border
            cv2.rectangle(out, (0, 0), (w, h), (200, 0, 255), 5)
            cols_str = str(status.firing_columns) if len(status.firing_columns) > 1 else str(status.firing_columns[0])
            alert_text = f"🔥 FIRING -> COLUMN {cols_str}"
            (tw, th), _ = cv2.getTextSize(alert_text, font, 0.75, 2)
            cv2.rectangle(out, (w // 2 - tw // 2 - 12, h - 52), (w // 2 + tw // 2 + 12, h - 14), (200, 0, 255), -1)
            cv2.putText(out, alert_text, (w // 2 - tw // 2, h - 24), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        elif status and status.scheduled_columns:
            # Scheduled firing countdown badge
            sched_str = str(status.scheduled_columns) if len(status.scheduled_columns) > 1 else str(status.scheduled_columns[0])
            sched_text = f"SCHEDULED -> COLUMN {sched_str}"
            (tw, th), _ = cv2.getTextSize(sched_text, font, 0.50, 1)
            cv2.rectangle(out, (w - tw - 24, 44), (w - 8, 44 + th + 12), (0, 140, 255), -1)
            cv2.putText(out, sched_text, (w - tw - 16, 44 + th + 6), font, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        return out
