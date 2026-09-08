"""
Crop and Weed Marker Detector for weed-rover.

Implements the physical/demo prototype vision model:
    ● = CROP (filled circular dots)
    X = WEED (X/cross geometric markers)

Distinguishes crop markers from weed markers geometrically and reports
structured detections with temporal persistence and spatial deduplication.

Only weed markers are passed to the robotic actuation/control layer.
Crop markers are detected and visualized for situational awareness but never targeted.
"""

from dataclasses import dataclass, field
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

# Ensure repository root is in sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from config.vision_config import (
        FARM_ROI,
        MARKER_LOWER_HSV,
        MARKER_UPPER_HSV,
        MARKER_MIN_AREA,
        MARKER_MAX_AREA,
        CROP_MIN_RADIUS,
        CROP_MAX_RADIUS,
        CROP_MIN_CIRCULARITY,
        CROP_MIN_SOLIDITY,
        CROP_MIN_CIRCLE_RATIO,
        WEED_X_MAX_ASPECT,
        WEED_X_MAX_SOLIDITY,
        WEED_X_MAX_CIRCULARITY,
        WEED_X_LINE_MIN_LEN_RATIO,
        CROP_CLASS_NAME,
        WEED_CLASS_NAME,
        STABILITY_MIN_HITS,
        MAX_MISSED_FRAMES,
        TARGET_MATCH_DISTANCE,
    )
except ImportError:
    FARM_ROI = None
    MARKER_LOWER_HSV = (0, 0, 0)
    MARKER_UPPER_HSV = (180, 255, 85)
    MARKER_MIN_AREA = 25.0
    MARKER_MAX_AREA = 8000.0
    CROP_MIN_RADIUS = 3.0
    CROP_MAX_RADIUS = 40.0
    CROP_MIN_CIRCULARITY = 0.70
    CROP_MIN_SOLIDITY = 0.85
    CROP_MIN_CIRCLE_RATIO = 0.75
    WEED_X_MAX_ASPECT = 1.85
    WEED_X_MAX_SOLIDITY = 0.72
    WEED_X_MAX_CIRCULARITY = 0.65
    WEED_X_LINE_MIN_LEN_RATIO = 0.25
    CROP_CLASS_NAME = "crop"
    WEED_CLASS_NAME = "weed"
    STABILITY_MIN_HITS = 2
    MAX_MISSED_FRAMES = 3
    TARGET_MATCH_DISTANCE = 30.0

try:
    from vision.robot_geometry import RobotGeometry
    from vision.control_interface import LEDMatrixMapper
except ImportError:
    try:
        from robot_geometry import RobotGeometry
        from control_interface import LEDMatrixMapper
    except ImportError:
        RobotGeometry = None
        LEDMatrixMapper = None


@dataclass
class MarkerDetection:
    """
    Structured detection output for Crop (dots) and Weed (X) markers.
    Compatible with downstream TargetController and WeedControlPipeline.
    """
    class_name: str           # 'crop' or 'weed'
    target_id: int            # Temporal tracking identifier
    center_x: int             # Centroid pixel X (u)
    center_y: int             # Centroid pixel Y (v)
    area: float               # Pixel area of marker
    confidence: float         # Shape classification score (0.0 to 1.0)
    box: List[int] = field(default_factory=lambda: [0, 0, 0, 0])  # [x1, y1, x2, y2]
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    is_stable: bool = False   # True after stability_min_hits consecutive frames
    hits: int = 1
    misses: int = 0

    @property
    def center_x_px(self) -> int:
        """Alias for center_x."""
        return self.center_x

    @property
    def center_y_px(self) -> int:
        """Alias for center_y."""
        return self.center_y

    @property
    def bbox(self) -> List[int]:
        """Alias for box."""
        return self.box

    def to_dict(self) -> Dict[str, Any]:
        """Converts detection to dictionary format."""
        return {
            "class_name": self.class_name,
            "target_id": self.target_id,
            "center_x_px": self.center_x,
            "center_y_px": self.center_y,
            "area": round(float(self.area), 2),
            "confidence": round(float(self.confidence), 4),
            "bbox": self.box,
            "is_stable": self.is_stable,
            "hits": self.hits,
            "misses": self.misses,
        }

    def to_debug_string(self) -> str:
        """Formatted string for logging."""
        status = "STABLE" if self.is_stable else "ACQ"
        return (
            f"[{self.class_name.upper()}] id={self.target_id} status={status} "
            f"conf={self.confidence:.2f} center=({self.center_x},{self.center_y}) "
            f"bbox=({self.x1},{self.y1},{self.x2},{self.y2}) area={self.area:.0f}"
        )

    def __str__(self) -> str:
        return self.to_debug_string()


class CropWeedDetector:
    """
    OpenCV geometric shape detector distinguishing:
      - CROP: filled circular dots (LED-sized)
      - WEED: X/cross markers (intersecting diagonals)
    """

    CLASS_CROP = "crop"
    CLASS_WEED = "weed"

    def __init__(
        self,
        lower_hsv: Tuple[int, int, int] = MARKER_LOWER_HSV,
        upper_hsv: Tuple[int, int, int] = MARKER_UPPER_HSV,
        min_area: float = MARKER_MIN_AREA,
        max_area: float = MARKER_MAX_AREA,
        roi: Optional[Tuple[int, int, int, int]] = FARM_ROI,
        crop_min_radius: float = CROP_MIN_RADIUS,
        crop_max_radius: float = CROP_MAX_RADIUS,
        crop_min_circularity: float = CROP_MIN_CIRCULARITY,
        crop_min_solidity: float = CROP_MIN_SOLIDITY,
        crop_min_circle_ratio: float = CROP_MIN_CIRCLE_RATIO,
        weed_max_aspect: float = WEED_X_MAX_ASPECT,
        weed_max_solidity: float = WEED_X_MAX_SOLIDITY,
        weed_max_circularity: float = WEED_X_MAX_CIRCULARITY,
        weed_min_line_len_ratio: float = WEED_X_LINE_MIN_LEN_RATIO,
        stability_min_hits: int = STABILITY_MIN_HITS,
        max_missed_frames: int = MAX_MISSED_FRAMES,
        match_distance: float = TARGET_MATCH_DISTANCE,
        merge_distance: float = 16.0,
        geometry: Optional[Any] = None,
        matrix_mapper: Optional[Any] = None,
    ):
        self.lower_hsv = lower_hsv
        self.upper_hsv = upper_hsv
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.roi = roi

        self.geometry = geometry if geometry is not None else (RobotGeometry() if RobotGeometry is not None else None)
        self.matrix_mapper = matrix_mapper if matrix_mapper is not None else (LEDMatrixMapper() if LEDMatrixMapper is not None else None)

        # Crop circle parameters
        self.crop_min_radius = float(crop_min_radius)
        self.crop_max_radius = float(crop_max_radius)
        self.crop_min_circularity = float(crop_min_circularity)
        self.crop_min_solidity = float(crop_min_solidity)
        self.crop_min_circle_ratio = float(crop_min_circle_ratio)

        # Weed X parameters
        self.weed_max_aspect = float(weed_max_aspect)
        self.weed_max_solidity = float(weed_max_solidity)
        self.weed_max_circularity = float(weed_max_circularity)
        self.weed_min_line_len_ratio = float(weed_min_line_len_ratio)

        # Temporal stability tracking parameters
        self.stability_min_hits = int(stability_min_hits)
        self.max_missed_frames = int(max_missed_frames)
        self.match_distance = float(match_distance)
        self.merge_distance = float(merge_distance)

        # Active tracking state per class
        self._next_id: int = 1
        self._active_targets: Dict[str, List[MarkerDetection]] = {
            self.CLASS_CROP: [],
            self.CLASS_WEED: [],
        }

    def reset(self) -> None:
        """Resets tracking history and next assigned ID."""
        self._next_id = 1
        self._active_targets = {
            self.CLASS_CROP: [],
            self.CLASS_WEED: [],
        }

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Creates a binary mask where markers are white (255) and floor is black (0).
        Supports grayscale / synthetic binary frames as well as color camera frames.
        """
        if len(frame.shape) == 2:
            # Grayscale / binary frame
            if frame.mean() > 127:
                # Dark markers on light background
                _, binary = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY_INV)
            else:
                # Light markers on dark background
                _, binary = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)
            return binary

        # BGR frame: HSV thresholding for dark markers
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.lower_hsv), np.array(self.upper_hsv))

        # Morphological opening removes speckle noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def _classify_crop(
        self,
        c: np.ndarray,
        area: float,
        w: int,
        h: int,
    ) -> Tuple[bool, float]:
        """
        Evaluates whether a candidate contour is a filled circular crop marker.

        Criteria:
        - Radius within crop_min_radius and crop_max_radius
        - Circularity 4*pi*area/perimeter^2 >= crop_min_circularity (>= 0.70)
        - Solidity area/hull_area >= crop_min_solidity (>= 0.85)
        - Aspect ratio max(w/h, h/w) <= 1.35
        - Minimum enclosing circle fill ratio area / (pi * r^2) >= crop_min_circle_ratio (>= 0.75)
        """
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            return False, 0.0

        circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return False, 0.0

        solidity = area / hull_area
        aspect = max(float(w) / h, float(h) / w)

        _, radius_enc = cv2.minEnclosingCircle(c)
        if not (self.crop_min_radius <= radius_enc <= self.crop_max_radius):
            return False, 0.0

        circle_area = math.pi * radius_enc * radius_enc
        circle_ratio = area / circle_area if circle_area > 0 else 0.0

        if (
            circularity >= self.crop_min_circularity
            and solidity >= self.crop_min_solidity
            and aspect <= 1.35
            and circle_ratio >= self.crop_min_circle_ratio
        ):
            conf = min(1.0, (circularity + solidity + circle_ratio) / 3.0)
            return True, round(conf, 4)

        return False, 0.0

    def _classify_weed_x(
        self,
        c: np.ndarray,
        area: float,
        w: int,
        h: int,
        binary_roi: np.ndarray,
    ) -> Tuple[bool, float]:
        """
        Evaluates whether a candidate contour is an X / cross weed marker.

        Geometric criteria:
        - Aspect ratio <= weed_max_aspect (roughly square bounding box)
        - Low solidity (0.10 <= solidity <= 0.72) due to 4 concavity bays between arms
        - Circularity < 0.65
        - Enclosing circle ratio < 0.65 (distinguishes from circles/disks)
        - Center intersection: foreground presence near the centroid
        - 4-corner presence: foreground pixels extending into all 4 outer corner quadrants
        - Diagonal line structures: evidence of opposing diagonal line orientations (approx 45 deg & -45 deg)
        """
        if w < 10 or h < 10:
            return False, 0.0

        aspect = max(float(w) / h, float(h) / w)
        if aspect > self.weed_max_aspect:
            return False, 0.0

        perimeter = cv2.arcLength(c, True)
        circularity = (4.0 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
        if circularity >= self.weed_max_circularity:
            return False, 0.0

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return False, 0.0

        solidity = area / hull_area
        if not (0.10 <= solidity <= self.weed_max_solidity):
            return False, 0.0

        _, radius_enc = cv2.minEnclosingCircle(c)
        circle_area = math.pi * radius_enc * radius_enc
        circle_ratio = area / circle_area if circle_area > 0 else 0.0
        if circle_ratio >= 0.68:
            return False, 0.0

        # 1. Check center intersection: central box of size w/6 x h/6 must have foreground
        cx, cy = w // 2, h // 2
        cw_box = max(2, w // 6)
        ch_box = max(2, h // 6)
        center_patch = binary_roi[
            max(0, cy - ch_box):min(h, cy + ch_box + 1),
            max(0, cx - cw_box):min(w, cx + cw_box + 1)
        ]
        if not np.any(center_patch > 0):
            return False, 0.0

        # 2. Check 4 corners: outer 25% of each corner quadrant must contain foreground
        w_c = max(2, w // 4)
        h_c = max(2, h // 4)
        has_tl = np.any(binary_roi[:h_c, :w_c] > 0)
        has_tr = np.any(binary_roi[:h_c, w - w_c:] > 0)
        has_bl = np.any(binary_roi[h - h_c:, :w_c] > 0)
        has_br = np.any(binary_roi[h - h_c:, w - w_c:] > 0)
        four_corners = has_tl and has_tr and has_bl and has_br

        # 3. Check diagonal line structures via Canny + HoughLinesP
        edges = cv2.Canny(binary_roi, 50, 150)
        min_line_len = max(6, int(min(w, h) * self.weed_min_line_len_ratio))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(6, int(min(w, h) * 0.15)),
            minLineLength=min_line_len,
            maxLineGap=max(3, int(min(w, h) * 0.1)),
        )

        has_pos_diag = False
        has_neg_diag = False

        if lines is not None:
            for line in lines:
                pts = line.flatten()
                x1, y1, x2, y2 = pts[0], pts[1], pts[2], pts[3]
                angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180

                # Opposing diagonal ranges: ~45 deg and ~ -45 deg
                if 18.0 <= angle <= 72.0:
                    has_pos_diag = True
                elif -72.0 <= angle <= -18.0:
                    has_neg_diag = True

        # Validation rule: must have 4 corners and opposing diagonal evidence
        # (or both diagonals confirmed if 3+ corners present)
        corner_count = sum([has_tl, has_tr, has_bl, has_br])
        has_opposing_diagonals = has_pos_diag and has_neg_diag

        if (four_corners and (has_pos_diag or has_neg_diag)) or (corner_count >= 3 and has_opposing_diagonals):
            score = 1.0 - (abs(aspect - 1.0) * 0.25)
            if has_opposing_diagonals:
                score = min(1.0, score + 0.1)
            return True, round(score, 4)

        return False, 0.0

    def detect(self, frame: np.ndarray) -> Tuple[List[MarkerDetection], np.ndarray]:
        """
        Runs complete Crop and Weed marker detection on the input frame.

        Args:
            frame: Input image (BGR or grayscale).

        Returns:
            Tuple[List[MarkerDetection], np.ndarray]: List of detected markers, binary mask.
        """
        if frame is None or frame.size == 0:
            return [], np.zeros((1, 1), dtype=np.uint8)

        # Step 1: Preprocess to binary mask
        binary_mask = self._preprocess_frame(frame)

        # Step 2: Apply Region of Interest (ROI) if configured
        h, w = binary_mask.shape[:2]
        proc_mask = binary_mask.copy()
        if self.roi is not None:
            rx1, ry1, rx2, ry2 = self.roi
            rx1 = max(0, min(w, rx1))
            ry1 = max(0, min(h, ry1))
            rx2 = max(0, min(w, rx2))
            ry2 = max(0, min(h, ry2))

            # Mask out everything outside ROI
            roi_stencil = np.zeros_like(proc_mask)
            roi_stencil[ry1:ry2, rx1:rx2] = 255
            proc_mask = cv2.bitwise_and(proc_mask, roi_stencil)

        # Step 3: Find external contours
        contours, _ = cv2.findContours(proc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_crops: List[MarkerDetection] = []
        raw_weeds: List[MarkerDetection] = []

        for c in contours:
            area = cv2.contourArea(c)
            if not (self.min_area <= area <= self.max_area):
                continue

            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < 4 or bh < 4:
                continue

            # Crop binary patch for detailed geometric shape analysis
            roi_patch = proc_mask[by:by + bh, bx:bx + bw]

            # Calculate centroid via moments
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = int(bx + (bw / 2))
                cy = int(by + (bh / 2))

            # Evaluate Crop circle first
            is_crop, crop_conf = self._classify_crop(c, area, bw, bh)
            if is_crop:
                raw_crops.append(MarkerDetection(
                    class_name=self.CLASS_CROP,
                    target_id=0,
                    center_x=cx,
                    center_y=cy,
                    area=area,
                    confidence=crop_conf,
                    box=[bx, by, bx + bw, by + bh],
                    x1=bx,
                    y1=by,
                    x2=bx + bw,
                    y2=by + bh,
                ))
                continue

            # Evaluate Weed X marker
            is_weed, weed_conf = self._classify_weed_x(c, area, bw, bh, roi_patch)
            if is_weed:
                raw_weeds.append(MarkerDetection(
                    class_name=self.CLASS_WEED,
                    target_id=0,
                    center_x=cx,
                    center_y=cy,
                    area=area,
                    confidence=weed_conf,
                    box=[bx, by, bx + bw, by + bh],
                    x1=bx,
                    y1=by,
                    x2=bx + bw,
                    y2=by + bh,
                ))

        # Step 4: Deduplicate per class
        dedup_crops = self._deduplicate(raw_crops)
        dedup_weeds = self._deduplicate(raw_weeds)

        # Step 5: Temporal tracking & stability update
        tracked_crops = self._update_temporal_tracks(self.CLASS_CROP, dedup_crops)
        tracked_weeds = self._update_temporal_tracks(self.CLASS_WEED, dedup_weeds)

        all_detections = tracked_crops + tracked_weeds
        return all_detections, binary_mask

    def _deduplicate(self, candidates: List[MarkerDetection]) -> List[MarkerDetection]:
        """Suppresses nearby duplicate detections within merge_distance."""
        if len(candidates) <= 1:
            return candidates

        candidates.sort(key=lambda d: -d.area)
        kept: List[MarkerDetection] = []

        for cand in candidates:
            duplicate = False
            for existing in kept:
                dist = math.hypot(cand.center_x - existing.center_x, cand.center_y - existing.center_y)
                if dist < self.merge_distance:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(cand)

        return kept

    def _update_temporal_tracks(
        self,
        class_name: str,
        current_detections: List[MarkerDetection],
    ) -> List[MarkerDetection]:
        """
        Updates frame-to-frame tracking for candidate targets of a given class.
        Assigns persistent IDs and stability flags.
        """
        active = self._active_targets[class_name]
        matched_active_indices = set()
        matched_current_indices = set()
        result: List[MarkerDetection] = []

        # Match current detections with active tracks
        for cur_idx, cur_det in enumerate(current_detections):
            best_act_idx = None
            best_dist = self.match_distance

            for act_idx, act_det in enumerate(active):
                if act_idx in matched_active_indices:
                    continue
                dist = math.hypot(cur_det.center_x - act_det.center_x, cur_det.center_y - act_det.center_y)
                if dist < best_dist:
                    best_dist = dist
                    best_act_idx = act_idx

            if best_act_idx is not None:
                matched_active_indices.add(best_act_idx)
                matched_current_indices.add(cur_idx)

                prev = active[best_act_idx]
                cur_det.target_id = prev.target_id
                cur_det.hits = prev.hits + 1
                cur_det.misses = 0
                cur_det.is_stable = cur_det.hits >= self.stability_min_hits
                result.append(cur_det)

        # New unassigned detections get fresh IDs
        for cur_idx, cur_det in enumerate(current_detections):
            if cur_idx not in matched_current_indices:
                cur_det.target_id = self._next_id
                self._next_id += 1
                cur_det.hits = 1
                cur_det.misses = 0
                cur_det.is_stable = cur_det.hits >= self.stability_min_hits
                result.append(cur_det)

        # Update missed active tracks
        updated_active = [d for d in result]
        for act_idx, act_det in enumerate(active):
            if act_idx not in matched_active_indices:
                act_det.misses += 1
                if act_det.misses <= self.max_missed_frames:
                    updated_active.append(act_det)

        self._active_targets[class_name] = updated_active
        return result

    def draw_visual_debug(
        self,
        frame: np.ndarray,
        detections: List[MarkerDetection],
        selected_weed: Optional[MarkerDetection] = None,
        selected_x_cm: Optional[float] = None,
        selected_y_cm: Optional[float] = None,
        selected_column: Optional[int] = None,
        control_payload: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """
        Renders the visual debug overlay specified in Requirement 10:
          - CROP: circular green marker + "CROP"
          - WEED: X marker in magenta/red + "WEED"
          - Selected Weed: highlighted with X_cm, Y_cm, COLUMN and HUD banner.

        Args:
            frame: Input BGR image.
            detections: List of MarkerDetection instances.
            selected_weed: The chosen weed targeted by control.
            selected_x_cm: Ground X in cm of chosen weed.
            selected_y_cm: Ground Y in cm of chosen weed.
            selected_column: Mapped LED column index (0..7).
            control_payload: {"weed_detected": 0 or 1, "column": ...}.

        Returns:
            np.ndarray: Annotated BGR frame.
        """
        out = frame.copy()

        # 1. Draw ROI boundary if present
        if self.roi is not None:
            rx1, ry1, rx2, ry2 = self.roi
            cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (60, 60, 60), 1)

        # 2. Draw detections
        for det in detections:
            cx, cy = det.center_x, det.center_y
            is_selected = selected_weed is not None and det.target_id == selected_weed.target_id

            if det.class_name == self.CLASS_CROP:
                # Green circle for crops
                radius = max(6, int(math.sqrt(det.area / math.pi)))
                cv2.circle(out, (cx, cy), radius, (0, 230, 0), 2)
                cv2.circle(out, (cx, cy), 2, (0, 255, 0), -1)
                cv2.putText(
                    out,
                    "CROP",
                    (cx - 18, cy - radius - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 230, 0),
                    1,
                    cv2.LINE_AA,
                )

            elif det.class_name == self.CLASS_WEED:
                # Purple/Magenta X for weeds
                w_half = max(10, (det.x2 - det.x1) // 2)
                h_half = max(10, (det.y2 - det.y1) // 2)

                color = (0, 0, 255) if is_selected else (255, 0, 255)
                thickness = 3 if is_selected else 2

                # Draw prominent X marker
                cv2.line(out, (cx - w_half, cy - h_half), (cx + w_half, cy + h_half), color, thickness)
                cv2.line(out, (cx - w_half, cy + h_half), (cx + w_half, cy - h_half), color, thickness)
                cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 1)

                # Compute ground coords and LED column for every detected weed
                det_x_led, det_y_led, det_col = None, None, None
                if is_selected and selected_x_cm is not None and selected_column is not None:
                    det_x_led = selected_x_cm
                    det_y_led = selected_y_cm
                    det_col = selected_column
                elif self.geometry and self.matrix_mapper:
                    pt = self.geometry.pixel_to_ground(cx, cy)
                    if pt is not None:
                        led_pt = self.geometry.ground_to_led_frame(pt[0], pt[1])
                        det_x_led, det_y_led = led_pt[0], led_pt[1]
                        det_col = self.matrix_mapper.x_to_column(det_x_led)

                # Target reticle corner brackets for every detected weed
                reticle_color = (0, 255, 255) if is_selected else (0, 200, 255)
                r_size = max(w_half, h_half) + 4
                cv2.line(out, (cx - r_size, cy - r_size), (cx - r_size + 6, cy - r_size), reticle_color, 2)
                cv2.line(out, (cx - r_size, cy - r_size), (cx - r_size, cy - r_size + 6), reticle_color, 2)
                cv2.line(out, (cx + r_size, cy - r_size), (cx + r_size - 6, cy - r_size), reticle_color, 2)
                cv2.line(out, (cx + r_size, cy - r_size), (cx + r_size, cy - r_size + 6), reticle_color, 2)
                cv2.line(out, (cx - r_size, cy + r_size), (cx - r_size + 6, cy + r_size), reticle_color, 2)
                cv2.line(out, (cx - r_size, cy + r_size), (cx - r_size, cy + r_size - 6), reticle_color, 2)
                cv2.line(out, (cx + r_size, cy + r_size), (cx + r_size - 6, cy + r_size), reticle_color, 2)
                cv2.line(out, (cx + r_size, cy + r_size), (cx + r_size, cy + r_size - 6), reticle_color, 2)

                col_label = f"Col: {det_col}" if det_col is not None else ""
                weed_label = f"WEED  {col_label}".strip()

                cv2.putText(
                    out,
                    weed_label,
                    (max(5, det.x1 - 10), max(16, det.y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    reticle_color,
                    1,
                    cv2.LINE_AA,
                )

                if det_x_led is not None and det_y_led is not None:
                    coord_str = f"({det_x_led:+.1f}, {det_y_led:+.1f})cm"
                    cv2.putText(
                        out,
                        coord_str,
                        (max(5, det.x1 - 10), min(out.shape[0] - 8, det.y2 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        # 3. Draw detailed overlay for the active weed
        if selected_weed is not None:
            scx, scy = selected_weed.center_x, selected_weed.center_y
            # Target reticle circle
            cv2.circle(out, (scx, scy), 20, (0, 255, 255), 2)

            info_lines = ["WEED"]
            if selected_x_cm is not None:
                info_lines.append(f"X={selected_x_cm:+.2f} cm")
            if selected_y_cm is not None:
                info_lines.append(f"Y={selected_y_cm:.1f} cm")
            if selected_column is not None:
                info_lines.append(f"Column={selected_column}")

            text_y = max(20, selected_weed.y1 - 8 - (len(info_lines) * 14))
            for i, line in enumerate(info_lines):
                cv2.putText(
                    out,
                    line,
                    (selected_weed.x1, text_y + (i * 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 4. Top HUD Banner displaying exact control payload
        hud_bg = np.zeros((36, out.shape[1], 3), dtype=np.uint8)
        hud_bg[:] = (20, 20, 20)
        out[0:36, 0:out.shape[1]] = cv2.addWeighted(out[0:36, 0:out.shape[1]], 0.3, hud_bg, 0.7, 0)

        weed_flag = 1 if (control_payload and control_payload.get("weed_detected", 0) == 1) else 0
        cols_list = control_payload.get("columns", []) if control_payload else []
        if not cols_list and control_payload and control_payload.get("column", -1) >= 0:
            cols_list = [control_payload["column"]]

        if len(cols_list) > 1:
            status_text = f"Weeds: {len(cols_list)}  |  Columns: {cols_list}"
        else:
            col_val = cols_list[0] if cols_list else -1
            status_text = f"Weed: {weed_flag}  |  Column: {col_val}"
        color_status = (0, 255, 255) if weed_flag == 1 else (180, 180, 180)

        cv2.putText(
            out,
            status_text,
            (15, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color_status,
            2,
            cv2.LINE_AA,
        )

        crops_count = sum(1 for d in detections if d.class_name == self.CLASS_CROP)
        weeds_count = sum(1 for d in detections if d.class_name == self.CLASS_WEED)
        counts_text = f"Crops: {crops_count}  Weeds: {weeds_count}"
        cv2.putText(
            out,
            counts_text,
            (out.shape[1] - 220, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return out

