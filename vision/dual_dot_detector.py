"""
Dual Dot Detector for weed-rover.
Simultaneously detects both black dots and blue dots on a light-colored mock farm or grid surface.

Includes:
- Dual-channel HSV thresholding (independent masks for black and blue)
- Region of Interest (ROI) spatial masking
- Noise, aspect-ratio, and proximity-based contour deduplication per color class
- Independent temporal stability tracking per color class (e.g. black T1 vs blue T1)
- Color-coded visualization (Red/White boxes for Black Targets, Bright Blue boxes for Blue Targets)
- Separate low-level DotDetection and high-level Target abstractions with class_name
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class DotDetection:
    """
    Low-level raw detection from a single image frame.
    Represents an instantaneous snapshot of a detected dark or blue contour.
    """
    class_name: str  # 'black_dot' or 'blue_dot'
    center_x: int
    center_y: int
    area: float
    box: List[int] = field(default_factory=lambda: [0, 0, 0, 0])  # [x1, y1, x2, y2]
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    contour: Optional[np.ndarray] = None

    def to_debug_string(self) -> str:
        """Format raw dot detection for terminal output."""
        return (
            f"class={self.class_name} center=({self.center_x},{self.center_y}) "
            f"bbox=({self.x1},{self.y1},{self.x2},{self.y2}) "
            f"area={self.area:.1f}"
        )

    def __str__(self) -> str:
        return self.to_debug_string()

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection data to dictionary format."""
        return {
            "class_name": self.class_name,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "area": round(float(self.area), 2),
            "box": self.box,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }


@dataclass
class Target:
    """
    High-level, temporally stable target abstraction.
    Maintains persistent identity, filtered coordinates, and stability status across frames
    independently for each color class.
    """
    class_name: str  # 'black_dot' or 'blue_dot'
    target_id: int
    center_x: int
    center_y: int
    area: float
    box: List[int] = field(default_factory=lambda: [0, 0, 0, 0])  # [x1, y1, x2, y2]
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    is_stable: bool = False
    hits: int = 1
    misses: int = 0

    def to_debug_string(self) -> str:
        """Format target with temporal status for terminal output."""
        status = "STABLE" if self.is_stable else "ACQUIRING"
        return (
            f"target_id={self.target_id} class={self.class_name} status={status} hits={self.hits} "
            f"center=({self.center_x},{self.center_y}) "
            f"bbox=({self.x1},{self.y1},{self.x2},{self.y2}) area={self.area:.1f}"
        )

    def __str__(self) -> str:
        return self.to_debug_string()

    def to_dict(self) -> Dict[str, Any]:
        """Convert target to dictionary format."""
        return {
            "class_name": self.class_name,
            "target_id": self.target_id,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "area": round(float(self.area), 2),
            "box": self.box,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "is_stable": self.is_stable,
            "hits": self.hits,
            "misses": self.misses,
        }


class DualDotDetector:
    """
    Simultaneous detector for black dots and blue dots using dual HSV threshold masks,
    ROI masking, spatial deduplication, and independent temporal tracking per color class.
    """

    CLASS_BLACK = "black_dot"
    CLASS_BLUE = "blue_dot"

    def __init__(
        self,
        black_lower_hsv: Tuple[int, int, int] = (0, 0, 0),
        black_upper_hsv: Tuple[int, int, int] = (180, 255, 75),
        blue_lower_hsv: Tuple[int, int, int] = (100, 50, 50),
        blue_upper_hsv: Tuple[int, int, int] = (135, 255, 255),
        min_area: float = 25.0,
        max_area: float = 4000.0,
        roi: Optional[Tuple[int, int, int, int]] = None,
        morph_kernel_size: int = 3,
        max_aspect_ratio: float = 3.5,
        merge_distance: float = 16.0,
        stability_min_hits: int = 3,
        max_missed_frames: int = 3,
        match_distance: float = 30.0,
    ):
        """
        Args:
            black_lower_hsv: Lower HSV threshold for black/dark dots.
            black_upper_hsv: Upper HSV threshold for black/dark dots.
            blue_lower_hsv: Lower HSV threshold for blue dots (BLUE_LOWER).
            blue_upper_hsv: Upper HSV threshold for blue dots (BLUE_UPPER).
            min_area: Minimum contour area in pixels.
            max_area: Maximum contour area in pixels.
            roi: Region of Interest (x1, y1, x2, y2). If None, full frame is used.
            morph_kernel_size: Kernel size for morphological noise cleanup.
            max_aspect_ratio: Max bounding box ratio max(w/h, h/w) to filter thin scratches/edges.
            merge_distance: Distance threshold to merge fragmented pieces of the same dot.
            stability_min_hits: Consecutive frames required before a dot is marked STABLE.
            max_missed_frames: Number of frames a dot can be missed before discarding track.
            match_distance: Max pixel movement between consecutive frames for the same dot.
        """
        self.black_lower_hsv = np.array(black_lower_hsv, dtype=np.uint8)
        self.black_upper_hsv = np.array(black_upper_hsv, dtype=np.uint8)
        self.blue_lower_hsv = np.array(blue_lower_hsv, dtype=np.uint8)
        self.blue_upper_hsv = np.array(blue_upper_hsv, dtype=np.uint8)

        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.roi = roi
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.merge_distance = float(merge_distance)

        # Temporal stability tracking state (independent for each color class)
        self.stability_min_hits = int(stability_min_hits)
        self.max_missed_frames = int(max_missed_frames)
        self.match_distance = float(match_distance)

        self.tracked_targets: Dict[str, Dict[int, Target]] = {
            self.CLASS_BLACK: {},
            self.CLASS_BLUE: {},
        }
        self.next_target_ids: Dict[str, int] = {
            self.CLASS_BLACK: 1,
            self.CLASS_BLUE: 1,
        }

        self.kernel = None
        if morph_kernel_size > 0:
            self.kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
            )

    def reset_tracker(self) -> None:
        """Resets all temporal tracking state across all classes."""
        self.tracked_targets[self.CLASS_BLACK].clear()
        self.tracked_targets[self.CLASS_BLUE].clear()
        self.next_target_ids[self.CLASS_BLACK] = 1
        self.next_target_ids[self.CLASS_BLUE] = 1

    def get_masks(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates two separate binary masks:
        1. mask_black: binary mask for black/dark pixels
        2. mask_blue: binary mask for blue pixels

        Applies morphological cleanup and ROI zeroing to both.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. Individual color thresholding
        mask_black = cv2.inRange(hsv, self.black_lower_hsv, self.black_upper_hsv)
        mask_blue = cv2.inRange(hsv, self.blue_lower_hsv, self.blue_upper_hsv)

        # 2. Morphological cleanup
        if self.kernel is not None:
            mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, self.kernel)
            mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, self.kernel)

            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, self.kernel)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, self.kernel)

        # 3. Apply ROI masking
        if self.roi is not None:
            h, w = frame.shape[:2]
            rx1 = max(0, min(w, self.roi[0]))
            ry1 = max(0, min(h, self.roi[1]))
            rx2 = max(rx1, min(w, self.roi[2]))
            ry2 = max(ry1, min(h, self.roi[3]))

            roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            roi_mask[ry1:ry2, rx1:rx2] = 255

            mask_black = cv2.bitwise_and(mask_black, roi_mask)
            mask_blue = cv2.bitwise_and(mask_blue, roi_mask)

        return mask_black, mask_blue

    def get_combined_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns a single combined binary mask (logical OR of black and blue masks).
        """
        mask_black, mask_blue = self.get_masks(frame)
        return cv2.bitwise_or(mask_black, mask_blue)

    def get_mask(self, frame: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for get_combined_mask."""
        return self.get_combined_mask(frame)

    def _merge_nearby_dots(self, dots: List[DotDetection]) -> List[DotDetection]:
        """
        Merges fragmented contour detections of the SAME class that belong to the same physical dot.
        Ensures each physical dot yields exactly one detection.
        """
        if len(dots) <= 1:
            return dots

        merged: List[DotDetection] = []
        used = [False] * len(dots)

        for i in range(len(dots)):
            if used[i]:
                continue
            used[i] = True

            group = [dots[i]]
            for j in range(i + 1, len(dots)):
                if used[j]:
                    continue
                # Only merge dots of the same class
                if dots[i].class_name != dots[j].class_name:
                    continue

                dist = math.hypot(
                    dots[i].center_x - dots[j].center_x,
                    dots[i].center_y - dots[j].center_y,
                )
                if dist <= self.merge_distance:
                    used[j] = True
                    group.append(dots[j])

            if len(group) == 1:
                merged.append(group[0])
            else:
                total_area = sum(d.area for d in group)
                avg_cx = int(round(sum(d.center_x * d.area for d in group) / max(1.0, total_area)))
                avg_cy = int(round(sum(d.center_y * d.area for d in group) / max(1.0, total_area)))
                union_x1 = min(d.x1 for d in group)
                union_y1 = min(d.y1 for d in group)
                union_x2 = max(d.x2 for d in group)
                union_y2 = max(d.y2 for d in group)

                merged_dot = DotDetection(
                    class_name=dots[i].class_name,
                    center_x=avg_cx,
                    center_y=avg_cy,
                    area=total_area,
                    box=[union_x1, union_y1, union_x2, union_y2],
                    x1=union_x1,
                    y1=union_y1,
                    x2=union_x2,
                    y2=union_y2,
                )
                merged.append(merged_dot)

        return merged

    def _extract_dots_from_mask(
        self, mask: np.ndarray, class_name: str, frame_shape: Tuple[int, int]
    ) -> List[DotDetection]:
        """Extracts contours, applies area & aspect ratio filtering, and returns clean DotDetections."""
        h, w = frame_shape
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        dots: List[DotDetection] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.min_area or area > self.max_area:
                continue

            rx, ry, rw, rh = cv2.boundingRect(cnt)
            x1 = max(0, min(w, rx))
            y1 = max(0, min(h, ry))
            x2 = max(0, min(w, rx + rw))
            y2 = max(0, min(h, ry + rh))

            aspect_ratio = max(float(rw) / max(1.0, float(rh)), float(rh) / max(1.0, float(rw)))
            if aspect_ratio > self.max_aspect_ratio:
                continue

            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

            cx = max(x1, min(x2, cx))
            cy = max(y1, min(y2, cy))

            dot = DotDetection(
                class_name=class_name,
                center_x=cx,
                center_y=cy,
                area=area,
                box=[x1, y1, x2, y2],
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                contour=cnt,
            )
            dots.append(dot)

        return self._merge_nearby_dots(dots)

    def detect_raw_dots(
        self, frame: np.ndarray
    ) -> Tuple[List[DotDetection], np.ndarray, np.ndarray]:
        """
        Runs low-level detection for both black and blue dots independently.

        Returns:
            Tuple[List[DotDetection], np.ndarray, np.ndarray]:
                - Combined list of raw DotDetection objects (each with class_name).
                - mask_black: Binary mask for black dots.
                - mask_blue: Binary mask for blue dots.
        """
        if frame is None or frame.size == 0:
            return [], np.zeros((0, 0), dtype=np.uint8), np.zeros((0, 0), dtype=np.uint8)

        mask_black, mask_blue = self.get_masks(frame)
        frame_shape = frame.shape[:2]

        black_dots = self._extract_dots_from_mask(mask_black, self.CLASS_BLACK, frame_shape)
        blue_dots = self._extract_dots_from_mask(mask_blue, self.CLASS_BLUE, frame_shape)

        all_dots = black_dots + blue_dots
        all_dots.sort(key=lambda d: (d.class_name, d.center_y, d.center_x))
        return all_dots, mask_black, mask_blue

    def _update_class_targets(self, class_name: str, dots: List[DotDetection]) -> None:
        """Runs temporal tracking independently for a single color class."""
        tracked = self.tracked_targets[class_name]
        matched_dot_indices = set()
        active_target_ids = list(tracked.keys())

        # Match existing targets to incoming dots of this class
        for target_id in active_target_ids:
            target = tracked[target_id]
            best_idx = None
            best_dist = float("inf")

            for idx, dot in enumerate(dots):
                if idx in matched_dot_indices:
                    continue
                dist = math.hypot(dot.center_x - target.center_x, dot.center_y - target.center_y)
                if dist < best_dist and dist <= self.match_distance:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None:
                matched_dot = dots[best_idx]
                matched_dot_indices.add(best_idx)

                # EMA smoothing
                target.center_x = int(round(0.7 * matched_dot.center_x + 0.3 * target.center_x))
                target.center_y = int(round(0.7 * matched_dot.center_y + 0.3 * target.center_y))
                target.area = matched_dot.area
                target.box = matched_dot.box
                target.x1 = matched_dot.x1
                target.y1 = matched_dot.y1
                target.x2 = matched_dot.x2
                target.y2 = matched_dot.y2

                target.hits += 1
                target.misses = 0

                if target.hits >= self.stability_min_hits:
                    target.is_stable = True
            else:
                target.misses += 1

        # Register new targets for unmatched dots of this class
        for idx, dot in enumerate(dots):
            if idx in matched_dot_indices:
                continue

            tid = self.next_target_ids[class_name]
            self.next_target_ids[class_name] += 1

            new_target = Target(
                class_name=class_name,
                target_id=tid,
                center_x=dot.center_x,
                center_y=dot.center_y,
                area=dot.area,
                box=dot.box,
                x1=dot.x1,
                y1=dot.y1,
                x2=dot.x2,
                y2=dot.y2,
                is_stable=(self.stability_min_hits <= 1),
                hits=1,
                misses=0,
            )
            tracked[tid] = new_target

        # Prune dead tracks
        to_delete = [
            tid for tid, t in tracked.items()
            if t.misses > self.max_missed_frames
        ]
        for tid in to_delete:
            del tracked[tid]

    def update_targets(self, dots: List[DotDetection]) -> List[Target]:
        """
        Splits dots by class and runs independent temporal tracking.

        Returns:
            List[Target]: Combined list of active targets across all classes.
        """
        black_dots = [d for d in dots if d.class_name == self.CLASS_BLACK]
        blue_dots = [d for d in dots if d.class_name == self.CLASS_BLUE]

        self._update_class_targets(self.CLASS_BLACK, black_dots)
        self._update_class_targets(self.CLASS_BLUE, blue_dots)

        combined = (
            list(self.tracked_targets[self.CLASS_BLACK].values()) +
            list(self.tracked_targets[self.CLASS_BLUE].values())
        )
        combined.sort(key=lambda t: (t.class_name, t.target_id))
        return combined

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[List[Target], np.ndarray, np.ndarray]:
        """
        Primary high-level pipeline method:
        Extracts black & blue dots and updates independent temporal target tracks.

        Returns:
            Tuple[List[Target], np.ndarray, np.ndarray]:
                - Active tracked targets
                - mask_black: Binary mask for black dots
                - mask_blue: Binary mask for blue dots
        """
        raw_dots, mask_black, mask_blue = self.detect_raw_dots(frame)
        targets = self.update_targets(raw_dots)
        return targets, mask_black, mask_blue

    def get_stable_targets(self, class_name: Optional[str] = None) -> List[Target]:
        """Returns stable targets, optionally filtered by color class."""
        if class_name:
            return [t for t in self.tracked_targets.get(class_name, {}).values() if t.is_stable]
        all_targets = (
            list(self.tracked_targets[self.CLASS_BLACK].values()) +
            list(self.tracked_targets[self.CLASS_BLUE].values())
        )
        return [t for t in all_targets if t.is_stable]

    def draw_detections(
        self,
        frame: np.ndarray,
        targets: List[Target],
        draw_roi: bool = True,
        draw_unstable: bool = True,
        crosshair_size: int = 7,
    ) -> np.ndarray:
        """
        Visualizes detections on the frame:
        - Outlines FARM ROI boundary
        - Black Targets: Drawn with Red / White boxes with coordinates and 'BLACK'
        - Blue Targets: Drawn with Bright Blue boxes with coordinates and 'BLUE'
        """
        annotated = frame.copy()

        # 1. Draw Region of Interest (ROI)
        if draw_roi and self.roi is not None:
            rx1, ry1, rx2, ry2 = self.roi
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 180, 0), 2)
            cv2.putText(
                annotated,
                "FARM ROI",
                (rx1 + 6, ry1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )

        # 2. Draw Targets
        for target in targets:
            if not target.is_stable and not draw_unstable:
                continue

            cx, cy = target.center_x, target.center_y

            if target.class_name == self.CLASS_BLACK:
                # Black Targets: Red/White boxes
                if target.is_stable:
                    box_color = (0, 0, 255)       # Red border
                    text_color = (255, 255, 255)  # White text
                    badge_bg = (0, 0, 200)        # Red badge
                    center_color = (0, 0, 255)    # Red crosshair
                    label = f"BLACK T{target.target_id} ({cx},{cy})"
                else:
                    box_color = (100, 100, 255)   # Light red
                    text_color = (255, 255, 255)
                    badge_bg = (50, 50, 150)
                    center_color = (100, 100, 255)
                    label = f"BLACK T{target.target_id} [ACQ {target.hits}/{self.stability_min_hits}]"
            else:
                # Blue Targets: Bright Blue boxes
                if target.is_stable:
                    box_color = (255, 180, 0)     # Bright Blue / Cyan in BGR
                    text_color = (255, 255, 255)  # White text
                    badge_bg = (200, 100, 0)      # Deep blue badge
                    center_color = (255, 180, 0)  # Blue crosshair
                    label = f"BLUE T{target.target_id} ({cx},{cy})"
                else:
                    box_color = (255, 220, 100)   # Light blue
                    text_color = (255, 255, 255)
                    badge_bg = (180, 80, 0)
                    center_color = (255, 220, 100)
                    label = f"BLUE T{target.target_id} [ACQ {target.hits}/{self.stability_min_hits}]"

            # Draw bounding box
            cv2.rectangle(
                annotated,
                (target.x1, target.y1),
                (target.x2, target.y2),
                box_color,
                2 if target.is_stable else 1
            )

            # Draw center marker & crosshair (+)
            cv2.circle(annotated, (cx, cy), 3, center_color, -1)
            cv2.line(annotated, (cx - crosshair_size, cy), (cx + crosshair_size, cy), center_color, 1)
            cv2.line(annotated, (cx, cy - crosshair_size), (cx, cy + crosshair_size), center_color, 1)

            # Draw label banner
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            thickness = 1

            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_x = target.x2 + 4
            if text_x + tw + 10 > annotated.shape[1]:
                text_x = max(5, target.x1 - tw - 10)
            text_y = max(15, min(annotated.shape[0] - 5, target.center_y + 4))

            cv2.rectangle(
                annotated,
                (text_x - 2, text_y - th - 2),
                (text_x + tw + 2, text_y + 2),
                badge_bg,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

        return annotated


# Backward compatibility alias
BlackDotDetector = DualDotDetector
