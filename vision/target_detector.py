"""
Color-based Target Detector for weed-rover.
Detects colored markers/LED targets (default: red) on the farm floor using
dual-range HSV thresholding, morphological filtering, and contour extraction.

Outputs structured pixel coordinates and bounding geometry for downstream
serial communication to ESP32 / laser actuators.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class TargetResult:
    """
    Structured representation of a detected target marker.
    Matches downstream target interface requirements for robotic actuation.
    """
    detected: bool = False
    confidence: float = 0.0
    box: List[int] = field(default_factory=lambda: [0, 0, 0, 0])  # [x1, y1, x2, y2]
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    center_x: int = 0
    center_y: int = 0
    area: float = 0.0

    def to_debug_string(self) -> str:
        """Format target detection for terminal output."""
        if not self.detected:
            return "target=none detected=False"
        return (
            f"target=red confidence={self.confidence:.2f} "
            f"bbox=({self.x1},{self.y1},{self.x2},{self.y2}) "
            f"center=({self.center_x},{self.center_y}) area={self.area:.1f}"
        )

    def __str__(self) -> str:
        return self.to_debug_string()

    def to_dict(self) -> Dict[str, Any]:
        """Convert target result to dictionary."""
        return {
            "detected": self.detected,
            "confidence": round(float(self.confidence), 4),
            "box": self.box,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "area": round(float(self.area), 2),
        }


class ColorTargetDetector:
    """
    Independent color marker detector using HSV color spaces.
    Handles red hue wrapping around 0/180 boundaries.
    """

    def __init__(
        self,
        lower_red1: Tuple[int, int, int] = (0, 100, 100),
        upper_red1: Tuple[int, int, int] = (10, 255, 255),
        lower_red2: Tuple[int, int, int] = (160, 100, 100),
        upper_red2: Tuple[int, int, int] = (179, 255, 255),
        min_area: float = 120.0,
        max_area: Optional[float] = None,
        morph_kernel_size: int = 5,
    ):
        """
        Args:
            lower_red1: Lower bound for first red hue range (0-10) [H, S, V]
            upper_red1: Upper bound for first red hue range [H, S, V]
            lower_red2: Lower bound for second red hue range (160-179) [H, S, V]
            upper_red2: Upper bound for second red hue range [H, S, V]
            min_area: Minimum contour area in pixels to filter out noise
            max_area: Optional maximum contour area to filter out oversized artifacts
            morph_kernel_size: Kernel size for morphological noise removal
        """
        self.lower_red1 = np.array(lower_red1, dtype=np.uint8)
        self.upper_red1 = np.array(upper_red1, dtype=np.uint8)
        self.lower_red2 = np.array(lower_red2, dtype=np.uint8)
        self.upper_red2 = np.array(upper_red2, dtype=np.uint8)
        self.min_area = min_area
        self.max_area = max_area
        
        # Morphological structuring element
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )

    def create_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Converts BGR frame to HSV, creates dual-range red masks,
        and applies morphological opening & closing to clean noise.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Dual-range thresholding to handle hue wrap-around at 0/180
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        combined_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations:
        # OPEN (erode then dilate) removes small stray noise specks
        # CLOSE (dilate then erode) bridges small internal gaps in the target
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel)
        return cleaned

    def detect_all(self, frame: np.ndarray) -> List[TargetResult]:
        """
        Finds all valid targets passing the area filter.
        Returns a list of TargetResult instances sorted by area descending.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        mask = self.create_mask(frame)

        # Find external contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        results: List[TargetResult] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue

            # Bounding box
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            x1 = max(0, min(w, rx))
            y1 = max(0, min(h, ry))
            x2 = max(0, min(w, rx + rw))
            y2 = max(0, min(h, ry + rh))

            # Compute precise center using moments
            moments = cv2.moments(cnt)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

            # Clamp center within box
            cx = max(x1, min(x2, cx))
            cy = max(y1, min(y2, cy))

            # Quality / Confidence score based on solidity (area / convex hull area)
            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            solidity = (area / hull_area) if hull_area > 0 else 0.0
            confidence = float(max(0.1, min(1.0, solidity)))

            res = TargetResult(
                detected=True,
                confidence=confidence,
                box=[x1, y1, x2, y2],
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                center_x=cx,
                center_y=cy,
                area=area,
            )
            results.append(res)

        # Sort largest target first
        results.sort(key=lambda r: r.area, reverse=True)
        return results

    def detect(self, frame: np.ndarray) -> TargetResult:
        """
        Finds and returns the best (largest valid) target marker in the frame.
        If no target is found, returns a TargetResult with detected=False.
        """
        all_targets = self.detect_all(frame)
        if all_targets:
            return all_targets[0]
        return TargetResult(detected=False)

    def draw_target(
        self,
        frame: np.ndarray,
        target: TargetResult,
        box_color: Tuple[int, int, int] = (0, 255, 255),
        crosshair_color: Tuple[int, int, int] = (0, 0, 255),
        crosshair_size: int = 10,
    ) -> np.ndarray:
        """
        Draws bounding box, center crosshair, and targeting metadata on the frame.
        """
        annotated = frame.copy()
        if not target.detected:
            return annotated

        # 1. Bounding box
        cv2.rectangle(
            annotated,
            (target.x1, target.y1),
            (target.x2, target.y2),
            box_color,
            2
        )

        # 2. Centroid circle & targeting crosshair
        cx, cy = target.center_x, target.center_y
        cv2.circle(annotated, (cx, cy), 4, crosshair_color, -1)
        cv2.line(
            annotated,
            (cx - crosshair_size, cy),
            (cx + crosshair_size, cy),
            crosshair_color,
            1
        )
        cv2.line(
            annotated,
            (cx, cy - crosshair_size),
            (cx, cy + crosshair_size),
            crosshair_color,
            1
        )

        # 3. Target label with pixel coordinates
        label = f"TARGET [{cx}, {cy}] A:{int(target.area)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        ly1 = max(target.y1 - th - 8, 0)
        ly2 = max(target.y1, th + 8)
        lx2 = min(target.x1 + tw + 10, annotated.shape[1])

        cv2.rectangle(annotated, (target.x1, ly1), (lx2, ly2), box_color, -1)
        cv2.putText(
            annotated,
            label,
            (target.x1 + 5, ly2 - 5),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

        return annotated


def run_target_camera_feed(camera_id: int = 0, width: int = 640, height: int = 480):
    """
    Runs live webcam feed with real-time target detection.
    Press 'q' or ESC to exit.
    """
    try:
        from vision.camera import Camera
    except ImportError:
        from camera import Camera

    camera = Camera(device_id=camera_id, width=width, height=height)
    if not camera.is_opened:
        print("[TargetDetector] Camera could not be opened. Testing on synthetic canvas instead.")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a synthetic red target marker
        cv2.circle(dummy, (320, 240), 25, (0, 0, 255), -1)
        detector = ColorTargetDetector()
        res = detector.detect(dummy)
        print(f"[TargetDetector Test] {res.to_debug_string()}")
        return

    detector = ColorTargetDetector()
    window_name = "Weed Rover - Target Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("[TargetDetector] Live target detection started. Press 'q' or ESC to exit.")

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("[TargetDetector WARNING] Frame grab failed.")
                break

            target = detector.detect(frame)
            annotated = detector.draw_target(frame, target)

            Camera.draw_fps(
                annotated,
                camera.get_fps(),
                detection_count=1 if target.detected else 0
            )

            if target.detected:
                print(target.to_debug_string())

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[TargetDetector] Feed closed cleanly.")


if __name__ == "__main__":
    run_target_camera_feed()

