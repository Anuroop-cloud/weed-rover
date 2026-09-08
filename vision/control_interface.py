"""
Control Interface Module for weed-rover.

Locks the AI perception output contract for the robotic control / actuation layer.

Final AI Output Contract:
1. `weed_detected`: int
   - 0 = no weed detected
   - 1 = weed detected
2. `column`: int
   - integer 0 through 7
   - identifies which column of the physical 8x8 LED matrix corresponds to the detected weed
   - -1 when no weed is detected (weed_detected = 0)

Physical LED Matrix:
- 8 columns x 8 rows
- Entire physical matrix width: 3.2 cm
- Column width: 3.2 / 8 = 0.4 cm per column
- Matrix center is the X = 0 reference:
    column 0: [-1.6, -1.2) cm, center = -1.4 cm
    column 1: [-1.2, -0.8) cm, center = -1.0 cm
    column 2: [-0.8, -0.4) cm, center = -0.6 cm
    column 3: [-0.4,  0.0) cm, center = -0.2 cm
    column 4: [ 0.0, +0.4) cm, center = +0.2 cm
    column 5: [+0.4, +0.8) cm, center = +0.6 cm
    column 6: [+0.8, +1.2) cm, center = +1.0 cm
    column 7: [+1.2, +1.6] cm, center = +1.4 cm

Pipeline:
camera frame -> YOLO detection -> candidate filtering -> ground projection via RobotGeometry
-> apply camera-to-LED lateral offset -> select closest target to tool -> map X to LED column 0..7
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from config.vision_config import (
        LED_MATRIX_WIDTH_CM,
        LED_MATRIX_COLUMNS,
        CONFIDENCE_THRESHOLD,
    )
    from vision.detector import Detection, YOLODetector
    from vision.robot_geometry import RobotGeometry
except ImportError:
    from detector import Detection, YOLODetector
    from robot_geometry import RobotGeometry
    LED_MATRIX_WIDTH_CM = 3.2
    LED_MATRIX_COLUMNS = 8
    CONFIDENCE_THRESHOLD = 0.35


class LEDMatrixMapper:
    """
    Translates continuous lateral ground coordinates (X_cm relative to matrix center)
    to physical LED matrix column indices (0 through num_columns - 1).

    Configurable with matrix physical width and column count.
    Safe clamping ensures valid coordinates always produce a valid column (0..num_columns-1).
    """

    def __init__(
        self,
        matrix_width_cm: float = LED_MATRIX_WIDTH_CM,
        num_columns: int = LED_MATRIX_COLUMNS,
    ):
        """
        Args:
            matrix_width_cm: Total physical width of the LED matrix in cm (default: 3.2).
            num_columns: Number of discrete columns in the matrix (default: 8).
        """
        if matrix_width_cm <= 0:
            raise ValueError("matrix_width_cm must be strictly positive.")
        if num_columns <= 0:
            raise ValueError("num_columns must be a positive integer.")

        self.matrix_width_cm = float(matrix_width_cm)
        self.num_columns = int(num_columns)
        self.column_width_cm = self.matrix_width_cm / self.num_columns
        self.half_width_cm = self.matrix_width_cm / 2.0

    def x_to_column(self, x_cm: float) -> int:
        """
        Converts lateral X position in cm (relative to LED matrix center X=0)
        to a column index 0 through (num_columns - 1).

        Coordinate system:
            Left edge of matrix  = -matrix_width_cm / 2  (-1.6 cm for 3.2 cm width)
            Right edge of matrix = +matrix_width_cm / 2  (+1.6 cm for 3.2 cm width)
            Center of matrix     = 0.0 cm

        Clamping:
            Values < -half_width_cm clamp safely to column 0.
            Values >= +half_width_cm clamp safely to column (num_columns - 1).

        Args:
            x_cm: Lateral distance in cm (+X = right, -X = left).

        Returns:
            int: Column index (0..num_columns-1).
        """
        offset_from_left = float(x_cm) + self.half_width_cm

        # Round to 6 decimal places to prevent IEEE 754 floating-point artifacts at column boundaries
        col = int(math.floor(round(offset_from_left / self.column_width_cm, 6)))

        # Clamp safely between 0 and num_columns - 1
        return max(0, min(self.num_columns - 1, col))

    def column_center_x(self, col: int) -> float:
        """
        Returns the theoretical X center in cm for a given column index.

        Args:
            col: Column index (0..num_columns-1).

        Returns:
            float: Center X coordinate in cm.
        """
        clamped_col = max(0, min(self.num_columns - 1, int(col)))
        left_edge = -self.half_width_cm + (clamped_col * self.column_width_cm)
        return round(left_edge + (self.column_width_cm / 2.0), 4)

    def column_bounds(self, col: int) -> Tuple[float, float]:
        """
        Returns (x_min, x_max) in cm for a given column index.

        Args:
            col: Column index (0..num_columns-1).

        Returns:
            Tuple[float, float]: (x_min, x_max) bounds in cm.
        """
        clamped_col = max(0, min(self.num_columns - 1, int(col)))
        x_min = -self.half_width_cm + (clamped_col * self.column_width_cm)
        x_max = x_min + self.column_width_cm
        return (round(x_min, 4), round(x_max, 4))


def x_to_matrix_column(
    x_cm: float,
    matrix_width_cm: float = LED_MATRIX_WIDTH_CM,
    num_columns: int = LED_MATRIX_COLUMNS,
) -> int:
    """
    Convenience functional interface mapping lateral X position (cm) to matrix column (0..7).

    Args:
        x_cm: Lateral X position in cm relative to matrix center.
        matrix_width_cm: Physical width of matrix in cm (default: 3.2).
        num_columns: Number of matrix columns (default: 8).

    Returns:
        int: Clamped column index (0..num_columns-1).
    """
    return LEDMatrixMapper(matrix_width_cm=matrix_width_cm, num_columns=num_columns).x_to_column(x_cm)


@dataclass
class WeedControlOutput:
    """
    Clean output object representing the AI perception output contract
    for the robot control and actuation system.
    """
    weed_detected: int  # 0 = no weed detected, 1 = weed detected
    column: int         # primary closest column (-1 when weed_detected == 0; 0..7 when weed_detected == 1)
    columns: List[int] = field(default_factory=list)  # all active columns [0..7] with detected weeds

    # Diagnostic metadata (available internally; omitted from pure control payload)
    confidence: Optional[float] = None
    bbox: Optional[List[int]] = None
    center_pixel: Optional[Tuple[int, int]] = None
    x_camera_cm: Optional[float] = None
    y_camera_cm: Optional[float] = None
    x_led_cm: Optional[float] = None
    y_led_cm: Optional[float] = None
    distance_cm: Optional[float] = None
    class_name: Optional[str] = None

    def to_control_payload(self) -> Dict[str, int]:
        """
        Returns the exact two-field control payload required by Advay's ROS / controller node:
        {
            "weed_detected": 0 or 1,
            "column": -1 or 0..7
        }
        """
        return {
            "weed_detected": int(self.weed_detected),
            "column": int(self.column),
        }

    def to_multi_control_payload(self) -> Dict[str, Any]:
        """
        Returns multi-column payload for simultaneous actuation across multiple columns:
        {
            "weed_detected": 0 or 1,
            "column": -1 or 0..7,
            "columns": [0, 5]
        }
        """
        return {
            "weed_detected": int(self.weed_detected),
            "column": int(self.column),
            "columns": list(self.columns),
        }

    @classmethod
    def no_weed(cls) -> "WeedControlOutput":
        """Factory method returning standard negative detection state."""
        return cls(weed_detected=0, column=-1, columns=[])

    def to_dict(self) -> Dict[str, Any]:
        """Returns full diagnostic dictionary including spatial and confidence metadata."""
        return {
            "weed_detected": self.weed_detected,
            "column": self.column,
            "columns": self.columns,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "bbox": self.bbox,
            "center_pixel": self.center_pixel,
            "x_camera_cm": self.x_camera_cm,
            "y_camera_cm": self.y_camera_cm,
            "x_led_cm": self.x_led_cm,
            "y_led_cm": self.y_led_cm,
            "distance_cm": self.distance_cm,
            "class_name": self.class_name,
        }


class WeedControlPipeline:
    """
    End-to-end perception pipeline that accepts camera frames or detections,
    selects the closest valid weed to the tool in ground coordinates,
    and produces the locked WeedControlOutput contract.
    """

    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        model_path: str = "models/best.pt",
        conf_threshold: float = CONFIDENCE_THRESHOLD,
        geometry: Optional[RobotGeometry] = None,
        matrix_mapper: Optional[LEDMatrixMapper] = None,
        target_classes: Optional[List[str]] = None,
    ):
        """
        Args:
            detector: Optional pre-instantiated YOLODetector.
            model_path: Model weights file path (used if detector is not provided).
            conf_threshold: Minimum confidence score to treat a detection as valid.
            geometry: RobotGeometry instance for top-down ground projection and camera-to-LED offsets.
            matrix_mapper: LEDMatrixMapper instance for X_led -> column 0..7 mapping.
            target_classes: Optional list of class names to filter for (None = accept all).
        """
        self._detector = detector
        self.model_path = model_path
        self.conf_threshold = float(conf_threshold)
        self.geometry = geometry if geometry is not None else RobotGeometry()
        self.matrix_mapper = matrix_mapper if matrix_mapper is not None else LEDMatrixMapper()
        self.target_classes = target_classes if target_classes is not None else ["weed"]

    @property
    def detector(self) -> YOLODetector:
        """Lazy-loaded YOLO detector instance."""
        if self._detector is None:
            self._detector = YOLODetector(
                model_path=self.model_path,
                conf_threshold=self.conf_threshold,
            )
        return self._detector

    def process_detections(self, detections: List[Detection]) -> WeedControlOutput:
        """
        Processes a list of structured detections, selects the closest target to the tool,
        and generates the final WeedControlOutput.

        Selection logic:
        1. Filters detections by confidence >= conf_threshold.
        2. Filters detections by target_classes (if configured).
        3. Projects bounding-box center (cx, cy) to ground coordinates (X_cam, Y_cam) via RobotGeometry.
        4. Applies camera-to-LED lateral and forward offsets via RobotGeometry.ground_to_led_frame.
        5. Computes Euclidean distance to tool center: hypot(X_led, Y_led).
        6. Deterministically selects ONE target: the weed closest to the tool.
           Tie-breaking uses higher confidence, then lower pixel x1 coordinate.
        7. Maps selected X_led to LED matrix column 0..7 (clamped safely).

        Args:
            detections: List of Detection objects.

        Returns:
            WeedControlOutput: Formatted output contract.
        """
        if not detections:
            return WeedControlOutput.no_weed()

        valid_candidates = []

        for det in detections:
            # 1. Confidence check
            if det.confidence < self.conf_threshold:
                continue

            # 2. Class check
            if self.target_classes is not None and det.class_name not in self.target_classes:
                continue

            # 3. Ground plane projection
            ground_cam = self.geometry.pixel_to_ground(det.center_x, det.center_y)
            if ground_cam is None:
                continue

            x_cam, y_cam = ground_cam

            # 4. Transform to LED / tool reference frame
            x_led, y_led = self.geometry.ground_to_led_frame(x_cam, y_cam)

            # 5. Distance to tool center (origin of LED frame)
            dist = math.hypot(x_led, y_led)

            valid_candidates.append({
                "det": det,
                "dist": dist,
                "x_cam": x_cam,
                "y_cam": y_cam,
                "x_led": x_led,
                "y_led": y_led,
            })

        if not valid_candidates:
            return WeedControlOutput.no_weed()

        # 6. Deterministic selection: sort by closest distance to tool,
        # tie-break by higher confidence, then lower bounding-box x1
        valid_candidates.sort(
            key=lambda c: (
                round(c["dist"], 4),
                -round(c["det"].confidence, 4),
                c["det"].x1,
                c["det"].y1,
            )
        )

        selected = valid_candidates[0]
        selected_det = selected["det"]

        # 7. Map lateral X in tool frame to LED column 0..7 for all weeds
        all_columns = sorted(list(set(
            self.matrix_mapper.x_to_column(c["x_led"]) for c in valid_candidates
        )))

        # Primary closest weed column
        col = self.matrix_mapper.x_to_column(selected["x_led"])

        return WeedControlOutput(
            weed_detected=1,
            column=col,
            columns=all_columns,
            confidence=round(float(selected_det.confidence), 4),
            bbox=selected_det.box,
            center_pixel=(selected_det.center_x, selected_det.center_y),
            x_camera_cm=selected["x_cam"],
            y_camera_cm=selected["y_cam"],
            x_led_cm=selected["x_led"],
            y_led_cm=selected["y_led"],
            distance_cm=round(selected["dist"], 2),
            class_name=selected_det.class_name,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> WeedControlOutput:
        """
        Runs full end-to-end inference on a raw OpenCV BGR image frame
        and outputs the WeedControlOutput.

        Args:
            frame: OpenCV BGR NumPy array (H, W, 3).
            conf_threshold: Optional override for minimum confidence.

        Returns:
            WeedControlOutput: Control output contract.
        """
        if frame is None or frame.size == 0:
            return WeedControlOutput.no_weed()

        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        detections = self.detector.detect(frame, conf_threshold=threshold)
        return self.process_detections(detections)


# Public alias matching alternative naming convention
AIControlPipeline = WeedControlPipeline


def demo_control_interface() -> None:
    """Prints a clear terminal demonstration of the column mapping and output payload."""
    print("=" * 70)
    print("WEED ROVER - AI TO CONTROL OUTPUT INTERFACE DEMO")
    print("=" * 70)
    mapper = LEDMatrixMapper()
    print(f"Matrix Width : {mapper.matrix_width_cm:.1f} cm")
    print(f"Columns      : {mapper.num_columns} (0 through {mapper.num_columns - 1})")
    print(f"Column Width : {mapper.column_width_cm:.2f} cm")
    print("-" * 70)
    print(f"{'Column':<8} | {'Bounds (cm)':<18} | {'Center X (cm)':<15} | Sample X -> Output Col")
    print("-" * 70)

    for c in range(mapper.num_columns):
        x_min, x_max = mapper.column_bounds(c)
        center = mapper.column_center_x(c)
        mapped = mapper.x_to_column(center)
        print(f"Col {c:<4} | [{x_min:>5.2f}, {x_max:>5.2f}) cm   | {center:>6.2f} cm        | {center:>5.2f} cm -> Col {mapped}")

    print("=" * 70)
    print("Edge & Clamping Behavior:")
    for test_x in [-5.0, -1.6, -0.01, 0.0, 0.01, 1.6, 5.0]:
        print(f"  X = {test_x:>6.2f} cm -> Column {mapper.x_to_column(test_x)}")

    print("-" * 70)
    print("Control Payload Examples:")
    no_weed = WeedControlOutput.no_weed()
    print(f"  No weed detected payload: {no_weed.to_control_payload()}")
    sample_weed = WeedControlOutput(weed_detected=1, column=5)
    print(f"  Weed detected payload   : {sample_weed.to_control_payload()}")
    print("=" * 70)


if __name__ == "__main__":
    demo_control_interface()
