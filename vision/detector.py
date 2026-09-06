"""
Detector module for weed-rover.
Provides modular YOLO-based object detection on OpenCV/NumPy frames,
returning structured bounding boxes, confidence scores, and centroid targeting coordinates.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """
    Structured detection representation.
    Designed for easy consumption by tracking, laser targeting, or robotic actuators.
    """
    class_id: int
    class_name: str
    confidence: float
    box: List[int]        # [x1, y1, x2, y2]
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: int
    center_y: int

    def to_debug_string(self) -> str:
        """Format detection for terminal debug output: class=<name> confidence=<confidence> bbox=(x1,y1,x2,y2) center=(cx,cy)"""
        return f"class={self.class_name} confidence={self.confidence:.2f} bbox=({self.x1},{self.y1},{self.x2},{self.y2}) center=({self.center_x},{self.center_y})"

    def __str__(self) -> str:
        return self.to_debug_string()

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection data to dictionary format."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "box": self.box,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "center_x": self.center_x,
            "center_y": self.center_y,
        }


class YOLODetector:
    """
    Modular YOLO detector wrapper.
    Accepts raw OpenCV NumPy frames and outputs structured detections.
    Supports easy switching from standard pretrained weights ('yolov8n.pt')
    to custom weed-specific fine-tuned models ('models/best.pt').
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        device: Optional[str] = None
    ):
        """
        Args:
            model_path: Path or identifier for the model weights (default: 'yolov8n.pt').
            conf_threshold: Default confidence threshold for detections.
            device: Computation device ('cpu', 'cuda', etc.). Defaults to auto-selection.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        
        print(f"[Detector] Loading YOLO model from '{self.model_path}'...")
        self.model = YOLO(self.model_path)
        if self.device:
            self.model.to(self.device)
        print(f"[Detector] Model loaded successfully. Class count: {len(self.model.names)}")

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        classes: Optional[List[int]] = None
    ) -> List[Detection]:
        """
        Runs object detection on a single OpenCV/NumPy frame.

        Args:
            frame: Input image as a NumPy BGR array (H, W, 3).
            conf_threshold: Optional override for minimum confidence threshold.
            classes: Optional list of class IDs to filter for.

        Returns:
            List[Detection]: List of structured detections.
        """
        if frame is None or frame.size == 0:
            return []

        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        # Run inference (verbose=False keeps terminal output clean during streaming)
        results = self.model.predict(
            source=frame,
            conf=threshold,
            classes=classes,
            verbose=False
        )

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        h, w = frame.shape[:2]

        for box in result.boxes:
            # Extract coordinates (ultralytics xyxy returns absolute pixel coords of input frame)
            raw_x1, raw_y1, raw_x2, raw_y2 = map(int, box.xyxy[0].tolist())
            x1 = max(0, min(w, raw_x1))
            y1 = max(0, min(h, raw_y1))
            x2 = max(0, min(w, raw_x2))
            y2 = max(0, min(h, raw_y2))

            # Ensure confidence is strictly between 0.0 and 1.0
            raw_conf = float(box.conf[0].item())
            conf = max(0.0, min(1.0, raw_conf))

            cls_id = int(box.cls[0].item())
            cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

            # Compute centroid (midpoint of the bounding box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            det = Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                box=[x1, y1, x2, y2],
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                center_x=cx,
                center_y=cy
            )
            detections.append(det)

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        box_color: Tuple[int, int, int] = (0, 220, 255),
        centroid_color: Tuple[int, int, int] = (0, 0, 255)
    ) -> np.ndarray:
        """
        Draws bounding boxes, labels, confidence, and targeting centroids onto the frame.

        Args:
            frame: OpenCV image array to draw on.
            detections: List of Detection instances.
            box_color: BGR color for bounding box.
            centroid_color: BGR color for target centroid & crosshair.

        Returns:
            np.ndarray: Annotated frame.
        """
        annotated = frame.copy()

        for det in detections:
            # 1. Draw Bounding Box
            cv2.rectangle(
                annotated,
                (det.x1, det.y1),
                (det.x2, det.y2),
                box_color,
                2
            )

            # 2. Draw Centroid & Crosshair (targeting marker)
            cx, cy = det.center_x, det.center_y
            cv2.circle(annotated, (cx, cy), 4, centroid_color, -1)
            # Small crosshair
            crosshair_size = 8
            cv2.line(annotated, (cx - crosshair_size, cy), (cx + crosshair_size, cy), centroid_color, 1)
            cv2.line(annotated, (cx, cy - crosshair_size), (cx, cy + crosshair_size), centroid_color, 1)

            # 3. Label badge
            label = f"{det.class_name} {det.confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Position label box above bounding box, or just below top edge if too close to border
            label_y1 = max(det.y1 - text_h - 8, 0)
            label_y2 = max(det.y1, text_h + 8)
            label_x2 = min(det.x1 + text_w + 10, annotated.shape[1])

            # Label background banner
            cv2.rectangle(
                annotated,
                (det.x1, label_y1),
                (label_x2, label_y2),
                box_color,
                -1
            )
            # Label text in dark contrast
            cv2.putText(
                annotated,
                label,
                (det.x1 + 5, label_y2 - 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        return annotated


if __name__ == "__main__":
    # Quick self-test with a synthetic canvas
    print("[Test] Testing YOLODetector initialization...")
    detector = YOLODetector()
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "Weed Rover Test", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    results = detector.detect(dummy_frame)
    print(f"[Test] Detections on synthetic canvas: {len(results)}")
    annotated = detector.draw_detections(dummy_frame, results)
    print("[Test] Detector self-test successful.")
