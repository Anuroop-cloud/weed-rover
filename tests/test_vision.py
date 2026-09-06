"""
Unit and integration tests for camera and detector components.
Can run in headless environments using synthetic test images.
"""

import numpy as np
import pytest
from vision.camera import Camera
from vision.detector import Detection, YOLODetector


def test_detection_dataclass():
    """Verify Detection structure and dictionary serialization."""
    det = Detection(
        class_id=0,
        class_name="weed",
        confidence=0.885,
        box=[10, 20, 100, 120],
        x1=10,
        y1=20,
        x2=100,
        y2=120,
        center_x=55,
        center_y=70
    )
    data = det.to_dict()
    assert data["class_id"] == 0
    assert data["class_name"] == "weed"
    assert data["confidence"] == 0.885
    assert data["box"] == [10, 20, 100, 120]
    assert data["center_x"] == 55
    assert data["center_y"] == 70


def test_camera_fps_calculation():
    """Verify camera FPS drawing on frame."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = Camera.draw_fps(dummy_frame, 30.5)
    assert annotated.shape == (480, 640, 3)
    assert isinstance(annotated, np.ndarray)


def test_detector_on_blank_frame():
    """Verify detector handles empty/synthetic frame without crashing."""
    detector = YOLODetector(model_path="yolov8n.pt")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)
    assert isinstance(detections, list)

    # Drawing test
    annotated = detector.draw_detections(dummy_frame, detections)
    assert annotated.shape == (480, 640, 3)
