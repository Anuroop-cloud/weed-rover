"""
Unit and integration tests for camera and detector components.
Can run in headless environments using synthetic test images.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


def test_detection_debug_format():
    """Verify debug string follows exact requested format: class=<name> confidence=<conf> bbox=(x1,y1,x2,y2) center=(cx,cy)."""
    det = Detection(
        class_id=1,
        class_name="crop",
        confidence=0.923,
        box=[50, 60, 200, 250],
        x1=50,
        y1=60,
        x2=200,
        y2=250,
        center_x=125,
        center_y=155
    )
    debug_str = det.to_debug_string()
    expected = "class=crop confidence=0.92 bbox=(50,60,200,250) center=(125,155)"
    assert debug_str == expected
    assert str(det) == expected


def test_camera_fps_with_detection_count():
    """Verify FPS overlay with detection count works seamlessly."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = Camera.draw_fps(dummy_frame, 28.4, detection_count=5)
    assert annotated.shape == (480, 640, 3)
    assert isinstance(annotated, np.ndarray)


def test_yolo_missing_model_raises_error():
    """Verify YOLODetector raises clear FileNotFoundError if custom model path does not exist."""
    with pytest.raises(FileNotFoundError) as exc_info:
        YOLODetector(model_path="models/nonexistent_model.pt")
    assert "Model weights file not found" in str(exc_info.value)


def test_black_dot_detector_alias():
    """Verify vision.black_dot_detector exports compatible detector classes."""
    from vision.black_dot_detector import BlackDotDetector, DualDotDetector, DotDetection, Target
    assert BlackDotDetector is DualDotDetector
    detector = BlackDotDetector()
    assert hasattr(detector, "detect")


