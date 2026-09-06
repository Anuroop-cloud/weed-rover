"""
Unit tests for ColorTargetDetector and TargetResult.
Tests HSV color thresholding, dual-range red detection, morphological filtering,
centroid calculations, and drawing on synthetic frames.
"""

import os
import sys
import cv2
import numpy as np
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.target_detector import ColorTargetDetector, TargetResult


def test_target_result_dataclass():
    """Verify TargetResult structure, debug formatting, and serialization."""
    res = TargetResult(
        detected=True,
        confidence=0.95,
        box=[100, 150, 220, 270],
        x1=100,
        y1=150,
        x2=220,
        y2=270,
        center_x=160,
        center_y=210,
        area=14400.0,
    )
    d = res.to_dict()
    assert d["detected"] is True
    assert d["confidence"] == 0.95
    assert d["box"] == [100, 150, 220, 270]
    assert d["center_x"] == 160
    assert d["center_y"] == 210
    assert d["area"] == 14400.0

    debug_str = res.to_debug_string()
    assert "target=red" in debug_str
    assert "center=(160,210)" in debug_str
    assert "bbox=(100,150,220,270)" in debug_str


def test_target_result_undetected():
    """Verify TargetResult when no target is detected."""
    res = TargetResult(detected=False)
    assert res.detected is False
    assert "detected=False" in res.to_debug_string()


def test_target_detector_on_blank_frame():
    """Verify detector returns detected=False on a black canvas."""
    detector = ColorTargetDetector()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(blank)
    assert res.detected is False
    assert res.center_x == 0
    assert res.center_y == 0


def test_target_detector_on_synthetic_red_target():
    """Verify detection of a prominent red circular target marker."""
    detector = ColorTargetDetector(min_area=100.0)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw pure red circle at center (300, 200) with radius 30 (area ~ 2827 px)
    # OpenCV uses BGR -> (0, 0, 255) is pure red
    cv2.circle(canvas, (300, 200), 30, (0, 0, 255), -1)

    target = detector.detect(canvas)
    assert target.detected is True
    assert abs(target.center_x - 300) <= 2
    assert abs(target.center_y - 200) <= 2
    assert target.area > 2000
    assert target.confidence > 0.5
    assert target.x1 < 300 < target.x2
    assert target.y1 < 200 < target.y2


def test_target_detector_filters_small_noise():
    """Verify noise specks smaller than min_area are ignored."""
    detector = ColorTargetDetector(min_area=200.0)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw small red noise speck with radius 2 (area ~ 12 px)
    cv2.circle(canvas, (100, 100), 2, (0, 0, 255), -1)

    target = detector.detect(canvas)
    assert target.detected is False


def test_target_detector_dual_red_ranges():
    """Verify detection works across both upper and lower red HSV hue ranges."""
    detector = ColorTargetDetector(min_area=50.0)

    # Lower range: Hue ~ 0-5 (e.g. BGR (20, 20, 230))
    canvas1 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(canvas1, (100, 100), 15, (20, 20, 230), -1)
    res1 = detector.detect(canvas1)
    assert res1.detected is True

    # Upper range: Hue ~ 170-179 (e.g. BGR (80, 20, 230))
    canvas2 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(canvas2, (100, 100), 15, (80, 20, 230), -1)
    res2 = detector.detect(canvas2)
    assert res2.detected is True


def test_target_detector_drawing():
    """Verify drawing bounding box and crosshairs produces valid ndarray."""
    detector = ColorTargetDetector()
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(canvas, (320, 240), 25, (0, 0, 255), -1)

    target = detector.detect(canvas)
    annotated = detector.draw_target(canvas, target)
    assert annotated.shape == (480, 640, 3)
    assert isinstance(annotated, np.ndarray)
