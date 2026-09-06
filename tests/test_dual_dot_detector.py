"""
Unit tests for DualDotDetector using synthetic NumPy images.
Tests simultaneous detection of black dots and blue dots, independent temporal tracking,
ROI filtering, class separation, separate threshold masks, and visualization.
"""

import os
import sys
import cv2
import numpy as np
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.dual_dot_detector import DualDotDetector, DotDetection, Target


def test_black_circle_on_white_background():
    """Verify that a black circle on a white background is accurately detected as 'black_dot'."""
    detector = DualDotDetector(min_area=25.0, max_area=4000.0)

    canvas = np.full((480, 640, 3), 255, dtype=np.uint8)
    center_target = (250, 180)
    radius = 20
    cv2.circle(canvas, center_target, radius, (0, 0, 0), -1)

    targets, mask_black, mask_blue = detector.detect(canvas)

    assert len(targets) == 1
    t = targets[0]
    assert t.class_name == "black_dot"
    assert abs(t.center_x - center_target[0]) <= 2
    assert abs(t.center_y - center_target[1]) <= 2
    assert t.box == [t.x1, t.y1, t.x2, t.y2]
    assert cv2.countNonZero(mask_black) > 0
    assert cv2.countNonZero(mask_blue) == 0


def test_blue_circle_on_white_background():
    """Verify that a blue circle on a white background is accurately detected as 'blue_dot'."""
    detector = DualDotDetector(min_area=25.0, max_area=4000.0)

    canvas = np.full((480, 640, 3), 255, dtype=np.uint8)
    center_target = (350, 220)
    radius = 22
    # BGR (220, 50, 20) -> vibrant blue (in HSV: H~115, S~230, V~220)
    cv2.circle(canvas, center_target, radius, (220, 50, 20), -1)

    targets, mask_black, mask_blue = detector.detect(canvas)

    assert len(targets) == 1
    t = targets[0]
    assert t.class_name == "blue_dot"
    assert abs(t.center_x - center_target[0]) <= 2
    assert abs(t.center_y - center_target[1]) <= 2
    assert cv2.countNonZero(mask_blue) > 0
    assert cv2.countNonZero(mask_black) == 0


def test_simultaneous_black_and_blue_detection():
    """Verify that both black and blue dots are detected simultaneously in the same frame."""
    detector = DualDotDetector(min_area=25.0, max_area=4000.0)
    canvas = np.full((480, 640, 3), 255, dtype=np.uint8)

    # 1 black circle at (180, 200)
    cv2.circle(canvas, (180, 200), 20, (0, 0, 0), -1)
    # 1 blue circle at (450, 200)
    cv2.circle(canvas, (450, 200), 20, (220, 50, 20), -1)

    targets, mask_black, mask_blue = detector.detect(canvas)

    assert len(targets) == 2
    classes = {t.class_name for t in targets}
    assert classes == {"black_dot", "blue_dot"}

    black_t = next(t for t in targets if t.class_name == "black_dot")
    blue_t = next(t for t in targets if t.class_name == "blue_dot")

    assert abs(black_t.center_x - 180) <= 2
    assert abs(black_t.center_y - 200) <= 2
    assert abs(blue_t.center_x - 450) <= 2
    assert abs(blue_t.center_y - 200) <= 2

    assert cv2.countNonZero(mask_black) > 0
    assert cv2.countNonZero(mask_blue) > 0


def test_independent_temporal_stability_and_ids():
    """Verify that black and blue targets track independently with non-conflicting IDs (e.g. black T1 and blue T1)."""
    detector = DualDotDetector(stability_min_hits=3, max_missed_frames=2, min_area=20.0)
    canvas = np.full((480, 640, 3), 255, dtype=np.uint8)

    cv2.circle(canvas, (200, 200), 16, (0, 0, 0), -1)        # black dot
    cv2.circle(canvas, (400, 200), 16, (220, 50, 20), -1)    # blue dot

    # Frame 1: Both registered as target_id=1 within their respective class
    targets, _, _ = detector.detect(canvas)
    assert len(targets) == 2

    black_t = next(t for t in targets if t.class_name == "black_dot")
    blue_t = next(t for t in targets if t.class_name == "blue_dot")

    assert black_t.target_id == 1
    assert blue_t.target_id == 1
    assert black_t.is_stable is False
    assert blue_t.is_stable is False

    # Frame 2 & 3: Advance to stable
    detector.detect(canvas)
    targets, _, _ = detector.detect(canvas)

    black_t = next(t for t in targets if t.class_name == "black_dot")
    blue_t = next(t for t in targets if t.class_name == "blue_dot")
    assert black_t.is_stable is True
    assert blue_t.is_stable is True

    # Occlude blue dot only: black stays stable, blue registers miss
    black_only = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.circle(black_only, (200, 200), 16, (0, 0, 0), -1)

    targets, _, _ = detector.detect(black_only)
    black_t = next(t for t in targets if t.class_name == "black_dot")
    blue_t = next(t for t in targets if t.class_name == "blue_dot")

    assert black_t.misses == 0
    assert blue_t.misses == 1  # blue survives missed frame independently


def test_roi_filtering():
    """Verify that dots outside the configured Region of Interest (ROI) are ignored for both classes."""
    detector = DualDotDetector(roi=(100, 100, 500, 400), min_area=20.0)
    canvas = np.full((480, 640, 3), 255, dtype=np.uint8)

    # Inside ROI:
    cv2.circle(canvas, (200, 200), 15, (0, 0, 0), -1)        # black dot inside
    cv2.circle(canvas, (350, 200), 15, (220, 50, 20), -1)    # blue dot inside

    # Outside ROI:
    cv2.circle(canvas, (50, 50), 15, (0, 0, 0), -1)          # black dot outside
    cv2.circle(canvas, (580, 450), 15, (220, 50, 20), -1)    # blue dot outside

    targets, mask_black, mask_blue = detector.detect(canvas)
    assert len(targets) == 2
    detected_classes = {t.class_name for t in targets}
    assert detected_classes == {"black_dot", "blue_dot"}

    # Coordinates must correspond to the inside dots
    centers = {(t.center_x, t.center_y) for t in targets}
    assert any(abs(c[0] - 200) <= 2 and abs(c[1] - 200) <= 2 for c in centers)
    assert any(abs(c[0] - 350) <= 2 and abs(c[1] - 200) <= 2 for c in centers)


def test_proximity_deduplication_by_class():
    """Verify that close fragments of the same color merge, but black and blue dots remain distinct."""
    detector = DualDotDetector(merge_distance=25.0, min_area=15.0)
    canvas = np.full((480, 640, 3), 255, dtype=np.uint8)

    # Two black fragments very close (should merge into 1 black dot)
    cv2.circle(canvas, (200, 200), 8, (0, 0, 0), -1)
    cv2.circle(canvas, (212, 200), 8, (0, 0, 0), -1)

    # One blue dot nearby (within merge_distance, but different class -> must NOT merge)
    cv2.circle(canvas, (220, 200), 8, (220, 50, 20), -1)

    raw_dots, _, _ = detector.detect_raw_dots(canvas)
    assert len(raw_dots) == 2
    classes = [d.class_name for d in raw_dots]
    assert classes.count("black_dot") == 1
    assert classes.count("blue_dot") == 1


def test_dataclasses_and_formatting():
    """Verify class_name attribute on DotDetection and Target dataclasses and string outputs."""
    dot = DotDetection(
        class_name="blue_dot",
        center_x=100,
        center_y=150,
        area=350.0,
        box=[85, 135, 115, 165],
        x1=85,
        y1=135,
        x2=115,
        y2=165,
    )
    assert dot.class_name == "blue_dot"
    assert "class=blue_dot" in dot.to_debug_string()
    assert dot.to_dict()["class_name"] == "blue_dot"

    target = Target(
        class_name="black_dot",
        target_id=2,
        center_x=220,
        center_y=310,
        area=420.0,
        box=[205, 295, 235, 325],
        x1=205,
        y1=295,
        x2=235,
        y2=325,
        is_stable=True,
    )
    assert target.class_name == "black_dot"
    assert "class=black_dot" in target.to_debug_string()
    assert target.to_dict()["class_name"] == "black_dot"


def test_draw_detections_rendering():
    """Verify visualization renders both Red/White (black) and Blue boxes without error."""
    detector = DualDotDetector()
    canvas = np.full((480, 640, 3), 255, dtype=np.uint8)

    cv2.circle(canvas, (200, 200), 16, (0, 0, 0), -1)
    cv2.circle(canvas, (400, 200), 16, (220, 50, 20), -1)

    targets, _, _ = detector.detect(canvas)
    annotated = detector.draw_detections(canvas, targets)

    assert annotated.shape == (480, 640, 3)
    assert isinstance(annotated, np.ndarray)


def test_separate_masks_returned():
    """Verify get_masks returns two separate binary masks showing only pixels for each color."""
    detector = DualDotDetector()
    canvas = np.full((480, 640, 3), 255, dtype=np.uint8)

    # Black dot at (150, 150)
    cv2.circle(canvas, (150, 150), 15, (0, 0, 0), -1)
    # Blue dot at (400, 300)
    cv2.circle(canvas, (400, 300), 15, (220, 50, 20), -1)

    mask_black, mask_blue = detector.get_masks(canvas)

    assert mask_black.shape == (480, 640)
    assert mask_blue.shape == (480, 640)

    # Black mask must have active pixels at (150, 150), but zero at (400, 300)
    assert mask_black[150, 150] == 255
    assert mask_black[300, 400] == 0

    # Blue mask must have active pixels at (400, 300), but zero at (150, 150)
    assert mask_blue[300, 400] == 255
    assert mask_blue[150, 150] == 0
