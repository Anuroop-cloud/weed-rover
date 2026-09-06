"""
Black Dot Detector module for weed-rover.

Provides dot detection interface and classes, aliased from dual_dot_detector
for clean repository organization and modular vision architecture.
"""

from vision.dual_dot_detector import (
    DotDetection,
    Target,
    DualDotDetector,
)

# Standardized alias for black dot detection in the mock-farm rover pipeline
BlackDotDetector = DualDotDetector

__all__ = [
    "DotDetection",
    "Target",
    "DualDotDetector",
    "BlackDotDetector",
]
