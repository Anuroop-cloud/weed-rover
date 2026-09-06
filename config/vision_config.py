"""
Configuration settings for camera, detector, and display settings.
"""

# Camera Settings
CAMERA_DEVICE_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# YOLO Detector Settings
YOLO_MODEL_PATH = "yolov8n.pt"  # Can be replaced with "models/weed_best.pt"
CONFIDENCE_THRESHOLD = 0.35
DEVICE = None  # None for auto, or "cpu", "cuda:0"

# Target Classes (None means detect all; or specify list of IDs/names)
TARGET_CLASSES = None

# UI Visualization Settings
BOX_COLOR = (0, 220, 255)       # BGR yellow/gold
CENTROID_COLOR = (0, 0, 255)    # BGR red for crosshairs
CROSSHAIR_SIZE = 8
