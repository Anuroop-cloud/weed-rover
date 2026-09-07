"""
Configuration settings for camera, detector, and display settings.
"""

# Camera Settings
CAMERA_DEVICE_INDEX = 1  # 0: Built-in laptop webcam, 1: External USB camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30

# ==============================================================================
# BLACK DOT DETECTOR SETTINGS (Tune these for mock farm / grid floor)
# ==============================================================================
# In HSV: Hue: 0-180, Saturation: 0-255, Value/Brightness: 0-255
# Dark/Black regions are determined primarily by the upper Value (V) threshold.
BLACK_DOT_LOWER_HSV = (0, 0, 0)
BLACK_DOT_UPPER_HSV = (180, 255, 75)   # Increase V if dots appear brownish/grey; decrease if shadows get picked up
BLACK_DOT_MIN_AREA = 25.0              # Ignore speckle noise smaller than this pixel area
BLACK_DOT_MAX_AREA = 4000.0            # Ignore huge dark patches / frame borders larger than this
BLACK_DOT_MORPH_KERNEL = 3             # 3x3 kernel removes 1-2 pixel salt-and-pepper noise
MAX_ASPECT_RATIO = 3.5                 # Ignore thin cracks/lines where max(w/h, h/w) exceeds this

# BLUE DOT DETECTOR SETTINGS (HSV range for blue dots/markers)
BLUE_LOWER = (100, 50, 50)             # Lower HSV bound for blue hue (100-135)
BLUE_UPPER = (135, 255, 255)           # Upper HSV bound for blue hue
BLUE_DOT_LOWER_HSV = BLUE_LOWER
BLUE_DOT_UPPER_HSV = BLUE_UPPER

SHOW_MASK_DEBUG_WINDOW = True          # Display separate threshold mask windows for visual tuning

# Region of Interest (ROI) for mock farm / grid floor
# Format: (x1, y1, x2, y2) in pixels, or None to process the entire camera frame.
# Pixels outside this bounding rectangle are ignored to eliminate border shadows & edges.
FARM_ROI = (40, 30, 600, 450)

# Temporal Stability Settings (frame-to-frame persistence)
STABILITY_MIN_HITS = 3      # Consecutive frames a dot must be seen before it is deemed STABLE
MAX_MISSED_FRAMES = 3       # Number of frames a dot can be missed before being discarded
TARGET_MATCH_DISTANCE = 30.0  # Maximum pixel distance to match a dot between consecutive frames

# YOLO Detector Settings
YOLO_MODEL_PATH = "yolov8n.pt"  # Can be replaced with "models/weed_best.pt"
CONFIDENCE_THRESHOLD = 0.35
DEVICE = None  # None for auto, or "cpu", "cuda:0"

# Target Classes (None means detect all; or specify list of IDs/names)
TARGET_CLASSES = None

# Target Detector Settings (Red Marker / LED on Mock Farm Floor)
LOWER_RED1 = (0, 100, 100)
UPPER_RED1 = (10, 255, 255)
LOWER_RED2 = (160, 100, 100)
UPPER_RED2 = (179, 255, 255)
TARGET_MIN_AREA = 120.0
TARGET_MAX_AREA = None
MORPH_KERNEL_SIZE = 5

# UI Visualization Settings
BOX_COLOR = (0, 220, 255)       # BGR yellow/gold
CENTROID_COLOR = (0, 0, 255)    # BGR red for crosshairs
CROSSHAIR_SIZE = 8

# ==============================================================================
# ROBOT GEOMETRY & CAMERA MOUNTING (PROVISIONAL PLACEHOLDERS - NOT CALIBRATED)
# ==============================================================================
# NOTE: The physical robot design uses a TOP-DOWN camera mounted above the rover
# and pointed straight downward at the floor, viewing the floor as a 2D workspace.
# These values are INITIAL PROVISIONAL MATHEMATICAL PLACEHOLDERS for development.
# They are NOT calibrated and MUST be measured and tuned on the physical hardware.

CAMERA_HEIGHT_CM = 6.0              # Camera lens optical center height above ground (Z=0) in cm
CAMERA_TILT_DEG = 0.0               # DEPRECATED: Physical camera is top-down (pointing straight down).
                                    # Retained only for backwards compatibility with legacy callers.

CAMERA_RESOLUTION_WIDTH = 640       # Camera image width in pixels
CAMERA_RESOLUTION_HEIGHT = 480      # Camera image height in pixels

CAMERA_FOV_HORIZONTAL_DEG = 70.0    # Lens horizontal Field of View in degrees
CAMERA_FOV_VERTICAL_DEG = 55.0      # Lens vertical Field of View in degrees

LED_HEIGHT_CM = 10.0                # Height of LED pointer/tool above ground (Z=0)

# Physical separation between Camera and LED/Tool:
# Coordinate convention in Robot Ground Frame:
#   +Y = Robot forward
#   -Y = Robot backward
#   +X = Robot right
#   -X = Robot left
#
# Sign convention for offsets:
#   CAMERA_TO_LED_FORWARD_CM > 0: LED is mounted physically ahead (+Y) of camera.
#   CAMERA_TO_LED_LATERAL_CM > 0: LED is mounted physically to the right (+X) of camera.
#
# Transformation formula (camera-frame floor coords -> LED-frame floor coords):
#   X_led = X_camera - CAMERA_TO_LED_LATERAL_CM
#   Y_led = Y_camera - CAMERA_TO_LED_FORWARD_CM
#
# Example:
#   If the LED is physically 5 cm forward of the camera (CAMERA_TO_LED_FORWARD_CM = 5.0),
#   a point directly below the camera (X_cam=0, Y_cam=0) appears at Y_led = -5.0 cm
#   (5 cm behind the LED in LED coordinates).
CAMERA_TO_LED_FORWARD_CM = 5.0      # Forward distance from camera to LED along +Y in cm
CAMERA_TO_LED_LATERAL_CM = 0.0      # Lateral distance from camera to LED along +X in cm

TARGET_FINAL_DISTANCE_CM = 5.0      # Desired distance to target point in cm

# ==============================================================================
# LOCK-AND-EXECUTE STATE MACHINE & BLIND-SPOT CONTROLLER SETTINGS
# ==============================================================================
STATE_MACHINE_TARGET_CLASS = "weed"          # Class to track and engage ('weed' for new prototype)
ALIGNMENT_TOLERANCE_XY_CM = 1.5             # Target alignment tolerance in cm (stop when within this distance of LED)
CAMERA_BLIND_SPOT_ROW_PX = 430              # Pixel row (Y) above which dot enters camera lower blind spot
SIMULATED_APPROACH_SPEED_CM_S = 8.0         # Simulated rover speed toward target in cm/s for dead-reckoning
LED_FIRE_DURATION_SEC = 1.5                 # Duration to hold prototype LED firing signal in seconds
COMPLETED_TARGET_EXPIRY_SEC = 15.0          # Time to remember completed targets to prevent immediate re-targeting

# ==============================================================================
# PHYSICAL LED MATRIX / ACTUATION INTERFACE SETTINGS
# ==============================================================================
LED_MATRIX_WIDTH_CM = 3.2          # Total physical width of the 8x8 LED matrix in cm
LED_MATRIX_COLUMNS = 8             # Number of columns in the LED matrix (0 through 7)
LED_MATRIX_COLUMN_WIDTH_CM = LED_MATRIX_WIDTH_CM / LED_MATRIX_COLUMNS  # 0.4 cm per column

# ==============================================================================
# CROP (DOT) & WEED (X) DETECTOR SETTINGS (Physical/Demo Prototype)
# ==============================================================================
CROP_CLASS_NAME = "crop"
WEED_CLASS_NAME = "weed"

MARKER_LOWER_HSV = (0, 0, 0)
MARKER_UPPER_HSV = (180, 255, 85)           # Upper Value threshold for dark markers on light floor
MARKER_MIN_AREA = 25.0
MARKER_MAX_AREA = 8000.0

# Crop circle parameters (approximately LED-sized)
CROP_MIN_RADIUS = 3.0                       # Minimum radius in pixels
CROP_MAX_RADIUS = 40.0                      # Maximum radius in pixels
CROP_MIN_CIRCULARITY = 0.70                 # 4*pi*area / perimeter^2
CROP_MIN_SOLIDITY = 0.85                    # area / hull_area
CROP_MIN_CIRCLE_RATIO = 0.75                # area / enclosing circle area

# Weed X marker parameters
WEED_X_MAX_ASPECT = 1.85                    # Max aspect ratio max(w/h, h/w)
WEED_X_MAX_SOLIDITY = 0.72                  # Weed X markers have concave bays (solidity <= 0.72)
WEED_X_MAX_CIRCULARITY = 0.65               # Weed X markers have high perimeter (circularity < 0.65)
WEED_X_LINE_MIN_LEN_RATIO = 0.25            # Min line length relative to bbox size


