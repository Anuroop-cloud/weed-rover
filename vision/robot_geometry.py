"""
Robot Geometry Module for weed-rover.

Top-Down Camera Floor Projection Model:
Converts 2D camera image coordinates (pixel_x, pixel_y) into an estimated 2D ground-plane
position (X_cm, Y_cm) relative to the rover, assuming a top-down camera looking straight
down at a flat floor workspace (ground plane Z = 0).

Coordinate Convention:
    Robot ground frame:
        +Y = Robot forward
        -Y = Robot backward
        +X = Robot right
        -X = Robot left
        +Z = Upward (ground plane is at Z = 0)

    Camera image convention:
        image TOP    = +Y (forward, v < cy)
        image BOTTOM = -Y (backward, v > cy)
        image RIGHT  = +X (right, u > cx)
        image LEFT   = -X (left, u < cx)
        image CENTER = (0, 0) (point directly below camera optical center)

Mathematical Model (Ideal Top-Down Pinhole Camera):
    Given:
        u, v   = target pixel coordinates (pixel_x, pixel_y)
        cx, cy = principal point (image_width / 2, image_height / 2)
        H      = camera height above floor (camera_height_cm)
        fx, fy = focal lengths in pixels

    Focal lengths derived from Field of View:
        fx = (image_width / 2.0) / tan(fov_horizontal / 2.0)
        fy = (image_height / 2.0) / tan(fov_vertical / 2.0)

    Ground coordinates in camera frame:
        X_camera = (u - cx) * H / fx
        Y_camera = (cy - v) * H / fy

Camera-to-LED / Tool Frame Transformation:
    The camera and tool/LED are physically separated by configurable offsets:
        CAMERA_TO_LED_FORWARD_CM: offset along robot forward (+Y) axis
        CAMERA_TO_LED_LATERAL_CM: offset along robot lateral (+X) axis

    Transformation formula:
        X_led = X_camera - CAMERA_TO_LED_LATERAL_CM
        Y_led = Y_camera - CAMERA_TO_LED_FORWARD_CM

    Example:
        If LED is physically 5 cm forward of the camera, a point directly below
        the camera (X_cam=0, Y_cam=0) has LED coordinates (0, -5.0 cm) — 5 cm behind the LED.

Future Calibration:
    Designed to accept an empirical 3x3 planar homography matrix H mapping
    [u, v, 1]^T -> [X_floor, Y_floor, 1]^T when physical calibration grid data is collected.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple

try:
    from config.vision_config import CAMERA_HEIGHT_CM
except ImportError:
    CAMERA_HEIGHT_CM = 6.0


@dataclass
class CalibrationPoint:
    """
    Ground-truth calibration pair for validating and tuning the geometry model.
    """
    pixel_x: float
    pixel_y: float
    known_x_cm: float
    known_y_cm: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "known_x_cm": self.known_x_cm,
            "known_y_cm": self.known_y_cm,
        }


class RobotGeometry:
    """
    Translates 2D image pixel coordinates into 2D ground-plane coordinates (X_cm, Y_cm)
    relative to the rover reference frame using an ideal top-down pinhole camera model.
    """

    def __init__(
        self,
        camera_height_cm: float = CAMERA_HEIGHT_CM,
        camera_tilt_deg: Optional[float] = None,
        image_width: int = 640,
        image_height: int = 480,
        fov_horizontal_deg: float = 70.0,
        fov_vertical_deg: float = 55.0,
        camera_to_led_forward_cm: float = 5.0,
        camera_to_led_lateral_cm: float = 0.0,
        led_height_cm: float = 10.0,
        max_ground_distance_cm: float = 1000.0,
    ):
        """
        Args:
            camera_height_cm: Camera lens optical center height above floor (Z=0).
            camera_tilt_deg: DEPRECATED. Physical camera is top-down (pointing straight down).
                             Accepted for backward compatibility with legacy callers; not used in math.
            image_width: Camera frame width in pixels.
            image_height: Camera frame height in pixels.
            fov_horizontal_deg: Lens horizontal Field of View in degrees.
            fov_vertical_deg: Lens vertical Field of View in degrees.
            camera_to_led_forward_cm: Offset from camera to LED along forward (+Y) axis.
            camera_to_led_lateral_cm: Offset from camera to LED along lateral (+X) axis.
            led_height_cm: Height of LED pointer/tool above ground.
            max_ground_distance_cm: Maximum realistic ground distance before discarding coordinates.
        """
        self.camera_height_cm = float(camera_height_cm)
        self.camera_tilt_deg = float(camera_tilt_deg) if camera_tilt_deg is not None else 0.0
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.fov_horizontal_deg = float(fov_horizontal_deg)
        self.fov_vertical_deg = float(fov_vertical_deg)
        self.camera_to_led_forward_cm = float(camera_to_led_forward_cm)
        self.camera_to_led_lateral_cm = float(camera_to_led_lateral_cm)
        self.led_height_cm = float(led_height_cm)
        self.max_ground_distance_cm = float(max_ground_distance_cm)

        # Future empirical planar homography matrix placeholder (3x3 matrix)
        self.homography_matrix: Optional[Any] = None

        # Precompute camera intrinsic parameters (cx, cy, fx, fy)
        self._update_intrinsics()

    def _update_intrinsics(self) -> None:
        """Computes principal point and focal lengths from resolution and FOV."""
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0

        half_fov_h = math.radians(self.fov_horizontal_deg / 2.0)
        half_fov_v = math.radians(self.fov_vertical_deg / 2.0)

        # Focal lengths in pixels (fx, fy)
        self.fx = (self.image_width / 2.0) / math.tan(half_fov_h) if half_fov_h > 0 else 1.0
        self.fy = (self.image_height / 2.0) / math.tan(half_fov_v) if half_fov_v > 0 else 1.0

    def set_homography(self, homography_matrix: Optional[Any]) -> None:
        """
        Sets a 3x3 planar homography matrix for empirical ground-plane calibration.
        When set to None, the default ideal top-down pinhole projection model is used.
        """
        self.homography_matrix = homography_matrix

    def pixel_to_ground(
        self, pixel_x: float, pixel_y: float
    ) -> Optional[Tuple[float, float]]:
        """
        Converts 2D image coordinates (pixel_x, pixel_y) to 2D ground-plane coordinates (X_cm, Y_cm)
        relative to the point on the floor directly below the camera optical center.

        Coordinate conventions:
            +Y = robot forward (image TOP: v < cy)
            -Y = robot backward (image BOTTOM: v > cy)
            +X = robot right (image RIGHT: u > cx)
            -X = robot left (image LEFT: u < cx)
            (0, 0) = image center (u = cx, v = cy)

        Formulas:
            X_camera = (u - cx) * H / fx
            Y_camera = (cy - v) * H / fy

        Args:
            pixel_x: Horizontal pixel position (u).
            pixel_y: Vertical pixel position (v).

        Returns:
            Tuple[float, float]: (X_cm, Y_cm) ground coordinates in cm,
            or None if input is invalid (NaN/Inf) or exceeds max_ground_distance_cm.
        """
        # Guard against non-numeric inputs
        if math.isnan(pixel_x) or math.isnan(pixel_y) or math.isinf(pixel_x) or math.isinf(pixel_y):
            return None

        if self.fx <= 0 or self.fy <= 0 or self.camera_height_cm <= 0:
            return None

        # Hook for future empirical planar homography calibration
        if self.homography_matrix is not None:
            raise NotImplementedError(
                "Empirical planar homography calibration is not yet implemented. "
                "Use the default top-down pinhole model (homography_matrix=None)."
            )

        u = float(pixel_x)
        v = float(pixel_y)

        # Ideal top-down pinhole projection
        x_cm = (u - self.cx) * self.camera_height_cm / self.fx
        y_cm = (self.cy - v) * self.camera_height_cm / self.fy

        # Check maximum ground distance boundary
        if math.hypot(x_cm, y_cm) > self.max_ground_distance_cm:
            return None

        return (round(float(x_cm), 2), round(float(y_cm), 2))

    def ground_point_from_pixel(
        self, pixel_x: float, pixel_y: float
    ) -> Optional[Tuple[float, float]]:
        """
        Alias for pixel_to_ground.
        Exposes ground point coordinates relative to the camera reference frame.
        """
        return self.pixel_to_ground(pixel_x, pixel_y)

    def ground_to_led_frame(
        self, x_robot_cm: float, y_robot_cm: float
    ) -> Tuple[float, float]:
        """
        Transforms a ground point from camera-frame floor coordinates
        to LED/tool-frame floor coordinates.

        Sign convention:
            X_led = X_camera - CAMERA_TO_LED_LATERAL_CM
            Y_led = Y_camera - CAMERA_TO_LED_FORWARD_CM

        Example:
            If the LED is physically 5 cm forward of the camera (CAMERA_TO_LED_FORWARD_CM = 5.0),
            a point directly below the camera (X_cam=0, Y_cam=0) appears at Y_led = -5.0 cm
            (5 cm behind the LED in LED coordinates).

        Args:
            x_robot_cm: Lateral distance in robot/camera reference frame (+X = right).
            y_robot_cm: Forward distance in robot/camera reference frame (+Y = forward).

        Returns:
            Tuple[float, float]: (X_led_cm, Y_led_cm) relative to the LED position.
        """
        x_led = x_robot_cm - self.camera_to_led_lateral_cm
        y_led = y_robot_cm - self.camera_to_led_forward_cm
        return (round(float(x_led), 2), round(float(y_led), 2))

    def pixel_to_led_ground(
        self, pixel_x: float, pixel_y: float
    ) -> Optional[Tuple[float, float]]:
        """
        Converts pixel coordinates directly to the ground point relative to the LED frame.
        """
        pt = self.pixel_to_ground(pixel_x, pixel_y)
        if pt is None:
            return None
        return self.ground_to_led_frame(pt[0], pt[1])

    def evaluate_calibration(
        self, points: List[CalibrationPoint]
    ) -> Dict[str, Any]:
        """
        Evaluates the current geometry model against a list of known ground-truth points.
        Provides metrics for validating or calibrating the model against physical measurements.

        Args:
            points: List of CalibrationPoint instances.

        Returns:
            Dict containing detailed per-point errors and summary statistics (MAE, RMSE, max error).
        """
        if not points:
            return {
                "total_points": 0,
                "valid_points": 0,
                "mae_x": 0.0,
                "mae_y": 0.0,
                "rmse_x": 0.0,
                "rmse_y": 0.0,
                "mean_distance_error": 0.0,
                "max_distance_error": 0.0,
                "evaluations": [],
            }

        evaluations = []
        errors_x = []
        errors_y = []
        dist_errors = []

        for p in points:
            pred = self.pixel_to_ground(p.pixel_x, p.pixel_y)
            if pred is None:
                evaluations.append({
                    "pixel": (p.pixel_x, p.pixel_y),
                    "known": (p.known_x_cm, p.known_y_cm),
                    "predicted": None,
                    "error_x": None,
                    "error_y": None,
                    "distance_error": None,
                    "valid": False,
                })
                continue

            pred_x, pred_y = pred
            err_x = pred_x - p.known_x_cm
            err_y = pred_y - p.known_y_cm
            dist_err = math.hypot(err_x, err_y)

            errors_x.append(err_x)
            errors_y.append(err_y)
            dist_errors.append(dist_err)

            evaluations.append({
                "pixel": (p.pixel_x, p.pixel_y),
                "known": (p.known_x_cm, p.known_y_cm),
                "predicted": (pred_x, pred_y),
                "error_x": round(err_x, 2),
                "error_y": round(err_y, 2),
                "distance_error": round(dist_err, 2),
                "valid": True,
            })

        valid_count = len(dist_errors)
        if valid_count == 0:
            return {
                "total_points": len(points),
                "valid_points": 0,
                "mae_x": None,
                "mae_y": None,
                "rmse_x": None,
                "rmse_y": None,
                "mean_distance_error": None,
                "max_distance_error": None,
                "evaluations": evaluations,
            }

        mae_x = sum(abs(e) for e in errors_x) / valid_count
        mae_y = sum(abs(e) for e in errors_y) / valid_count
        rmse_x = math.sqrt(sum(e ** 2 for e in errors_x) / valid_count)
        rmse_y = math.sqrt(sum(e ** 2 for e in errors_y) / valid_count)
        mean_dist_err = sum(dist_errors) / valid_count
        max_dist_err = max(dist_errors)

        return {
            "total_points": len(points),
            "valid_points": valid_count,
            "mae_x": round(mae_x, 2),
            "mae_y": round(mae_y, 2),
            "rmse_x": round(rmse_x, 2),
            "rmse_y": round(rmse_y, 2),
            "mean_distance_error": round(mean_dist_err, 2),
            "max_distance_error": round(max_dist_err, 2),
            "evaluations": evaluations,
        }
