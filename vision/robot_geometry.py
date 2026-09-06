"""
Robot Geometry Module for weed-rover.

Converts 2D camera image coordinates (pixel_x, pixel_y) into an estimated 2D ground-plane
position (X_cm, Y_cm) relative to the rover, assuming a flat ground plane (Z = 0).

Coordinate Convention:
    +Y = Forward direction of the rover
    -Y = Behind the rover
    +X = Right direction of the rover
    -X = Left direction of the rover
    +Z = Upward (ground is at Z = 0)

Camera Reference Origin:
    The camera optical center is located at (X = 0, Y = 0, Z = CAMERA_HEIGHT_CM).
    The camera is tilted downward from the horizontal plane by CAMERA_TILT_DEG.

Mathematical Model:
    1. Pinhole Camera Model (Intrinsic parameters):
       Given image dimensions (W, H) and Field of View (FOV_h, FOV_v):
           c_x = W / 2.0,  c_y = H / 2.0
           f_x = (W / 2.0) / tan(FOV_h / 2.0)
           f_y = (H / 2.0) / tan(FOV_v / 2.0)

       For pixel (u, v):
           x_c = (u - c_x) / f_x   (normalized coordinate along camera X-axis)
           y_c = (v - c_y) / f_y   (normalized coordinate along camera Y-axis, pointing down)

    2. Camera-to-Rover Frame Rotation:
       The camera optical axis points forward and tilted downward by pitch angle theta.
       A 3D point (X_c, Y_c, Z_c) in camera frame relates to rover frame (X, Y, Z) by:
           X = X_c = x_c * Z_c
           Y = Z_c * cos(theta) - Y_c * sin(theta) = Z_c * (cos(theta) - y_c * sin(theta))
           Z = h - Z_c * sin(theta) - Y_c * cos(theta) = h - Z_c * (sin(theta) + y_c * cos(theta))

    3. Flat Ground Plane Intersection (Z = 0):
       Setting Z = 0 gives:
           Z_c = h / (sin(theta) + y_c * cos(theta))

       Valid intersection requires:
           sin(theta) + y_c * cos(theta) > 0  (ray points down towards ground)

       Substituting Z_c back yields the ground position:
           X_cm = (h * x_c) / (sin(theta) + y_c * cos(theta))
           Y_cm = h * (cos(theta) - y_c * sin(theta)) / (sin(theta) + y_c * cos(theta))

NOTE:
    The geometric parameters in config/vision_config.py are INITIAL MATHEMATICAL PLACEHOLDERS.
    They must be calibrated against measured ground positions on the physical robot.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple


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
    relative to the rover reference frame.
    """

    def __init__(
        self,
        camera_height_cm: float = 20.0,
        camera_tilt_deg: float = 35.0,
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
            camera_height_cm: Camera optical center height above ground (Z=0).
            camera_tilt_deg: Downward pitch angle of camera from horizontal in degrees.
            image_width: Camera frame width in pixels.
            image_height: Camera frame height in pixels.
            fov_horizontal_deg: Horizontal Field of View in degrees.
            fov_vertical_deg: Vertical Field of View in degrees.
            camera_to_led_forward_cm: Offset from camera to LED along forward (+Y) axis.
            camera_to_led_lateral_cm: Offset from camera to LED along lateral (+X) axis.
            led_height_cm: Height of LED pointer above ground.
            max_ground_distance_cm: Maximum realistic ground distance before discarding ray.
        """
        self.camera_height_cm = float(camera_height_cm)
        self.camera_tilt_deg = float(camera_tilt_deg)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.fov_horizontal_deg = float(fov_horizontal_deg)
        self.fov_vertical_deg = float(fov_vertical_deg)
        self.camera_to_led_forward_cm = float(camera_to_led_forward_cm)
        self.camera_to_led_lateral_cm = float(camera_to_led_lateral_cm)
        self.led_height_cm = float(led_height_cm)
        self.max_ground_distance_cm = float(max_ground_distance_cm)

        # Precompute camera intrinsic and rotation parameters
        self._update_intrinsics()

    def _update_intrinsics(self) -> None:
        """Computes principal point, focal lengths, and trigonometric terms."""
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0

        half_fov_h = math.radians(self.fov_horizontal_deg / 2.0)
        half_fov_v = math.radians(self.fov_vertical_deg / 2.0)

        # Focal lengths in pixels (fx, fy)
        self.fx = self.cx / math.tan(half_fov_h) if half_fov_h > 0 else 1.0
        self.fy = self.cy / math.tan(half_fov_v) if half_fov_v > 0 else 1.0

        # Camera tilt angle
        self.tilt_rad = math.radians(self.camera_tilt_deg)
        self.sin_tilt = math.sin(self.tilt_rad)
        self.cos_tilt = math.cos(self.tilt_rad)

    def pixel_to_ground(
        self, pixel_x: float, pixel_y: float
    ) -> Optional[Tuple[float, float]]:
        """
        Converts 2D image coordinates (pixel_x, pixel_y) to 2D ground-plane coordinates (X_cm, Y_cm)
        relative to the rover reference frame.

        Args:
            pixel_x: Horizontal pixel position (0 <= pixel_x < image_width).
            pixel_y: Vertical pixel position (0 <= pixel_y < image_height).

        Returns:
            Tuple[float, float]: (X_cm, Y_cm) where:
                X_cm: Lateral distance (+X is right, -X is left).
                Y_cm: Forward distance (+Y is forward in front of rover).
            Returns None if the pixel ray does not intersect the ground plane in front of the rover
            (e.g., ray points at or above the horizon, or beyond max_ground_distance_cm).
        """
        # Guard against non-numeric inputs
        if math.isnan(pixel_x) or math.isnan(pixel_y) or math.isinf(pixel_x) or math.isinf(pixel_y):
            return None

        # Normalized coordinates on unit-distance camera image plane
        # x_c > 0: right of optical center
        # y_c > 0: below optical center (closer to ground)
        x_c = (pixel_x - self.cx) / self.fx
        y_c = (pixel_y - self.cy) / self.fy

        # Denominator represents the vertical downward component of the ray:
        # denom = sin(theta) + y_c * cos(theta)
        denom = self.sin_tilt + (y_c * self.cos_tilt)

        # If denom <= 0, the ray is horizontal or pointing upward into the sky (above the horizon)
        # We require a small positive epsilon to prevent division by near-zero at the horizon
        if denom <= 1e-5:
            return None

        # Distance along camera optical axis to the ground plane (Z = 0)
        z_c = self.camera_height_cm / denom
        if z_c <= 0:
            return None

        # Forward ground distance (Y) and lateral ground distance (X)
        # Y = z_c * (cos(theta) - y_c * sin(theta))
        y_cm = z_c * (self.cos_tilt - (y_c * self.sin_tilt))
        x_cm = x_c * z_c

        # Reject points behind the camera or beyond max usable range
        if y_cm <= 0 or y_cm > self.max_ground_distance_cm:
            return None

        return (round(float(x_cm), 2), round(float(y_cm), 2))

    def ground_point_from_pixel(
        self, pixel_x: float, pixel_y: float
    ) -> Optional[Tuple[float, float]]:
        """
        Alias for pixel_to_ground.
        Exposes ground point coordinate relative to the camera/rover reference frame.
        """
        return self.pixel_to_ground(pixel_x, pixel_y)

    def ground_to_led_frame(
        self, x_robot_cm: float, y_robot_cm: float
    ) -> Tuple[float, float]:
        """
        Transforms a ground point from the camera/robot reference frame
        to the LED emitter reference frame.

        Args:
            x_robot_cm: Lateral distance in robot reference frame.
            y_robot_cm: Forward distance in robot reference frame.

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
