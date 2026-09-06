"""
Camera module for weed-rover.
Handles camera hardware capture, FPS measurement, and provides frames as standard NumPy arrays.
"""

import time
from typing import Optional, Tuple
import cv2
import numpy as np


class Camera:
    """
    Modular OpenCV Camera capture class.
    
    Attributes:
        device_id (int): Camera index (default: 0 for primary webcam).
        width (int): Target frame width in pixels.
        height (int): Target frame height in pixels.
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480
    ):
        self.device_id = device_id
        self.width = width
        self.height = height
        
        self.cap: Optional[cv2.VideoCapture] = None
        self._prev_time = time.perf_counter()
        self._fps = 0.0
        
        self._initialize_camera()

    def _initialize_camera(self) -> bool:
        """Initializes the VideoCapture instance with target settings."""
        print(f"[Camera] Initializing camera index {self.device_id} ({self.width}x{self.height})...")
        
        # On Windows, cv2.CAP_DSHOW can provide faster startup, but default backend is safe
        self.cap = cv2.VideoCapture(self.device_id)
        
        if not self.cap.isOpened():
            print(f"[Camera ERROR] Unable to open camera at index {self.device_id}.")
            return False

        # Set requested resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Read actual set resolution
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Successfully initialized. Active resolution: {actual_w}x{actual_h}")
        return True

    @property
    def is_opened(self) -> bool:
        """Check if camera device is opened and ready."""
        return self.cap is not None and self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads the next frame from the camera and calculates FPS.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success_flag, frame_as_numpy_array)
        """
        if not self.is_opened:
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("[Camera WARNING] Failed to grab frame from camera stream.")
            return False, None

        # Calculate FPS via moving timestamp
        current_time = time.perf_counter()
        delta = current_time - self._prev_time
        if delta > 0:
            self._fps = 1.0 / delta
        self._prev_time = current_time

        return True, frame

    def get_fps(self) -> float:
        """Returns the most recent calculated FPS."""
        return self._fps

    @staticmethod
    def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Draws an aesthetic FPS overlay badge on the top-left of the frame.
        """
        fps_text = f"FPS: {fps:.1f}"
        
        # Badge background
        cv2.rectangle(frame, (10, 10), (130, 42), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (130, 42), (0, 255, 0), 1)
        
        # FPS text
        cv2.putText(
            frame,
            fps_text,
            (18, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return frame

    def release(self) -> None:
        """Closes and releases camera hardware resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[Camera] Hardware resources released.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def run_camera_feed(device_id: int = 0, width: int = 640, height: int = 480):
    """
    Standalone runner to preview camera feed with live FPS display.
    Press 'q' or ESC to exit.
    """
    cam = Camera(device_id=device_id, width=width, height=height)
    if not cam.is_opened:
        print("[Camera ERROR] Camera could not be opened. Exiting.")
        return

    window_name = "Weed Rover - Camera Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("[Camera] Camera feed running. Press 'q' to exit.")

    try:
        while True:
            success, frame = cam.read()
            if not success or frame is None:
                print("[Camera WARNING] Dropped frame or camera disconnected.")
                break

            # Overlay FPS
            Camera.draw_fps(frame, cam.get_fps())

            # Display feed
            cv2.imshow(window_name, frame)

            # Check keypress: 1ms wait
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("[Camera] Exit requested by user.")
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_feed()
