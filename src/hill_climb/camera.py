"""Camera interface for webcam capture."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from hill_climb.config import CameraConfig


class Camera:
    """Manages OpenCV webcam capture and release."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        """Open the camera device."""
        self.cap = cv2.VideoCapture(self.config.index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera index {self.config.index}. "
                "Check macOS Camera permissions in System Settings -> "
                "Privacy & Security."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Read a frame from the camera, optionally mirroring it."""
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        if self.config.mirror:
            frame = cv2.flip(frame, 1)

        return True, frame

    def release(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> Camera:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()
