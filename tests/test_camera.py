"""Unit tests for Camera capture interface and UI preview overlay."""

from typing import Any

import numpy as np
import pytest

from hill_climb.camera import Camera
from hill_climb.config import CameraConfig, UIConfig
from hill_climb.ui import UIOverlay


class MockVideoCaptureSuccess:
    """Simulates a successfully opened OpenCV VideoCapture device."""

    def __init__(self, index: int) -> None:
        self.index = index
        self._opened = True
        self.properties: dict[int, Any] = {}
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop_id: int, value: Any) -> bool:
        self.properties[prop_id] = value
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        # Return a simple 480x640 3-channel test image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Put an asymmetrical marker to verify horizontal flipping
        frame[0:50, 0:50] = 255
        return True, frame

    def release(self) -> None:
        self._opened = False
        self.released = True


class MockVideoCaptureFailure:
    """Simulates an inaccessible or denied camera device."""

    def __init__(self, index: int) -> None:
        self.index = index

    def isOpened(self) -> bool:
        return False

    def release(self) -> None:
        pass


def test_camera_start_success() -> None:
    cfg = CameraConfig(index=0, width=640, height=480, fps=30, mirror=False)
    camera = Camera(cfg, capture_factory=MockVideoCaptureSuccess)

    assert not camera.is_opened
    camera.start()
    assert camera.is_opened

    ret, frame = camera.read_frame()
    assert ret is True
    assert frame is not None
    assert frame.shape == (480, 640, 3)

    camera.release()
    assert not camera.is_opened


def test_camera_start_failure_raises_runtime_error() -> None:
    cfg = CameraConfig(index=1)
    camera = Camera(cfg, capture_factory=MockVideoCaptureFailure)

    with pytest.raises(RuntimeError, match="Check macOS Camera permissions"):
        camera.start()


def test_camera_read_before_start() -> None:
    cfg = CameraConfig(index=0)
    camera = Camera(cfg, capture_factory=MockVideoCaptureSuccess)

    ret, frame = camera.read_frame()
    assert ret is False
    assert frame is None


def test_camera_mirroring() -> None:
    cfg_unmirrored = CameraConfig(index=0, mirror=False)
    cam_unmirrored = Camera(cfg_unmirrored, capture_factory=MockVideoCaptureSuccess)
    cam_unmirrored.start()
    _, frame_normal = cam_unmirrored.read_frame()
    cam_unmirrored.release()

    cfg_mirrored = CameraConfig(index=0, mirror=True)
    cam_mirrored = Camera(cfg_mirrored, capture_factory=MockVideoCaptureSuccess)
    cam_mirrored.start()
    _, frame_mirrored = cam_mirrored.read_frame()
    cam_mirrored.release()

    assert frame_normal is not None and frame_mirrored is not None
    # Asymmetrical top-left pixel in unmirrored should be top-right in mirrored
    assert frame_normal[25, 25, 0] == 255
    assert frame_mirrored[25, 640 - 25, 0] == 255


def test_camera_context_manager() -> None:
    cfg = CameraConfig(index=0)
    with Camera(cfg, capture_factory=MockVideoCaptureSuccess) as camera:
        assert camera.is_opened
        ret, frame = camera.read_frame()
        assert ret is True
        assert frame is not None

    assert not camera.is_opened


def test_ui_draw_preview() -> None:
    ui = UIOverlay(UIConfig())
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered = ui.draw_preview(test_frame, fps=29.5, status="Testing")

    assert rendered.shape == test_frame.shape
    # Frame content should be modified by the HUD overlay
    assert not np.array_equal(rendered, test_frame)
